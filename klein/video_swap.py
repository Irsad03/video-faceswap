#!/usr/bin/env python3
"""Teil B: Video-Faceswap in hoher Qualität.

  1. Wählt das frontalste Frame der Zielperson als Schlüsselbild.
  2. Tauscht dieses Frame mit FLUX.2 Klein 9B + BFS (Teil A, über ComfyUI).
  3. Wan-VACE überträgt das neue Gesicht vom Schlüsselbild auf alle anderen Frames, vorwärts und rückwärts.
     Nur der Gesichtsbereich wird neu erzeugt, und jeder Abschnitt knüpft am vorigen an.
  4. Das Gesicht wird in voller Originalauflösung eingesetzt, der Ton bleibt erhalten.

    python video_swap.py --video clip.mp4 --source gesicht.jpg --workflow workflow_api.json --out ergebnis.mp4
    # ohne ComfyUI, wenn du das Schlüsselbild schon selbst getauscht hast:
    python video_swap.py --video clip.mp4 --keyframe-image getauscht.png --keyframe 42 --out ergebnis.mp4
"""
import argparse, os, subprocess, sys, tempfile

# Alles auf D: (Modelle, Cache), sofern nicht anders gesetzt
ROOT = os.environ.get("AI_ROOT", r"D:\AI" if os.name == "nt" else os.path.expanduser("~/AI"))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

VACE_MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"   # besser, aber viel VRAM: Wan-AI/Wan2.1-VACE-14B-diffusers
DEFAULT_PROMPT = ("the same person, consistent face identity in every frame, natural skin texture with pores, "
                  "realistic lighting matching the scene, sharp detailed face")
DEFAULT_NEG = "blurry face, plastic skin, deformed face, flicker, different person, extra eyes, watermark"


# ---------- Video I/O ----------------------------------------------------------------
def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Video nicht lesbar: {path}")
    fps, frames = cap.get(cv2.CAP_PROP_FPS) or 25, []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        sys.exit("Video enthält keine Frames.")
    return frames, fps


def work_size(w, h, max_side):
    s = min(1.0, max_side / max(w, h))
    return max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)


# ---------- Gesichter finden + Masken ------------------------------------------------
def detect(frames, gender, providers):
    """Pro Frame: (bbox, yaw) der Zielperson oder None. Folgt der Person über Mittelpunkt-Abstand."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=os.path.join(ROOT, "insightface"), providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    out, prev = [], None
    for f in tqdm(frames, desc="Gesichter"):
        faces = app.get(f)
        if gender != "any":
            faces = [x for x in faces if (x.sex == "F") == (gender == "female")] or []
        if not faces:
            out.append(None)
            continue
        if prev is not None:
            c = lambda b: np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
            face = min(faces, key=lambda x: np.linalg.norm(c(x.bbox) - c(prev)))
        else:
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        prev = face.bbox
        out.append((face.bbox.astype(np.float32), float(face.pose[1])))
    return out


def face_masks(dets, shape, smooth=5, grow=1.25):
    """Weiche Ellipsen-Masken um das Gesicht (inkl. Stirn/Kinn), zeitlich geglättet; Lücken werden überbrückt."""
    h, w = shape
    boxes = [d[0] if d else None for d in dets]
    # fehlende Frames mit nächster bekannter Box füllen (max. 10 Frames)
    known = [i for i, b in enumerate(boxes) if b is not None]
    for i, b in enumerate(boxes):
        if b is None and known:
            j = min(known, key=lambda k: abs(k - i))
            if abs(j - i) <= 10:
                boxes[i] = boxes[j]
    arr = np.stack([b if b is not None else np.full(4, np.nan, np.float32) for b in boxes])
    valid = ~np.isnan(arr[:, 0])
    if valid.any():
        arr[valid] = uniform_filter1d(arr[valid], size=max(1, smooth), axis=0, mode="nearest")
    masks = np.zeros((len(boxes), h, w), np.float32)
    for i, b in enumerate(arr):
        if np.isnan(b[0]):
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2 - 0.05 * (b[3] - b[1])
        ax, ay = (b[2] - b[0]) / 2 * grow, (b[3] - b[1]) / 2 * grow * 1.1
        cv2.ellipse(masks[i], (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 1.0, -1)
        masks[i] = cv2.GaussianBlur(masks[i], (0, 0), max(1.0, ax * 0.12))
    return masks


def pick_keyframe(dets):
    cands = [(abs(d[1]) - 1e-4 * (d[0][2] - d[0][0]), i) for i, d in enumerate(dets) if d]
    if not cands:
        sys.exit("Keine passende Person im Video gefunden (Geschlechtsfilter? --gender any probieren).")
    return min(cands)[1]


# ---------- Wan-VACE-Weitergabe -------------------------------------------------------
def plan_chunks(n, chunk=81, overlap=9):
    spans, s = [], 0
    while True:
        e = min(n, s + chunk)
        spans.append((s, e))
        if e >= n:
            return spans
        s = e - overlap


def valid_len(k):
    """Wan braucht 4m+1 Frames (aufrunden, fehlende werden aufgefüllt)."""
    return max(5, -(-(k - 1) // 4) * 4 + 1)


def load_vace(args):
    import torch
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.vace_model, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(args.vace_model, vae=vae, torch_dtype=dt)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload() if args.low_vram else pipe.to("cuda")
    return pipe


def propagate(pipe, frames_rgb, masks, key_rgb, ref_img, args):
    """frames_rgb[0] ist das Schlüsselbild; liefert editierte RGB-Frames in derselben Reihenfolge."""
    import torch
    from PIL import Image
    n = len(frames_rgb)
    h, w = frames_rgb[0].shape[:2]
    result = [None] * n
    result[0] = key_rgb
    zero = Image.fromarray(np.zeros((h, w), np.uint8))
    for (s, e) in tqdm(plan_chunks(n, args.chunk, args.overlap), desc="Wan-VACE", leave=False):
        idx = list(range(s, e))
        real = len(idx)
        while len(idx) < valid_len(real):
            idx.append(idx[-1])
        vid, msk = [], []
        for j, i in enumerate(idx):
            if j < real and result[i] is not None:          # schon fertig (Schlüsselbild / Überlappung) -> festhalten
                vid.append(Image.fromarray(result[i])); msk.append(zero)
            else:
                vid.append(Image.fromarray(frames_rgb[i]))
                msk.append(Image.fromarray(((masks[i] > 0.3) * 255).astype(np.uint8)))
        kw = dict(video=vid, mask=msk, prompt=args.prompt, negative_prompt=args.negative, height=h, width=w,
                  num_frames=len(vid), num_inference_steps=args.steps, guidance_scale=args.guidance,
                  generator=torch.Generator("cpu").manual_seed(args.seed))
        try:
            out = pipe(reference_images=[ref_img], **kw).frames[0]
        except TypeError:                                   # ältere diffusers ohne reference_images
            out = pipe(**kw).frames[0]
        for j, i in enumerate(idx[:real]):
            if result[i] is None:
                fr = np.asarray(out[j])
                result[i] = (fr * 255).clip(0, 255).astype(np.uint8) if fr.dtype != np.uint8 else fr
    return result


def reference_from(key_rgb, mask):
    """Gesicht aus dem Schlüsselbild auf weißem Grund (VACE-Referenzformat)."""
    from PIL import Image
    ys, xs = np.where(mask > 0.3)
    if len(xs) == 0:
        return Image.fromarray(key_rgb)
    crop = key_rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1, None]
    white = np.full_like(crop, 255)
    return Image.fromarray((crop * m + white * (1 - m)).astype(np.uint8))


def composite(orig, edited_rgb, mask_s):
    H, W = orig.shape[:2]
    ed = cv2.resize(edited_rgb[:, :, ::-1], (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    m = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    return (orig.astype(np.float32) * (1 - m) + ed * m).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Video-Faceswap: Klein 9B + BFS fürs Schlüsselbild, Wan-VACE für den Rest")
    p.add_argument("--video", required=True)
    p.add_argument("--source", help="Foto des Gesichts, das eingesetzt wird (für ComfyUI)")
    p.add_argument("--workflow", help="ComfyUI-Workflow (API-Export, siehe comfy_faceswap.py)")
    p.add_argument("--keyframe-image", help="bereits getauschtes Schlüsselbild (überspringt ComfyUI)")
    p.add_argument("--keyframe", type=int, help="Index des Schlüsselbilds (Standard: frontalstes Frame)")
    p.add_argument("--server", default="http://127.0.0.1:8188")
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--gender", choices=["female", "male", "any"], default="female")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default=DEFAULT_NEG)
    p.add_argument("--vace-model", default=VACE_MODEL)
    p.add_argument("--max-side", type=int, default=832)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=81)
    p.add_argument("--overlap", type=int, default=9)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--masks-only", action="store_true", help="nur Gesichtsmasken als Vorschau ausgeben")
    p.add_argument("--crf", type=int, default=16)
    args = p.parse_args()
    if not (args.masks_only or args.keyframe_image or (args.source and args.workflow)):
        sys.exit("Entweder --source + --workflow (ComfyUI) oder --keyframe-image angeben.")

    import onnxruntime as ort
    prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]

    frames, fps = read_video(args.video)
    H, W = frames[0].shape[:2]
    ww, hh = work_size(W, H, args.max_side)
    small = [cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA) for f in frames]
    dets = detect(small, args.gender, prov)
    masks = face_masks(dets, (hh, ww))
    k = args.keyframe if args.keyframe is not None else pick_keyframe(dets)
    print(f"Schlüsselbild: Frame {k}")

    tmp = tempfile.mkdtemp(dir=os.path.join(ROOT, "tmp") if os.path.isdir(os.path.join(ROOT, "tmp")) else None)
    silent = os.path.join(tmp, "silent.mp4")
    wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if args.masks_only:
        for f, m in zip(frames, masks):
            mm = cv2.resize(m, (W, H))[..., None]
            wr.write((f * (1 - 0.6 * mm) + np.array([0, 0, 255]) * 0.6 * mm).astype(np.uint8))
    else:
        # 1) Schlüsselbild tauschen
        if args.keyframe_image:
            key_path = args.keyframe_image
        else:
            from comfy_faceswap import swap
            src_frame = os.path.join(tmp, f"frame_{k}.png")
            cv2.imwrite(src_frame, frames[k])
            key_path = swap(args.workflow, args.source, src_frame, os.path.join(tmp, "key_swapped.png"), args.server)
        key = cv2.imread(key_path)
        if key is None:
            sys.exit(f"Schlüsselbild nicht lesbar: {key_path}")
        key_small = cv2.resize(key, (ww, hh), interpolation=cv2.INTER_AREA)
        # Nur das Gesicht aus dem Klein-Ergebnis übernehmen, Rest bleibt Original
        m = masks[k][..., None]
        key_small = (small[k] * (1 - m) + key_small * m).astype(np.uint8)
        key_rgb = key_small[:, :, ::-1].copy()
        ref = reference_from(key_rgb, masks[k])

        # 2) vorwärts (k..Ende) und rückwärts (k..0) weitergeben
        pipe = load_vace(args)
        rgb = [f[:, :, ::-1].copy() for f in small]
        edited = [None] * len(frames)
        fwd = propagate(pipe, rgb[k:], masks[k:], key_rgb, ref, args)
        for i, fr in enumerate(fwd):
            edited[k + i] = fr
        if k > 0:
            bwd = propagate(pipe, rgb[k::-1], masks[k::-1], key_rgb, ref, args)
            for i, fr in enumerate(bwd):
                edited[k - i] = fr
        for f, e, msk in zip(frames, edited, masks):
            wr.write(composite(f, e, msk))
    wr.release()

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", args.video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", args.out],
                   check=True)
    print("Fertig:", args.out)


if __name__ == "__main__":
    main()
