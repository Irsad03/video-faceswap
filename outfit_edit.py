#!/usr/bin/env python3
"""Kleidung in bestehenden Videos per Text-Prompt austauschen (lokal).

    python outfit_edit.py --video tanz.mp4 --region upper \
        --prompt "a colorful Hawaiian shirt with palm leaves and hibiscus flowers" --out tanz_hawaii.mp4

Pipeline:
  1. SegFormer (Kleidungs-Segmentierung) findet pro Frame z. B. die Jacke  -> Maske
  2. Maske wird über die Zeit geglättet und leicht vergrößert (kein Flackern an den Rändern)
  3. Wan2.1-VACE (Video-Inpainting, folgt dem Prompt) erzeugt NUR den maskierten Bereich neu,
     in überlappenden Abschnitten, wobei jeder Abschnitt am vorigen anknüpft (gleiches Hemd im ganzen Video)
  4. Ergebnis wird in voller Originalauflösung nur im Maskenbereich einkopiert, Ton bleibt erhalten
"""
import argparse, os, subprocess, sys, tempfile
import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# mattmdjaga/segformer_b2_clothes Labels
REGIONS = {
    "upper": [4],              # Oberteil / Jacke / Hemd
    "pants": [6],
    "dress": [7],
    "skirt": [5],
    "scarf": [17],
    "hat": [1],
    "shoes": [9, 10],
    "outfit": [4, 5, 6, 7, 17],
}
PROTECT = [2, 11, 12, 13, 14, 15]    # Haare, Gesicht, Beine, Arme: werden nie verändert
SEG_MODEL = "mattmdjaga/segformer_b2_clothes"
FIT_HINT = ", fitted to the body, same body shape and proportions, natural fabric folds"
VACE_MODEL = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"


# ---------- Video I/O ----------------------------------------------------------------
def read_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Video nicht lesbar: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
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


# ---------- Segmentierung + zeitliche Glättung ---------------------------------------
def segment(frames, labels, device, batch=8):
    """liefert (Ziel-Wahrscheinlichkeit, Körper-Wahrscheinlichkeit) pro Frame"""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()
    probs, body = [], []
    for i in tqdm(range(0, len(frames), batch), desc="Segmentierung"):
        chunk = frames[i:i + batch]
        inp = proc(images=[f[:, :, ::-1] for f in chunk], return_tensors="pt").to(device)
        with torch.no_grad():
            lg = model(**inp).logits
        H, W = chunk[0].shape[:2]
        p = torch.softmax(torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear"), 1)
        probs.append(p[:, labels].sum(1).cpu().numpy().astype(np.float16))
        body.append(p[:, PROTECT].sum(1).cpu().numpy().astype(np.float16))
    del model
    return np.concatenate(probs), np.concatenate(body)


def refine_masks(prob, body=None, smooth=5, grow=4, feather=7):
    """Zeitliches Mittel (gegen Flackern), Schwelle, Vergrößern (weite Hemden), weicher Rand."""
    p = uniform_filter1d(prob.astype(np.float32), size=max(1, smooth), axis=0, mode="nearest")
    bd = uniform_filter1d(body.astype(np.float32), size=max(1, smooth), axis=0, mode="nearest") if body is not None else None
    out = []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)) if grow > 0 else None
    for n, m in enumerate(p):
        b = (m > 0.4).astype(np.uint8)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # Löcher schließen
        if k is not None:
            b = cv2.dilate(b, k)
        if bd is not None:                                   # Körper (Arme, Gesicht, Beine) nie überschreiben
            b = b * (bd[n] < 0.5).astype(np.uint8)
        out.append(cv2.GaussianBlur(b.astype(np.float32), (0, 0), feather) if feather else b.astype(np.float32))
    return np.stack(out)


# ---------- Abschnitte planen --------------------------------------------------------
def plan_chunks(n, chunk=81, overlap=8):
    """[(start, end)], Länge je Abschnitt <= chunk; Folgeabschnitte beginnen `overlap` Frames früher."""
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


# ---------- Editieren ----------------------------------------------------------------
def edit_with_vace(frames_s, masks_s, args):
    import torch
    from PIL import Image
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    dev_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    vae = AutoencoderKLWan.from_pretrained(VACE_MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(VACE_MODEL, vae=vae, torch_dtype=dev_dtype)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload() if args.low_vram else pipe.to("cuda")
    h, w = frames_s[0].shape[:2]
    result = [None] * len(frames_s)
    prev_tail = None                                   # bereits editierte Frames des Vorgänger-Abschnitts
    for (s, e) in tqdm(plan_chunks(len(frames_s), args.chunk, args.overlap), desc="Editieren"):
        idx = list(range(s, e))
        # Länge auf 4m+1 bringen (ggf. letzten Frame auffüllen)
        real = len(idx)
        while len(idx) < valid_len(real):
            idx.append(idx[-1])
        vid, msk = [], []
        for j, i in enumerate(idx):
            if prev_tail is not None and j < args.overlap and result[i] is not None:
                vid.append(Image.fromarray(result[i]))                      # schon fertig -> beibehalten
                msk.append(Image.fromarray(np.zeros((h, w), np.uint8)))     # schwarz = nicht neu erzeugen
            else:
                vid.append(Image.fromarray(frames_s[i][:, :, ::-1]))
                msk.append(Image.fromarray((masks_s[i] > 0.5).astype(np.uint8) * 255))
        gen = torch.Generator("cpu").manual_seed(args.seed)
        out = pipe(video=vid, mask=msk, prompt=args.prompt + FIT_HINT, negative_prompt=args.negative,
                   height=h, width=w, num_frames=len(vid), num_inference_steps=args.steps,
                   guidance_scale=args.guidance, generator=gen).frames[0]
        for j, i in enumerate(idx):
            if j >= real or (prev_tail is not None and j < args.overlap and result[i] is not None):
                continue
            fr = np.asarray(out[j])
            fr = (fr * 255).clip(0, 255).astype(np.uint8) if fr.dtype != np.uint8 else fr
            result[i] = fr                                                   # RGB
        prev_tail = True
    return result


def composite(orig, edited_rgb, mask_s):
    """Edit nur im (weichen) Maskenbereich in Originalauflösung einsetzen."""
    H, W = orig.shape[:2]
    ed = cv2.resize(edited_rgb[:, :, ::-1], (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    m = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    return (orig.astype(np.float32) * (1 - m) + ed * m).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Kleidung in Videos per Prompt austauschen")
    p.add_argument("--video", required=True)
    p.add_argument("--prompt", required=True, help='z. B. "a colorful Hawaiian shirt with palm leaves" (Englisch klappt am besten)')
    p.add_argument("--region", choices=list(REGIONS), default="upper", help="was ersetzt wird (upper = Jacke/Oberteil)")
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--max-side", type=int, default=832, help="Arbeitsauflösung (längste Seite); höher = mehr VRAM")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=81, help="Frames pro Abschnitt")
    p.add_argument("--overlap", type=int, default=9)
    p.add_argument("--grow", type=int, default=4, help="Maske um N px vergrößern (Körper bleibt immer geschützt)")
    p.add_argument("--smooth", type=int, default=5, help="zeitliche Glättung der Maske (Frames)")
    p.add_argument("--negative", default="different body shape, wider shoulders, changed proportions, blurry, distorted, deformed, low quality, extra limbs, watermark")
    p.add_argument("--low-vram", action="store_true", help="CPU-Offload (langsamer, weniger VRAM)")
    p.add_argument("--masks-only", action="store_true", help="nur Masken als Vorschau-Video ausgeben (prüfen, ob die Jacke getroffen wird)")
    p.add_argument("--crf", type=int, default=16)
    args = p.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not args.masks_only:
        print("Warnung: keine GPU gefunden - das Editieren ist auf der CPU praktisch unbenutzbar langsam.")

    frames, fps = read_video(args.video)
    H, W = frames[0].shape[:2]
    ww, hh = work_size(W, H, args.max_side)
    frames_s = [cv2.resize(f, (ww, hh), interpolation=cv2.INTER_AREA) for f in frames]
    prob, body = segment(frames_s, REGIONS[args.region], dev)
    masks_s = refine_masks(prob, body, args.smooth, args.grow)
    cover = masks_s.mean()
    print(f"Maskenanteil im Mittel: {cover * 100:.1f}% des Bildes")
    if cover < 0.002:
        sys.exit("Fast nichts erkannt - ist die Kleidung im Video sichtbar? Andere --region probieren.")

    tmp = tempfile.mkdtemp()
    silent = os.path.join(tmp, "silent.mp4")
    wr = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if args.masks_only:
        for f, m in zip(frames, masks_s):
            mm = cv2.resize(m, (W, H))[..., None]
            wr.write((f * (1 - 0.6 * mm) + np.array([0, 0, 255]) * 0.6 * mm).astype(np.uint8))
    else:
        edited = edit_with_vace(frames_s, masks_s, args)
        for f, e, m in zip(frames, edited, masks_s):
            wr.write(composite(f, e, m))
    wr.release()

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", args.video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", str(args.crf), "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", args.out], check=True)
    print("Fertig:", args.out)


if __name__ == "__main__":
    main()
