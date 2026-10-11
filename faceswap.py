#!/usr/bin/env python3
"""Lokaler Video-Faceswapper (insightface/inswapper_128 + Okklusions-Maske + zeitliche Glättung).

Nutzung:
    python faceswap.py --source gesicht.jpg --target video.mp4 --out ergebnis.mp4

Warum es stabil aussieht (auch beim Essen):
  1. Okklusions-Maske: Gabel/Hand/Essen vor dem Mund werden per Face-Parsing aus der
     Swap-Maske genommen und bleiben im Original sichtbar (optional, models/face_parser.onnx).
  2. Zeitliche Glättung: Landmarks werden über Frames geglättet -> kein Zittern/Flackern.
  3. Farbangleichung + weich ausgeblendete Maske -> keine harten Kanten.
  4. Optionaler Enhancer (models/gfpgan.onnx) gegen die 128px-Unschärfe.
"""
import argparse, os, subprocess, sys, tempfile
import cv2
import numpy as np
from tqdm import tqdm

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SWAPPER = os.path.join(MODELS, "inswapper_128.onnx")
PARSER = os.path.join(MODELS, "face_parser.onnx")      # BiSeNet (CelebAMask-HQ, 19 Klassen), optional
ENHANCER = os.path.join(MODELS, "gfpgan.onnx")         # GFPGAN 1.4 (512x512), optional

# CelebAMask-HQ Klassen, die zum Gesicht gehören (Haut, Brauen, Augen, Nase, Mund, Lippen)
FACE_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]


def providers():
    import onnxruntime as ort
    avail = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if p in avail]


class Smoother:
    """Exponentielle Glättung der 5 Gesichts-Keypoints (pro Track, nahe Gesichter = gleiche Person)."""
    def __init__(self, alpha):
        self.alpha, self.prev = alpha, None

    def __call__(self, kps):
        if self.prev is not None and np.linalg.norm(kps.mean(0) - self.prev.mean(0)) < 0.5 * np.ptp(kps[:, 0]) * 3:
            kps = self.alpha * kps + (1 - self.alpha) * self.prev
        self.prev = kps
        return kps


class Swapper:
    def __init__(self, args):
        import insightface
        from insightface.app import FaceAnalysis
        prov = providers()
        self.app = FaceAnalysis(name="buffalo_l", providers=prov)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.model = insightface.model_zoo.get_model(SWAPPER, providers=prov)
        self.parser = self._sess(PARSER, prov)
        self.enh = self._sess(ENHANCER, prov) if args.enhance else None
        self.smooth = {}
        self.args = args
        self.source = self._load_source(args.source)
        if not self.parser:
            print("Hinweis: models/face_parser.onnx fehlt -> keine Okklusions-Maske (Essen/Hände können überdeckt werden).")

    @staticmethod
    def _sess(path, prov):
        if not os.path.exists(path):
            return None
        import onnxruntime as ort
        return ort.InferenceSession(path, providers=prov)

    def _load_source(self, path):
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"Quellbild nicht lesbar: {path}")
        faces = self.app.get(img)
        if not faces:
            sys.exit("Kein Gesicht im Quellbild gefunden.")
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # --- Hilfsfunktionen -------------------------------------------------------------
    def _occlusion_mask(self, crop512):
        """Gesichts-Maske (0..1, 512x512) aus Face-Parsing; Essen/Hände/Besteck zählen nicht dazu."""
        x = cv2.resize(crop512, (512, 512))[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        out = self.parser.run(None, {self.parser.get_inputs()[0].name: x})[0][0]
        m = np.isin(out.argmax(0), FACE_CLASSES).astype(np.float32)
        return cv2.GaussianBlur(m, (0, 0), 3)

    def _enhance(self, crop512):
        x = crop512[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        y = self.enh.run(None, {self.enh.get_inputs()[0].name: x})[0][0]
        y = ((y.transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1) * 255)[:, :, ::-1]
        return y.astype(np.uint8)

    @staticmethod
    def _match_color(src, ref, mask):
        """Mittelwert/Streuung des Swaps an das Original anpassen (nur Maskenbereich)."""
        m = mask > 0.5
        if m.sum() < 100:
            return src
        s, r = src.astype(np.float32), ref.astype(np.float32)
        for c in range(3):
            sm, ss = s[..., c][m].mean(), s[..., c][m].std() + 1e-6
            rm, rs = r[..., c][m].mean(), r[..., c][m].std() + 1e-6
            s[..., c] = (s[..., c] - sm) * (rs / ss) * 0.5 + (s[..., c] - sm) * 0.5 + rm
        return s.clip(0, 255).astype(np.uint8)

    # --- Hauptlogik ------------------------------------------------------------------
    def process(self, frame):
        faces = self.app.get(frame)
        faces = sorted(faces, key=lambda f: f.bbox[0])
        if self.args.only_largest and faces:
            faces = [max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))]
        for i, face in enumerate(faces):
            face.kps = self.smooth.setdefault(i, Smoother(self.args.smooth))(face.kps.astype(np.float32))
            frame = self._swap_one(frame, face)
        return frame

    def _swap_one(self, frame, face):
        fake, M = self.model.get(frame, face, self.source, paste_back=False)   # 128px Crop + Matrix
        H, W = frame.shape[:2]
        size = 512
        # Crop-Koordinaten (128) -> 512
        S = np.diag([size / 128.0, size / 128.0, 1.0])
        M3 = np.vstack([M, [0, 0, 1]])
        M512 = (S @ M3)[:2]
        orig_crop = cv2.warpAffine(frame, M512, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        fake = cv2.resize(fake, (size, size), interpolation=cv2.INTER_CUBIC)
        if self.enh is not None:
            enh = self._enhance(fake)
            fake = cv2.addWeighted(enh, self.args.enhance_blend, fake, 1 - self.args.enhance_blend, 0)

        # Basismaske: weiche Ellipse
        base = np.zeros((size, size), np.float32)
        cv2.ellipse(base, (size // 2, size // 2 + 20), (int(size * 0.36), int(size * 0.44)), 0, 0, 360, 1, -1)
        base = cv2.GaussianBlur(base, (0, 0), size * 0.04)
        mask = base
        if self.parser is not None:
            mask = np.minimum(mask, self._occlusion_mask(orig_crop))
        fake = self._match_color(fake, orig_crop, mask)

        inv = cv2.invertAffineTransform(M512)
        fake_full = cv2.warpAffine(fake, inv, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        mask_full = cv2.warpAffine(mask, inv, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        out = frame.astype(np.float32) * (1 - mask_full) + fake_full.astype(np.float32) * mask_full
        return out.clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Lokaler Video-Faceswapper")
    p.add_argument("--source", required=True, help="Bild mit dem Gesicht, das eingesetzt wird")
    p.add_argument("--target", required=True, help="Zielvideo")
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--smooth", type=float, default=0.6, help="1.0 = keine Glättung, kleiner = stärker (Standard 0.6)")
    p.add_argument("--enhance", action="store_true", help="GFPGAN-Enhancer (models/gfpgan.onnx)")
    p.add_argument("--enhance-blend", type=float, default=0.7)
    p.add_argument("--only-largest", action="store_true", help="nur das größte Gesicht pro Frame tauschen")
    p.add_argument("--crf", type=int, default=16, help="x264-Qualität (kleiner = besser)")
    args = p.parse_args()

    if not os.path.exists(SWAPPER):
        sys.exit(f"Modell fehlt: {SWAPPER}\nSiehe README.md (Modelle herunterladen).")

    sw = Swapper(args)
    cap = cv2.VideoCapture(args.target)
    if not cap.isOpened():
        sys.exit(f"Video nicht lesbar: {args.target}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = tempfile.mkdtemp()
    silent = os.path.join(tmp, "silent.mp4")
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in tqdm(range(n), desc="Swap"):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(sw.process(frame))
    writer.release(); cap.release()

    # Ton aus Original übernehmen + sauber neu kodieren
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", args.target,
                    "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", str(args.crf),
                    "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", args.out], check=True)
    print("Fertig:", args.out)


if __name__ == "__main__":
    main()
