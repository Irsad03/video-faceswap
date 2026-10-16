#!/usr/bin/env python3
"""Bessere Qualität: nutzt FaceFusion (3.x) mit dem 256-px-Swapper HyperSwap + GFPGAN-Enhancer + Okklusions-Maske.

Einrichtung (einmalig):
    git clone https://github.com/facefusion/facefusion && cd facefusion
    python install.py --onnxruntime cuda        # oder default / openvino / coreml
Modelle lädt FaceFusion beim ersten Lauf selbst herunter.

Nutzung:
    python facefusion_swap.py --facefusion-dir ../facefusion --source foto.jpg --target video.mp4 --out ergebnis.mp4
Die Flag-Namen können je nach FaceFusion-Version abweichen -> `--print-only` zeigt den Befehl, `python facefusion.py headless-run --help` die gültigen Flags.
"""
import argparse, os, subprocess, sys

p = argparse.ArgumentParser()
p.add_argument("--facefusion-dir", required=True)
p.add_argument("--source", nargs="+", required=True, help="ein oder mehrere Fotos")
p.add_argument("--target", required=True)
p.add_argument("--out", default="out.mp4")
p.add_argument("--gender", choices=["female", "male", "any"], default="female")
p.add_argument("--swapper", default="hyperswap_1a_256", help="z. B. hyperswap_1a_256, hyperswap_1b_256, hyperswap_1c_256, inswapper_128")
p.add_argument("--enhancer", default="gfpgan_1.4", help="z. B. gfpgan_1.4, codeformer, gfpgan_1.3")
p.add_argument("--enhancer-blend", type=int, default=70)
p.add_argument("--providers", nargs="+", default=["cuda"], help="cuda | cpu | coreml | openvino")
p.add_argument("--print-only", action="store_true")
a = p.parse_args()

cmd = [sys.executable, "facefusion.py", "headless-run",
       "--source-paths", *[os.path.abspath(s) for s in a.source],
       "--target-path", os.path.abspath(a.target),
       "--output-path", os.path.abspath(a.out),
       "--processors", "face_swapper", "face_enhancer",
       "--face-swapper-model", a.swapper,
       "--face-enhancer-model", a.enhancer,
       "--face-enhancer-blend", str(a.enhancer_blend),
       "--face-selector-mode", "many",
       "--face-mask-types", "occlusion",
       "--output-video-quality", "90",
       "--execution-providers", *a.providers]
if a.gender != "any":
    cmd += ["--face-selector-gender", a.gender]
print(" ".join(cmd))
if not a.print_only:
    subprocess.run(cmd, cwd=a.facefusion_dir, check=True)
