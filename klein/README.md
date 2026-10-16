# Klein 9B + BFS: Foto- und Video-Faceswap (Windows, Laufwerk D:)

Alles landet unter `D:\AI\` (ComfyUI, Python-Umgebung, Modelle, Hugging-Face-Cache). Anderer Ort: `setup_windows.ps1 -Root E:\AI` und Umgebungsvariable `AI_ROOT` setzen.

## 1. Einrichtung (einmalig)
Voraussetzungen: Python 3.11/3.12, Git, aktueller NVIDIA-Treiber, `winget install ffmpeg`.
```powershell
cd klein
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```
Danach FLUX.2 Klein 9B, den passenden Text-Encoder und die FLUX.2-VAE von den Modellseiten in die Ordner legen, die das Skript am Ende anzeigt. Die BFS-LoRA lädt das Skript selbst.

## 2. Workflow in ComfyUI bauen (einmalig)
```powershell
D:\AI\venv\Scripts\python.exe D:\AI\ComfyUI\main.py      # dann http://127.0.0.1:8188 öffnen
```
Klein-9B-Faceswap-Workflow mit BFS-LoRA aufbauen (Vorlagen gibt es auf civitai, Suche "BFS face swap klein"). LoRA-Stärke um 0.5 ist laut Community ein guter Start.
Diesen Nodes per Rechtsklick → Title einen Titel geben:

| Titel | Node |
|---|---|
| `SOURCE` | Load Image: das Gesicht, das eingesetzt wird |
| `TARGET` | Load Image: das Bild, in dem getauscht wird |
| `OUTPUT` | Save Image |
| `PROMPT` | optional: Text-Encode (z. B. Gesichtsausdruck beschreiben) |

Dann Menü → Export (API) → `workflow_api.json`.

## A) Foto tauschen
```powershell
D:\AI\venv\Scripts\python.exe comfy_faceswap.py --workflow workflow_api.json --source gesicht.jpg --target bild.jpg --out ergebnis.png
```
ComfyUI muss dabei laufen.

## B) Video tauschen
```powershell
# erst Maske prüfen (Gesicht rot markiert):
D:\AI\venv\Scripts\python.exe video_swap.py --video clip.mp4 --masks-only --out maske.mp4
# dann tauschen:
D:\AI\venv\Scripts\python.exe video_swap.py --video clip.mp4 --source gesicht.jpg --workflow workflow_api.json --out ergebnis.mp4
```
Ablauf: frontalstes Frame der Person wählen, mit Klein + BFS tauschen (Teil A), Wan-VACE überträgt das Gesicht vorwärts und rückwärts auf alle Frames, Einsetzen in Originalauflösung mit Ton.

Optionen: `--gender any|male|female` (Standard female), `--keyframe 120` (Frame selbst wählen), `--keyframe-image datei.png` (Schlüsselbild selbst getauscht, ohne ComfyUI), `--vace-model Wan-AI/Wan2.1-VACE-14B-diffusers` (besser, braucht viel mehr VRAM), `--low-vram`, `--max-side 640`.

**Hardware:** Klein 9B realistisch 16–24 GB VRAM; Wan-VACE 1.3B ca. 12–16 GB (mit `--low-vram` weniger). Fang mit 3–5 Sekunden Video an.

**Grenzen:** Die Identität ist im Schlüsselbild am besten und kann über lange Videos oder starke Kopfdrehungen etwas abweichen. Schnelle Bewegungen können an Abschnittsgrenzen sichtbar werden.
