# Video-Outfit-Editor (lokal)

Ersetzt Kleidung in bestehenden Videos per Text-Prompt, z. B. beim Tanzen die Jacke durch ein Hawaii-Hemd.

```bash
pip install -r requirements.txt     # PyTorch passend zu deiner GPU installieren
# ffmpeg muss installiert sein

# 1. Erst prüfen, ob die Jacke richtig erkannt wird (rot eingefärbt):
python outfit_edit.py --video tanz.mp4 --prompt x --region upper --masks-only --out maske.mp4

# 2. Dann editieren:
python outfit_edit.py --video tanz.mp4 --region upper \
  --prompt "a colorful Hawaiian shirt with palm leaves and hibiscus flowers" --out tanz_hawaii.mp4
```

Modelle (SegFormer + Wan2.1-VACE-1.3B) lädt Hugging Face beim ersten Start (mehrere GB). Danach läuft alles offline.

**Hardware:** NVIDIA-GPU mit ≥ 16 GB VRAM empfohlen. Weniger: `--low-vram` und `--max-side 640`. Ohne GPU nicht praktikabel.

**Regionen:** `upper` (Jacke/Oberteil), `pants`, `dress`, `skirt`, `scarf`, `hat`, `shoes`, `outfit`.

**Tipps**
- Prompt auf Englisch, konkret (Farbe, Muster, Stoff).
- Maske zu knapp/zu weit → `--grow`. Rand flackert → `--smooth 9`.
- Anderes Ergebnis → `--seed` ändern.
- Lange Videos werden in Abschnitte zerlegt; jeder knüpft an den vorigen an, damit das Hemd gleich bleibt. Kleine Abweichungen bei sehr schnellen Bewegungen sind möglich.
