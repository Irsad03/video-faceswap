# Lokaler Video-Faceswapper

Läuft komplett offline auf deinem Rechner (insightface + ONNX). Kein Upload, keine Cloud, keine Server-Filter.

## Setup
```bash
pip install -r requirements.txt          # GPU: onnxruntime-gpu statt onnxruntime
# ffmpeg muss installiert sein
```
Modelle in `models/` ablegen:

| Datei | Pflicht | Zweck |
|---|---|---|
| `inswapper_128.onnx` | ja | eigentlicher Swap |
| `face_parser.onnx` | empfohlen | Okklusions-Maske (Essen/Hand/Besteck vor dem Gesicht) |
| `gfpgan.onnx` | optional | schärfere Gesichter (`--enhance`) |

(`buffalo_l` für die Gesichtserkennung lädt insightface beim ersten Start selbst.)
Die drei Dateien findest du z. B. in den Model-Assets von FaceFusion/roop bzw. auf Hugging Face.

## Nutzung
```bash
python faceswap.py --source gesicht.jpg --target video.mp4 --out ergebnis.mp4 --enhance
```
Als Quelle geht auch ein **Video** der Person (`--source person.mp4`, auch gemischt mit Bildern): Das Tool sucht automatisch die schärfsten Frames pro Kopfwinkel heraus. Mehrere Bilder (oder ein Ordner) gehen auch: `--source ordner/` bzw. `--source a.jpg b.jpg c.jpg`. Das Tool wählt pro Frame automatisch passend zum Kopfwinkel.
Automatisch: Glättung, Überbrückung kurzer Erkennungsaussetzer (`--hold`), weiches Ausblenden bei extremen Winkeln (`--max-yaw`).

Geschlechtsfilter: Standard ist `--gender female` (nur als weiblich erkannte Personen werden getauscht, alle anderen bleiben unverändert). Mit `--gender any` oder `--gender male` änderst du das. Die Erkennung läuft über mehrere Frames per Mehrheitsentscheid und kann bei Kindern, starken Winkeln oder Verdeckung danebenliegen.
Ein einziges Quellfoto reicht (am besten scharf und frontal).

Tipps: `--smooth 0.4` bei zitterndem Gesicht, `--only-largest` bei Hintergrundpersonen, GPU ist ca. 20–50× schneller als CPU.

## Verantwortung
Nutze nur Gesichter von Personen, die zugestimmt haben. Fremde Personen ohne Einwilligung zu tauschen oder Fakes als echt auszugeben ist in vielen Ländern (u. a. DE: Persönlichkeitsrecht/KUG) strafbar.
