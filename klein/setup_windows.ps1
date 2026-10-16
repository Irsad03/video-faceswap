# Einmalige Einrichtung auf Laufwerk D: (PowerShell, im Ordner klein\ ausführen):
#   powershell -ExecutionPolicy Bypass -File setup_windows.ps1
# Voraussetzungen: Python 3.11/3.12, Git, NVIDIA-Treiber. ffmpeg: winget install ffmpeg
param([string]$Root = "D:\AI")

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Root, "$Root\hf-cache", "$Root\insightface" | Out-Null
[Environment]::SetEnvironmentVariable("HF_HOME", "$Root\hf-cache", "User")   # alle HF-Downloads auf D:
$env:HF_HOME = "$Root\hf-cache"

# --- ComfyUI (für Teil A: Klein 9B + BFS-LoRA) ---
if (-not (Test-Path "$Root\ComfyUI")) { git clone https://github.com/comfyanonymous/ComfyUI "$Root\ComfyUI" }
if (-not (Test-Path "$Root\ComfyUI\custom_nodes\ComfyUI-Flux2Klein-Enhancer")) {
  git clone https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer "$Root\ComfyUI\custom_nodes\ComfyUI-Flux2Klein-Enhancer"
}

# --- eine gemeinsame Python-Umgebung auf D: ---
if (-not (Test-Path "$Root\venv")) { python -m venv "$Root\venv" }
$py = "$Root\venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r "$Root\ComfyUI\requirements.txt"
& $py -m pip install -r "$PSScriptRoot\requirements.txt"

# --- BFS-LoRA herunterladen ---
& $py -m pip install "huggingface_hub[cli]"
& "$Root\venv\Scripts\huggingface-cli.exe" download Alissonerdx/BFS-Best-Face-Swap --local-dir "$Root\ComfyUI\models\loras\BFS"

Write-Host ""
Write-Host "Fertig. Noch von Hand (Dateinamen siehe Modellseiten):"
Write-Host "  FLUX.2 Klein 9B   -> $Root\ComfyUI\models\diffusion_models\"
Write-Host "  Text-Encoder      -> $Root\ComfyUI\models\text_encoders\"
Write-Host "  FLUX.2 VAE        -> $Root\ComfyUI\models\vae\"
Write-Host "ComfyUI starten:  $py $Root\ComfyUI\main.py"
