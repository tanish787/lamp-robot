#!/usr/bin/env bash
# One-time setup: Python dependencies plus the local model weights.
#
# Be aware of what this does and does not do. The spec asks for pinned
# weights fetched once into models/ so nothing touches the network at
# runtime. That is true here for the Piper voice and the GGUF LLM, which
# are downloaded explicitly below to fixed URLs. It is NOT yet true for
# two of the four models:
#
#   * faster-whisper downloads its CTranslate2 weights from Hugging Face
#     on first use, into the HF cache (~/.cache/huggingface), not models/.
#   * ultralytics downloads yolov8n.pt on first use into the working
#     directory / its own cache.
#
# This script warms both of those caches so the *first run after setup* is
# offline, but they are cache-managed by their libraries rather than
# vendored into models/. Making them genuinely offline-first would mean
# passing explicit local paths to WhisperModel() and YOLO() — see
# KNOWN_LIMITATIONS.md.
#
#   ./scripts/setup.sh

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="models"
PIPER_DIR="$MODELS_DIR/piper"
LLM_DIR="$MODELS_DIR/llm"

PIPER_VOICE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL="$PIPER_VOICE_URL.json"
LLM_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

echo "==> Installing Python dependencies"
python -m pip install -r requirements.txt

mkdir -p "$PIPER_DIR" "$LLM_DIR"

fetch() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "    already present: $dest"
    return
  fi
  echo "    downloading $(basename "$dest")"
  curl -fL --progress-bar -o "$dest" "$url"
}

echo "==> Piper voice (TTS)"
fetch "$PIPER_VOICE_URL" "$PIPER_DIR/en_US.onnx"
fetch "$PIPER_CONFIG_URL" "$PIPER_DIR/en_US.onnx.json"

echo "==> Local LLM (GGUF)"
fetch "$LLM_URL" "$LLM_DIR/model.gguf"

echo "==> Warming the faster-whisper and YOLO caches"
echo "    (these libraries manage their own cache; this just does the"
echo "     first-use download now instead of mid-demo)"
python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("tiny", device="cpu", compute_type="int8")
print("    whisper tiny ready")

from ultralytics import YOLO
YOLO("yolov8n.pt")
print("    yolov8n ready")
PY

echo "==> Placeholder audio assets"
python -m scripts.generate_placeholder_audio

echo
echo "Setup complete. Run the demo with ./scripts/run_all.sh"
