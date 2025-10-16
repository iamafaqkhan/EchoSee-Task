# EchoSee: Bilingual Real-Time Speech-to-Text (Whisper Tiny)
# ----------------------------------------------------------
# Supports English 🇺🇸 and Urdu 🇵🇰 in real-time.
# Live mic → Whisper Tiny (auto-detects language) → Socket.IO → Browser captions.

import queue
import threading
import sys
import io
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from flask import Flask, send_from_directory
import socketio

# -----------------------------
# CONFIGURATION
# -----------------------------
SAMPLE_RATE = 16000
BLOCKSIZE = 4000
MODEL_SIZE = "tiny"       # Can be "tiny", "base", "small", etc.
DEVICE = "cpu"            # Change to "cuda" if you have GPU
BYTES_PER_SAMPLE = 2       # int16 -> 2 bytes

# -----------------------------
# LOAD WHISPER MODEL
# -----------------------------
print("\n🔹 Initializing Whisper Tiny Model...\n")
model_path = Path(__file__).resolve().parent.parent / "models" / "whisper_tiny"

if model_path.exists():
    print(f"✅ Loading local Whisper Tiny model from: {model_path}")
    model = whisper.load_model(str(model_path), device=DEVICE)
else:
    print("⚠️ Local model not found. Downloading Whisper Tiny...")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

# -----------------------------
# AUDIO STREAM HANDLING
# -----------------------------
audio_queue = queue.Queue()
buffer = b""
BYTES_PER_SAMPLE = 2  # int16 -> 2 bytes

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))

def transcribe_stream():
    global buffer
    print("🎧 Starting bilingual real-time transcription... Speak now!\n")

    with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                           dtype='int16', channels=1, callback=audio_callback):
        while True:
            audio_chunk = audio_queue.get()
            buffer += audio_chunk

            # Process audio roughly every second (SAMPLE_RATE samples * bytes per sample)
            if len(buffer) >= SAMPLE_RATE * BYTES_PER_SAMPLE:
                try:
                    # Ensure we only convert whole int16 samples
                    bytes_available = len(buffer)
                    bytes_to_process = (bytes_available // BYTES_PER_SAMPLE) * BYTES_PER_SAMPLE
                    data_bytes = buffer[:bytes_to_process]
                    buffer = buffer[bytes_to_process:]

                    # Convert raw int16 PCM bytes to float32 numpy array in range [-1, 1]
                    audio_int16 = np.frombuffer(data_bytes, dtype=np.int16)
                    audio_data = audio_int16.astype(np.float32) / 32768.0

                    # 🧠 Auto-detect language (English or Urdu)
                    result = model.transcribe(audio_data, fp16=False, language=None)
                    text = result.get("text", "").strip()

                    if text:
                        print(f"🗣️ {text}")
                        sio.emit("transcription", {"text": text})
                except Exception as e:
                    print("Error:", e)

                    if text:
                        print(f"🗣️ {text}")
                        sio.emit("transcription", {"text": text})
                except Exception as e:
                    print("Error:", e)

# -----------------------------
# FLASK + SOCKET.IO SERVER
# -----------------------------
app = Flask(__name__)
sio = socketio.Server(cors_allowed_origins="*")
app.wsgi_app = socketio.WSGIApp(sio, app.wsgi_app)

# Serve index.html for captions page
@app.route('/')
def serve_index():
    return send_from_directory(Path(__file__).resolve().parent, "index.html")

@sio.event
def connect(sid, environ):
    print(f"📱 Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"❌ Client disconnected: {sid}")

# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    t = threading.Thread(target=transcribe_stream, daemon=True)
    t.start()

    print("\n🌐 Server running at http://localhost:5000 ...\n")
    app.run(host="0.0.0.0", port=5000, debug=False)