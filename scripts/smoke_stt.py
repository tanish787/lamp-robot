"""Load the real whisper model and transcribe a sample WAV file, printing
the transcript and latency. Run manually on the target hardware:
    python scripts/smoke_stt.py path/to/sample.wav
"""

import sys
import time
import wave

from brain.stt import SpeechToText


def main() -> None:
    path = sys.argv[1]
    with wave.open(path, "rb") as wav:
        audio = wav.readframes(wav.getnframes())
        sample_rate = wav.getframerate()

    stt = SpeechToText(model_size="tiny")
    start = time.monotonic()
    text = stt.transcribe(audio, sample_rate=sample_rate)
    elapsed = time.monotonic() - start

    print(f"Transcript: {text!r}")
    print(f"Latency: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
