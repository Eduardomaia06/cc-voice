#!/usr/bin/env python3
"""
Voice to clipboard. Press Space to toggle recording. Paste with Ctrl+V.
"""

import tempfile
import wave
import os
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import keyboard

# Config
WHISPER_MODEL = "large-v3"
SAMPLE_RATE = 16000

# State
recording = False
audio_buffer = []
stream = None
model = None


def copy_to_clipboard(text):
    import subprocess
    subprocess.run(['clip'], input=text.encode('utf-16-le'), check=True)


def toggle_recording():
    global recording, audio_buffer, stream

    if recording:
        stop_recording()
        return

    audio_buffer = []
    recording = True

    def callback(indata, frames, time, status):
        if recording:
            audio_buffer.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, callback=callback)
    stream.start()
    print("Recording... (press Space to stop)")


def stop_recording():
    global recording, stream

    if not recording:
        return

    recording = False
    stream.stop()
    stream.close()

    if len(audio_buffer) < 10:
        print("Too short")
        return

    audio = np.concatenate(audio_buffer, axis=0)

    # Save temp wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        with wave.open(f.name, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())

    # Transcribe
    print("Transcribing...")
    segments, _ = model.transcribe(temp_path, beam_size=5, vad_filter=True)
    text = " ".join(s.text for s in segments).strip()
    os.unlink(temp_path)

    if text:
        copy_to_clipboard(text)
        print(f"Copied: {text}")
    else:
        print("No speech detected")


if __name__ == "__main__":
    print("Loading Whisper...")
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Ready! Press Space to start/stop recording.")
    print("Paste into Claude with Ctrl+V")
    print()

    keyboard.on_press_key('space', lambda e: toggle_recording())

    try:
        keyboard.wait('esc')  # Keep running until Escape
    except KeyboardInterrupt:
        print("Bye")
