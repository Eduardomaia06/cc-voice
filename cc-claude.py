#!/usr/bin/env python3
"""
Voice to Claude. Press Space to toggle recording. Sends directly to Claude CLI.
"""

import tempfile
import wave
import os
import subprocess
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

# Config
WHISPER_MODEL = "large-v3"
SAMPLE_RATE = 16000
CLAUDE_TIMEOUT = 120  # seconds

# State
recording = False
audio_buffer = []
stream = None
model = None
is_first_message = True  # Track if this is the first message in the session


def send_to_claude(text):
    """Send text to Claude CLI and return response."""
    global is_first_message

    print(f"\n> {text}\n")
    print("Claude is thinking...")
    try:
        cmd = ['claude', '--dangerously-skip-permissions', '-p', text]
        if not is_first_message:
            cmd.insert(1, '--continue')  # Continue the conversation

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT
        )
        is_first_message = False  # After first successful call, use --continue
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[Claude timed out]"
    except FileNotFoundError:
        return "[Error: Claude CLI not found. Make sure 'claude' is in PATH]"


def start_recording():
    global recording, audio_buffer, stream

    if recording:
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
    segments, _ = model.transcribe(temp_path, beam_size=5, vad_filter=True, language="en")
    text = " ".join(s.text for s in segments).strip()
    os.unlink(temp_path)

    if text:
        response = send_to_claude(text)
        print(f"\n{response}\n")
        print("-" * 40)
        print("Ready! Press Space to record again.")
    else:
        print("No speech detected")


def toggle_recording():
    if recording:
        stop_recording()
    else:
        start_recording()


if __name__ == "__main__":
    print("Loading Whisper...")
    model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    print("Ready! Press Space to start/stop recording.")
    print("Your speech will be sent directly to Claude.")
    print("Conversation context is maintained until you close this script.")
    print()

    def on_press(key):
        if key == keyboard.Key.space:
            toggle_recording()

    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            print("Bye")
