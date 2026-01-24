# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice-Claude (cc-voice) is a hands-free voice interface for Claude Code. It enables users to speak instructions, have them transcribed via Whisper, sent to the Claude CLI, and hear responses via Piper text-to-speech.

## Architecture

The application follows a simple pipeline in a single file (`cc-voice.py`):
1. **Audio Recording** - Uses `sounddevice` to capture microphone input until Enter is pressed
2. **Speech-to-Text** - Uses `faster-whisper` with CUDA/GPU acceleration to transcribe audio
3. **Claude Integration** - Sends transcribed text to `claude -p` CLI and captures response
4. **Text-to-Speech** - Uses local Piper TTS to speak Claude's response

## Running the Application

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run the voice interface
python cc-voice.py
```

## Dependencies

The project uses a Python virtual environment with:
- `faster-whisper` - Whisper speech recognition with CUDA support
- `sounddevice` - Audio recording/playback
- `numpy` - Audio data processing

External dependencies:
- **Piper TTS** - Located at `piper/piper/piper.exe` with voice model at `piper/voices/en_US-lessac-medium.onnx`
- **Claude CLI** - Must be installed and available in PATH

## Configuration

Key settings at the top of `cc-voice.py`:
- `WHISPER_MODEL` - Model size (tiny.en, base.en, small.en, medium.en)
- `SAMPLE_RATE` - Audio sample rate (default 16000)
- `PIPER_PATH` / `PIPER_VOICE` - Paths to Piper executable and voice model

## Key Implementation Details

- Audio is recorded to a numpy buffer and saved as a temporary WAV file for Whisper processing
- Minimum recording length is 0.5 seconds to avoid accidental triggers
- Code blocks in Claude responses are replaced with "[code block omitted]" for cleaner TTS
- The `claude -p` command has a 120-second timeout
