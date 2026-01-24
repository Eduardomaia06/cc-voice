# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cc-voice is a hands-free voice interface for Claude Code. It enables users to speak instructions, have them transcribed via Whisper (GPU-accelerated with CUDA), and sent to the Claude CLI with streaming responses.

## Architecture

The application follows a pipeline in a single file (`cc-claude.py`):

1. **Audio Recording** - Uses `sounddevice` to capture microphone input (toggle with Space key)
2. **Speech-to-Text** - Uses `faster-whisper` with CUDA/GPU acceleration (float16) to transcribe audio
3. **Voice Command Detection** - Checks for commands like "clear context", "reset", etc.
4. **Claude Integration** - Sends transcribed text to `claude -p` CLI with streaming JSON output
5. **Interactive Q&A** - When Claude uses AskUserQuestion, prompts for voice input

## Running the Application

```powershell
# Using the launcher script
.\run.bat

# Or manually
.\venv\Scripts\Activate.ps1
python cc-claude.py

# For a specific project
python cc-claude.py --project C:\path\to\project
python cc-claude.py -C C:\path\to\project
```

## Dependencies

Python packages (in virtual environment):
- `faster-whisper` - Whisper speech recognition with CUDA support
- `sounddevice` - Audio recording
- `numpy` - Audio data processing
- `torch` - For CUDA support
- `colorama` (optional) - Better Windows terminal colors

External dependencies:
- **Claude CLI** - Must be installed and available in PATH (`npm install -g @anthropic-ai/claude-code`)
- **CUDA Toolkit 12.x** - For GPU acceleration
- **cuDNN** - CUDA Deep Neural Network library

## Configuration

Key settings at the top of `cc-claude.py`:
- `WHISPER_MODEL` - Model size (default: `large-v3`; options: tiny.en, base.en, small.en, medium.en, large-v3)
- `SAMPLE_RATE` - Audio sample rate (default: 16000 Hz)
- `CLAUDE_TIMEOUT` - Max seconds to wait for Claude (default: 3600 = 1 hour)

## Key Implementation Details

- Audio is recorded to a numpy buffer and saved as a temporary WAV file for Whisper processing
- Whisper runs with `beam_size=5` and `vad_filter=True` for accuracy
- Space key toggles recording; Space during response interrupts Claude
- Voice commands ("clear context", "reset", etc.) clear the session without calling Claude
- Uses `--continue` flag to maintain conversation context across messages
- Streams Claude's JSON output for real-time display of thinking, tools, and responses
- Handles AskUserQuestion tool by prompting for voice input
- Cleanup on exit frees GPU memory via `torch.cuda.empty_cache()`

## Controls

| Key | Action |
|-----|--------|
| Space | Toggle recording |
| Space (during response) | Interrupt and record new input |
| Ctrl+C | Exit |

## Voice Commands

These phrases trigger session reset instead of being sent to Claude:
- "clear context" / "clear the context"
- "new session" / "start new session"
- "start over" / "reset" / "reset context" / "reset session"
