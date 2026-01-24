# cc-voice

A hands-free voice interface for Claude Code. Speak your instructions, have them transcribed via Whisper (GPU-accelerated with CUDA), sent to Claude CLI, and see streaming responses with formatted thinking/tool usage.

## Features

- **Voice-to-Claude pipeline**: Speak naturally, get Claude responses
- **GPU-accelerated transcription**: Uses faster-whisper with CUDA for fast, accurate speech recognition
- **Streaming output**: See Claude's thinking, responses, and tool usage in real-time
- **Conversation continuity**: Maintains context across multiple voice inputs (uses `--continue`)
- **Push-to-talk interrupt**: Press Space anytime during Claude's response to interrupt and speak
- **Voice commands**: Say "clear context", "new session", or "reset" to start fresh
- **Project-aware**: Point cc-voice at any project folder to load its `CLAUDE.md` and `.mcp.json` settings
- **Interactive questions**: When Claude asks a question (AskUserQuestion), record your voice response

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3060)
- Microphone

### Software
- Windows 10/11 (also works on Linux with minor adjustments)
- Python 3.10+
- Claude CLI installed and in PATH (`npm install -g @anthropic-ai/claude-code`)
- CUDA Toolkit 12.x and cuDNN

## Installation

### 1. Clone and create virtual environment

```powershell
git clone <repo-url> cc-voice
cd cc-voice
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```powershell
pip install faster-whisper sounddevice numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Optional (for better Windows terminal color support):
```powershell
pip install colorama
```

### 3. Install CUDA Toolkit

1. Check your GPU compatibility at [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
2. Download CUDA Toolkit 12.x from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. Run the installer (Express installation is fine)
4. Verify installation:
   ```powershell
   nvcc --version
   ```

### 4. Install cuDNN

1. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
2. Extract and copy files to your CUDA installation directory:
   - `bin\*.dll` -> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
   - `include\*.h` -> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\`
   - `lib\x64\*.lib` -> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\`

### 5. Verify CUDA is working

```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Should print `CUDA available: True` and your GPU name.

### 6. Verify Claude CLI is installed

```powershell
claude --version
```

## Usage

### Basic usage (run from cc-voice directory)

```powershell
.\run.bat
```

Or with PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
python cc-claude.py
```

### Run for a specific project

You can point cc-voice at any project folder so Claude loads that project's `CLAUDE.md` and `.mcp.json` settings:

```powershell
.\run.bat C:\path\to\your\project
```

Or directly:
```powershell
python cc-claude.py --project C:\path\to\your\project
python cc-claude.py -C C:\path\to\your\project
```

### Portable launcher for any project

Copy `run-project.bat` into any project folder. When you run it (or create a shortcut to it), cc-voice starts with that folder as the working directory.

1. Copy `run-project.bat` to your project folder (e.g., `C:\Users\you\projects\myapp\`)
2. Double-click it or create a desktop shortcut to it
3. cc-voice starts and Claude will use that project's settings

This is the easiest way to use cc-voice with multiple projects.

## Controls

| Key | Action |
|-----|--------|
| **Space** | Toggle recording (press to start, press again to stop) |
| **Space** (during response) | Interrupt Claude and record new input |
| **Ctrl+C** | Exit the application |

## Voice Commands

Say these phrases to trigger special actions (instead of sending to Claude):

| Voice Command | Action |
|---------------|--------|
| "clear context" | Start a new conversation session |
| "clear the context" | Start a new conversation session |
| "new session" | Start a new conversation session |
| "start new session" | Start a new conversation session |
| "start over" | Start a new conversation session |
| "reset" | Start a new conversation session |
| "reset context" | Start a new conversation session |
| "reset session" | Start a new conversation session |

## How it works

1. Press **Space** to start recording your voice
2. Speak your instruction or question
3. Press **Space** to stop recording
4. Whisper transcribes your speech (GPU-accelerated via CUDA with float16 precision)
5. Transcription is sent to Claude CLI with `--continue` to maintain conversation context
6. Claude's response streams back with:
   - Thinking shown in dim cyan italic
   - Tool usage shown with progress indicators
   - Final response in bright white
   - Stats shown at end (duration, tool count, cost)
7. **Interrupt anytime**: Press Space during Claude's response to interrupt and speak
8. When Claude uses `AskUserQuestion`, you can record a voice response
9. Repeat from step 1 for follow-up messages

## Configuration

Edit `cc-claude.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `large-v3` | Whisper model size (see table below) |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `CLAUDE_TIMEOUT` | `3600` | Max seconds to wait for Claude (1 hour) |

### Whisper Model Comparison

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | ~1GB | Fastest | Low |
| base.en | ~1GB | Fast | Fair |
| small.en | ~2GB | Medium | Good |
| medium.en | ~3GB | Slower | Very Good |
| large-v3 | ~4GB | Slowest | Best |

For an RTX 3060 (12GB), `large-v3` runs comfortably with room to spare.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         cc-claude.py                            │
├─────────────────────────────────────────────────────────────────┤
│  1. Audio Recording (sounddevice)                               │
│     └─> Captures microphone input while Space is held          │
│                                                                 │
│  2. Speech-to-Text (faster-whisper + CUDA)                      │
│     └─> GPU-accelerated transcription with VAD filtering       │
│                                                                 │
│  3. Voice Command Detection                                     │
│     └─> Checks for "clear context", "reset", etc.              │
│                                                                 │
│  4. Claude Integration                                          │
│     └─> Sends to Claude CLI with --continue for context        │
│     └─> Streams JSON output for real-time display              │
│     └─> Handles tool use, thinking, and responses              │
│                                                                 │
│  5. Interactive Q&A                                             │
│     └─> When Claude asks questions, prompts for voice input    │
└─────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `cc-claude.py` | Main application - voice interface for Claude |
| `run.bat` | Windows launcher script (handles venv activation) |
| `run-project.bat` | Portable launcher to copy into project folders |
| `CLAUDE.md` | Instructions for Claude when working on this codebase |
| `piper/` | Piper TTS files (included but not currently used) |

## Troubleshooting

### "Claude CLI not found"
Make sure Claude is installed globally: `npm install -g @anthropic-ai/claude-code`

### CUDA errors
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU drivers are up to date
- Ensure cuDNN files are in the correct CUDA directories
- Make sure no other process is using all GPU memory

### No audio input
- Check your microphone is set as default recording device in Windows
- Try running as administrator
- Verify sounddevice sees your mic: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### Window closes immediately
If using `run-project.bat` and it closes immediately, there may be an error. The script includes a `pause` command so you can see any error messages.

### Transcription quality issues
- Try a larger Whisper model (e.g., `large-v3` instead of `small.en`)
- Ensure you have a good microphone
- Reduce background noise
- Speak clearly and at a moderate pace

### High GPU memory usage
- Use a smaller Whisper model (`small.en` uses ~2GB vs `large-v3` at ~4GB)
- Close other GPU-intensive applications
- The model is loaded once at startup and reused for efficiency

## License

MIT
