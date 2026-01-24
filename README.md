# Voice-Claude (cc-voice)

Hands-free voice-to-clipboard tool. Speak, transcribe with Whisper AI (GPU-accelerated), paste anywhere. Designed for use with Claude Code or any application that accepts text input.

## Quick Start

```powershell
.\venv\Scripts\Activate.ps1
python cc-voice.py
```

## Usage

1. Press **Space** to start recording
2. Speak your message
3. Press **Space** to stop recording
4. Transcribed text is automatically copied to clipboard
5. Paste with **Ctrl+V** (into Claude Code, a text editor, etc.)
6. Press **Esc** to exit

## Requirements

- Python 3.x with virtual environment
- NVIDIA GPU with CUDA support (tested on RTX 3060)
- `faster-whisper` - Whisper speech recognition with CUDA
- `sounddevice` - Audio recording
- `numpy` - Audio processing
- `keyboard` - Hotkey detection

## CUDA Setup (NVIDIA GPU)

To enable GPU acceleration, you need CUDA and cuDNN installed:

### 1. Install CUDA Toolkit

1. Check your GPU compatibility at [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
2. Download CUDA Toolkit 12.x from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. Run the installer (Express installation is fine)
4. Verify installation:
   ```powershell
   nvcc --version
   ```

### 2. Install cuDNN

1. Download cuDNN from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
2. Extract and copy files to your CUDA installation directory:
   - `bin\*.dll` -> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\`
   - `include\*.h` -> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\`
   - `lib\x64\*.lib` -> `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\`

### 3. Install PyTorch with CUDA

```powershell
pip install torch torchvision torchaudio
```

### 4. Verify CUDA is working

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Should print `True`.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install faster-whisper sounddevice numpy keyboard
pip install torch
```

## Configuration

Edit `cc-voice.py` to change:
- `WHISPER_MODEL` - Model size (default: `large-v3` for best accuracy)
  - Options: `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`
- `SAMPLE_RATE` - Audio sample rate (default: 16000)

## Model Comparison

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | ~1GB | Fastest | Low |
| base.en | ~1GB | Fast | Fair |
| small.en | ~2GB | Medium | Good |
| medium.en | ~3GB | Slower | Very Good |
| large-v3 | ~4GB | Slowest | Best |

For an RTX 3060 (12GB), `large-v3` runs comfortably with room to spare.
