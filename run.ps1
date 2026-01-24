# Voice-Claude launcher
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot

try {
    # Activate venv and run
    & .\venv\Scripts\Activate.ps1
    python cc-voice.py
}
finally {
    Pop-Location
}
