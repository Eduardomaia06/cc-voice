@echo off
cd /d "C:\Users\reged\dev\cc-voice"

:: Kill any lingering cc-claude.py processes from previous runs
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr "PID:"') do (
    wmic process where "ProcessId=%%i" get CommandLine 2>nul | findstr "cc-claude.py" >nul
    if not errorlevel 1 (
        echo Killing lingering process %%i...
        taskkill /f /pid %%i >nul 2>&1
    )
)

:: Activate venv and run
call venv\Scripts\activate
python cc-claude.py

:: Cleanup on exit - kill any remaining child processes
taskkill /f /im python.exe /fi "WINDOWTITLE eq *cc-claude*" >nul 2>&1
