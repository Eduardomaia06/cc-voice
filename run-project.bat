@echo off
:: Drop this script in any project folder to run cc-voice there
:: It uses the script's location as the project directory
set "PROJECT_PATH=%~dp0"
:: Remove trailing backslash
if "%PROJECT_PATH:~-1%"=="\" set "PROJECT_PATH=%PROJECT_PATH:~0,-1%"

call "C:\Users\reged\dev\cc-voice\run.bat" "%PROJECT_PATH%"
pause
