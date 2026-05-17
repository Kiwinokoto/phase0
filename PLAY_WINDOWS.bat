@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_windows.ps1" %*
set ERR=%ERRORLEVEL%

if not "%ERR%"=="0" (
  echo.
  echo Launcher failed with error code %ERR%.
  echo.
  pause
)

exit /b %ERR%
