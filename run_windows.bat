@echo off
setlocal
cd /d "%~dp0"
call "%~dp0PLAY_WINDOWS.bat" %*
exit /b %ERRORLEVEL%
