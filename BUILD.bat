@echo off
setlocal
cd /d "%~dp0"
echo TTS Studio: setup, download models, test and build.
echo First run needs Internet and about 20 GB free space; no admin required.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\bootstrap.ps1" %*
set "TTS_RESULT=%ERRORLEVEL%"
if not "%TTS_RESULT%"=="0" (echo BUILD FAILED. Read the error above.) else (echo DONE. Open VieNeuTTSStudio.exe after a full build.)
echo %* | findstr /I /C:"-NoPause" >nul
if errorlevel 1 pause
exit /b %TTS_RESULT%
