@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. Run setup.ps1 first.
  exit /b 1
)
set PYTHONUTF8=1
set CUBLAS_WORKSPACE_CONFIG=:4096:8
".venv\Scripts\python.exe" "scripts\run.py" %*
exit /b %errorlevel%
