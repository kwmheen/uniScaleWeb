@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 가상환경을 만든 뒤 패키지를 설치합니다.
  uv venv --python 3.11 .venv
  uv pip install -r requirements.txt
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:8765" ^| findstr LISTENING') do (
  taskkill /F /PID %%a >nul 2>&1
)
start "" cmd /c "timeout /t 1 /nobreak >nul & start http://127.0.0.1:8765/web/index.html"
".venv\Scripts\python.exe" serve.py
