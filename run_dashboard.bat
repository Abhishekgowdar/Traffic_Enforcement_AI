@echo off
REM Launch the Traffic Enforcement AI dashboard using the venv that has the full ML stack.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    echo Could not find .venv or venv with Python. Create one and install requirements first.
    exit /b 1
)

echo Starting dashboard at http://127.0.0.1:8000
"%PY%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
