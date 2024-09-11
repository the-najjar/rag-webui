@echo off
if not exist .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt
set /p VARS=<.env
for /f "tokens=1,2 delims==" %%a in ("%VARS%") do (
    set %%a=%%b
)
python main.py
PAUSE
