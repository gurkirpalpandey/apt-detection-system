@echo off
echo ╔══════════════════════════════════════════╗
echo ║   APT Detection System — Starting...    ║
echo ╚══════════════════════════════════════════╝
call venv\Scripts\activate.bat
echo [OK] Dashboard running at: http://127.0.0.1:5000
echo [INFO] Press Ctrl+C to stop.
echo.
python app.py
pause
