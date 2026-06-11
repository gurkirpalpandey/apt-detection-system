@echo off
echo ╔══════════════════════════════════════════╗
echo ║  APT Detection System — Windows Setup   ║
echo ╚══════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found.

:: Create virtual environment
if not exist "venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)
echo [OK] Virtual environment ready.

:: Activate venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo [SETUP] Installing dependencies (this may take 3-5 minutes)...
pip install flask flask-cors scikit-learn numpy pandas matplotlib seaborn --quiet
echo [OK] Core packages installed.

echo [SETUP] Installing TensorFlow CPU (this may take a few minutes)...
pip install tensorflow-cpu --quiet
echo [OK] TensorFlow installed.

:: Train models
echo.
echo [TRAIN] Training ML models on synthetic data...
python train.py
echo [OK] Models trained and saved.

echo.
echo ╔══════════════════════════════════════════╗
echo ║  Setup Complete!                         ║
echo ║  Run the app with: run.bat               ║
echo ╚══════════════════════════════════════════╝
pause
