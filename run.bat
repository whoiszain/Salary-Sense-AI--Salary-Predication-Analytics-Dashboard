@echo off
title SalarySense AI
cd /d "%~dp0"
echo ============================================
echo   SalarySense AI  -  starting
echo ============================================

if not exist venv\Scripts\python.exe (
  echo [1/3] Creating virtual environment...
  python -m venv venv
  if errorlevel 1 goto nopython
)

echo [2/3] Installing packages (first time: 1-3 minutes)...
venv\Scripts\python.exe -m pip install -q -r requirements.txt
if errorlevel 1 goto pipfail

if not exist models\salary_model.joblib (
  echo [3/3] Training the model (about 1 minute)...
  venv\Scripts\python.exe train.py
  if errorlevel 1 goto trainfail
)

echo.
echo Dashboard: http://localhost:8000   (press Ctrl+C to stop)
start "" cmd /c "timeout /t 4 >nul & start http://localhost:8000"
venv\Scripts\python.exe -m uvicorn app.main:app
goto end

:nopython
echo.
echo ERROR: Python was not found. Install Python 3.10+ from python.org
echo and tick "Add Python to PATH", then run this file again.
goto end
:pipfail
echo.
echo ERROR: package installation failed. Check your internet and run again.
goto end
:trainfail
echo.
echo ERROR: training failed. Copy the error above and send it for help.
:end
pause
