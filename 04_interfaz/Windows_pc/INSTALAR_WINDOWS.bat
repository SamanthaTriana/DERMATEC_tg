@echo off
REM DERMATEC App V6 - instalacion en Windows (una sola vez)
cd /d "%~dp0"
if not exist .venv (
  py -3.13 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
echo Instalacion terminada. Abra la app con EJECUTAR_DERMATEC.bat
pause
