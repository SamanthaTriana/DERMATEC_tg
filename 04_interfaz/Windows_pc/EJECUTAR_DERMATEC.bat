@echo off
REM DERMATEC App V6 - abrir la interfaz
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=..\DERMATEC_APP\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo No se encontro el entorno de Python. Ejecute primero INSTALAR_WINDOWS.bat
  pause
  exit /b
)
echo Usando: %PY%
"%PY%" dermatec_app_v6.py
echo.
echo Si la app no abrio, el error esta arriba. Envie una foto de esta ventana.
pause
