@echo off
REM DERMATEC - evaluar una carpeta de imagenes sin abrir la app
REM Uso: arrastre la carpeta sobre este archivo, o: EVALUAR_CARPETA.bat "C:\CAPTURAS"
cd /d "%~dp0.."
.venv\Scripts\python.exe herramientas\evaluar_carpeta.py %1
pause
