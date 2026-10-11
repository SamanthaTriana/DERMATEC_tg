@echo off
REM DERMATEC - resumen de tiempos, RAM y CPU registrados por la app (registros\RESUMEN_RENDIMIENTO.txt)
cd /d "%~dp0.."
.venv\Scripts\python.exe herramientas\resumen_rendimiento.py
pause
