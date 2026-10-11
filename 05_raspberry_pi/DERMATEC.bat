@echo off
chcp 65001 >nul
REM DERMATEC - doble clic para abrir. La primera vez instala todo (10-15 min, con internet).
cd /d "%~dp0"
if exist .venv\instalado.ok goto abrir
echo === DERMATEC: instalacion en %CD% ===
set "PYBASE="
for %%V in (3.12 3.11 3.13) do (
  if not defined PYBASE ( py -%%V -c "import sys" >nul 2>&1 && set "PYBASE=py -%%V" )
)
if not defined PYBASE ( python -c "import sys; assert (3,9)<=sys.version_info[:2]<=(3,13)" >nul 2>&1 && set "PYBASE=python" )
set "PY313=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYBASE if exist "%PY313%" set "PYBASE="%PY313%""
if defined PYBASE goto crear_entorno

echo.
echo === Instalando Python 3.13 (una sola vez, 1-3 min) ===
set "PYINST="
for %%F in (python-3.13*-amd64.exe) do set "PYINST=%%F"
if defined PYINST (
  "%PYINST%" /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_test=0
) else (
  winget install -e --id Python.Python.3.13 --scope user --silent --accept-package-agreements --accept-source-agreements
)
if exist "%PY313%" set "PYBASE="%PY313%""
if defined PYBASE goto crear_entorno
echo.
echo No se pudo instalar Python automaticamente.
echo Descargue e instale: https://www.python.org/ftp/python/3.13.16/python-3.13.16-amd64.exe
echo (marque "Add python.exe to PATH") y vuelva a abrir DERMATEC.bat
pause
exit /b 1

:crear_entorno
if not exist .venv\Scripts\python.exe %PYBASE% -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 ( echo. & echo La instalacion de librerias fallo. Revise la conexion a internet y repita. & pause & exit /b 1 )
.venv\Scripts\python.exe -c "import sys; sys.path[:0]=['MODELO_DERMATEC_FINAL','rama','calibracion','herramientas']; import cv2, numpy, PIL, tkinter, rama1_v2, parametros_lesion, escala_acople, calibracion_color, registro_rendimiento; from dermatec_inferencia import DermatecFinal; m=DermatecFinal(); print('OK - modelos:', [x[0] for x in m.modelos])"
if errorlevel 1 ( echo. & echo La comprobacion fallo. Envie una foto de esta ventana. & pause & exit /b 1 )
if not exist registros mkdir registros
if not exist DERMATEC_DATA mkdir DERMATEC_DATA
echo ok> .venv\instalado.ok
REM Acceso directo en el escritorio
powershell -NoProfile -Command "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'DERMATEC.lnk')); $s.TargetPath='%~dp0DERMATEC.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.IconLocation='%SystemRoot%\System32\shell32.dll,22'; $s.Save()"
echo.
echo === Instalacion terminada. Abriendo DERMATEC... ===

:abrir
if not exist registros mkdir registros
start "DERMATEC" /min cmd /c ".venv\Scripts\python.exe dermatec_app_v6.py > registros\ultima_ejecucion.log 2>&1 || (echo. & echo La app se cerro con error. Detalle en registros\ultima_ejecucion.log & type registros\ultima_ejecucion.log & pause)"
