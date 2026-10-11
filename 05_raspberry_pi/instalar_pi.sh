#!/usr/bin/env bash
# DERMATEC — instalación en Raspberry Pi 5 (una sola vez). Uso: bash pi/instalar_pi.sh
set -e
cd "$(dirname "$0")/.."
APP="$(pwd)"
echo "=== DERMATEC: instalación en $APP ==="

if [ "$(uname -m)" != "aarch64" ]; then
  echo "ERROR: este sistema no es de 64 bits ($(uname -m)). Grabe Raspberry Pi OS (64-bit) con Raspberry Pi Imager."
  exit 1
fi

echo "--- 1/4 Paquetes del sistema (pide la contraseña) ---"
sudo apt update
sudo apt install -y python3-venv python3-tk v4l-utils fonts-dejavu-core unzip
sudo usermod -aG video,input "$USER"   # cámara y botón SNAP (efectivo tras reiniciar)

echo "--- 2/4 Entorno de Python (.venv) ---"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r pi/requirements_pi.txt

echo "--- 3/4 Comprobación ---"
.venv/bin/python - <<'PY'
import sys, platform
sys.path.insert(0, "MODELO_DERMATEC_FINAL")
import tkinter, cv2, numpy, PIL
from ai_edge_litert.interpreter import Interpreter
from dermatec_inferencia import DermatecFinal
m = DermatecFinal()
print(f"OK · Python {platform.python_version()} · OpenCV {cv2.__version__} · NumPy {numpy.__version__} · modelos: {[x[0] for x in m.modelos]}")
PY

echo "--- 4/4 Acceso directo en el escritorio ---"
chmod +x pi/*.sh
DESK="[Desktop Entry]
Type=Application
Name=DERMATEC
Comment=Dermatoscopio electrónico DERMATEC
Exec=$APP/pi/ejecutar_dermatec.sh
Path=$APP
Icon=camera-photo
Terminal=false
Categories=Medical;Science;"
mkdir -p "$HOME/.local/share/applications" "$HOME/Desktop"
echo "$DESK" > "$HOME/.local/share/applications/dermatec.desktop"
echo "$DESK" > "$HOME/Desktop/DERMATEC.desktop"
chmod +x "$HOME/Desktop/DERMATEC.desktop"

echo
echo "=== Instalación terminada ==="
echo "Abrir:      doble clic en DERMATEC del escritorio, o:  bash pi/ejecutar_dermatec.sh"
echo "Cámara:     bash pi/probar_camara.sh"
echo "Benchmark:  .venv/bin/python herramientas/benchmark_pi.py CARPETA_CON_FOTOS --n 10"
