#!/usr/bin/env bash
# Lista las cámaras y toma una foto de prueba con la primera cámara USB
cd "$(dirname "$0")/.."
echo "=== Cámaras conectadas ==="
v4l2-ctl --list-devices
.venv/bin/python - <<'PY'
import re, time
from pathlib import Path
import cv2
usb = []
for n in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name))):
    if "usb" in str((n / "device").resolve()):
        usb.append(int(re.sub(r"\D", "", n.name)))
if not usb:
    raise SystemExit("No se encontró ninguna cámara USB. Revise el cable del capilógrafo.")
idx = usb[0]
cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(10):
    ok, f = cap.read(); time.sleep(0.05)
cap.release()
if not ok:
    raise SystemExit(f"/dev/video{idx} no entregó imagen.")
cv2.imwrite("prueba_camara.jpg", f)
print(f"OK: /dev/video{idx} · {f.shape[1]}×{f.shape[0]} · foto guardada en prueba_camara.jpg")
PY
