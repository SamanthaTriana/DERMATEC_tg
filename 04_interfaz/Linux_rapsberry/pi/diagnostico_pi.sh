#!/usr/bin/env bash
# DERMATEC — diagnóstico de cámara y botón SNAP en la Raspberry. Uso: bash pi/diagnostico_pi.sh
cd "$(dirname "$0")/.."
echo "=== Sistema ==="; uname -m; echo "Usuario: $USER · grupos: $(id -nG)"
echo; echo "=== Cámaras ==="; v4l2-ctl --list-devices 2>/dev/null
.venv/bin/python - <<'PY'
import os, re, select, struct, subprocess, time
from pathlib import Path
import cv2
usb = [int(re.sub(r"\D", "", n.name)) for n in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name)))
       if "usb" in str((n / "device").resolve())]
if not usb:
    raise SystemExit("No hay cámara USB conectada.")
idx = usb[0]
print(f"\n=== Formatos de /dev/video{idx} ===")
print(subprocess.run(["v4l2-ctl", "-d", f"/dev/video{idx}", "--list-formats-ext"], capture_output=True, text=True).stdout[:2500])

cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(15):
    ok, f = cap.read(); time.sleep(0.05)
cap.release()
if ok:
    cv2.imwrite("diagnostico_captura.jpg", f)
    b, g, r = [float(f[..., i].mean()) for i in range(3)]
    print(f"Captura: {f.shape[1]}×{f.shape[0]} · media R {r:.0f} G {g:.0f} B {b:.0f} · guardada en diagnostico_captura.jpg")

print("\n=== Botón SNAP ===")
usb_dev = str(Path(f"/sys/class/video4linux/video{idx}/device").resolve().parent)
txt = Path("/proc/bus/input/devices").read_text()
rutas = []
for bl in txt.strip().split("\n\n"):
    n = re.search(r'N: Name="([^"]*)"', bl); s = re.search(r"S: Sysfs=(\S+)", bl); e = re.search(r"Handlers=.*?\b(event\d+)\b", bl)
    if n and e:
        print(f"  {e.group(1):8s} {n.group(1)}")
        if s and ("/sys" + s.group(1)).startswith(usb_dev):
            rutas.append(f"/dev/input/{e.group(1)}")
if not rutas:
    raise SystemExit("La cámara no publica un botón en /dev/input: el SNAP físico no se puede leer en Linux; use CAPTURAR o la tecla C.")
print("Botón de la cámara:", rutas)
try:
    fds = [os.open(r, os.O_RDONLY | os.O_NONBLOCK) for r in rutas]
except PermissionError:
    raise SystemExit("Sin permiso. Ejecute: sudo usermod -aG input $USER  y reinicie la Pi.")
print(">>> Presione el botón SNAP del capilógrafo en los próximos 15 s...")
fin = time.time() + 15
while time.time() < fin:
    listos, _, _ = select.select(fds, [], [], 0.5)
    for fd in listos:
        d = os.read(fd, 24 * 32)
        for k in range(0, len(d) - 23, 24):
            _, _, t, c, v = struct.unpack("llHHi", d[k:k + 24])
            if t == 1:
                print(f"  tecla código {c} valor {v}  → SNAP detectado")
                fin = 0
print("Fin del diagnóstico.")
PY
