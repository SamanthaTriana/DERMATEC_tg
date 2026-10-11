"""
DERMATEC — Registro de rendimiento (Raspberry Pi 5 y PC)
========================================================
La app lo usa sola: no hay que hacer nada. Mientras la app está abierta escribe en la carpeta registros/:

  sesiones.csv               una fila al abrir y otra al cerrar la app (equipo, SO, RAM total, duración de la sesión)
  monitor_sistema.csv        cada 60 s: tiempo de app abierta, CPU %, RAM del sistema y de la app, temperatura,
                             frecuencia de la CPU, estado de throttling (Pi) y disco libre
  analisis_por_imagen.csv    cada imagen analizada: tiempo de cada etapa (calibración, Rama 1, parámetros,
                             clasificador) y total, RAM de la app, temperatura, resultado
  analisis_por_lesion.csv    cada diagnóstico (lote de N imágenes): carga del modelo, tiempo total,
                             tiempo por imagen, RAM pico, temperatura máxima, resultado general

Para el resumen (anexos L y H):  python herramientas/resumen_rendimiento.py
No guarda nombres de pacientes: solo el id de la lesión.
"""
import csv
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "registros"
INTERVALO_S = 60
ES_LINUX = sys.platform.startswith("linux")
ES_WIN = sys.platform.startswith("win")

_estado = {"inicio": None, "sesion": None, "hilo": None, "parar": threading.Event(), "cpu_prev": None,
           "n_analisis": 0, "n_imagenes": 0}


# ---------------------------------------------------------------- lecturas del sistema
def _leer(ruta):
    try:
        return Path(ruta).read_text().strip()
    except Exception:
        return ""


def ram_app_mb():
    """(RSS actual, pico) de este proceso en MB."""
    if ES_LINUX:
        act = pico = None
        for linea in _leer("/proc/self/status").splitlines():
            if linea.startswith("VmRSS:"):
                act = int(linea.split()[1]) / 1024.0
            elif linea.startswith("VmHWM:"):
                pico = int(linea.split()[1]) / 1024.0
        return act, pico
    if ES_WIN:
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                            ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.WorkingSetSize / 1048576.0, c.PeakWorkingSetSize / 1048576.0
        except Exception:
            pass
    return None, None


def ram_sistema_mb():
    """(usada, total) en MB."""
    if ES_LINUX:
        info = {}
        for linea in _leer("/proc/meminfo").splitlines():
            k, v = linea.split(":", 1)
            info[k] = int(v.split()[0]) / 1024.0
        tot = info.get("MemTotal")
        disp = info.get("MemAvailable")
        return (tot - disp if tot and disp else None), tot
    if ES_WIN:
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                            ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return (m.ullTotalPhys - m.ullAvailPhys) / 1048576.0, m.ullTotalPhys / 1048576.0
        except Exception:
            pass
    return None, None


def _tiempos_cpu():
    if ES_LINUX:
        v = [int(x) for x in _leer("/proc/stat").splitlines()[0].split()[1:]]
        ocioso = v[3] + (v[4] if len(v) > 4 else 0)
        return ocioso, sum(v)
    if ES_WIN:
        try:
            import ctypes
            from ctypes import wintypes
            i, k, u = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
            f = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa: E731
            return f(i), f(k) + f(u)          # el tiempo de kernel ya incluye el ocioso
        except Exception:
            pass
    return None


def cpu_pct():
    """CPU % del sistema desde la lectura anterior."""
    t = _tiempos_cpu()
    prev, _estado["cpu_prev"] = _estado["cpu_prev"], t
    if not t or not prev or t[1] == prev[1]:
        return None
    return round(100.0 * (1 - (t[0] - prev[0]) / (t[1] - prev[1])), 1)


def temperatura_c():
    t = _leer("/sys/class/thermal/thermal_zone0/temp")
    return round(int(t) / 1000.0, 1) if t.isdigit() else None


def frecuencia_mhz():
    f = _leer("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    return round(int(f) / 1000.0) if f.isdigit() else None


def throttling():
    """Raspberry Pi: 0x0 = sin limitación por temperatura ni por voltaje."""
    if not shutil.which("vcgencmd"):
        return ""
    try:
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True,
                              timeout=2).stdout.strip().split("=")[-1]
    except Exception:
        return ""


def disco_libre_gb():
    try:
        return round(shutil.disk_usage(str(RAIZ)).free / 1e9, 2)
    except Exception:
        return None


def equipo():
    modelo = _leer("/proc/device-tree/model").replace("\x00", "") if ES_LINUX else ""
    return modelo or platform.node() or "PC"


# ---------------------------------------------------------------- escritura
def _fila(nombre, datos):
    try:
        CARPETA.mkdir(exist_ok=True)
        ruta = CARPETA / nombre
        nuevo = not ruta.is_file()
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(datos))
            if nuevo:
                w.writeheader()
            w.writerow(datos)
    except Exception as exc:                       # el registro nunca debe tumbar la app
        print("registro_rendimiento:", repr(exc))


def _ahora():
    return datetime.now().isoformat(timespec="seconds")


def minutos_abierta():
    return round((time.time() - _estado["inicio"]) / 60.0, 2) if _estado["inicio"] else None


def muestra_sistema():
    usada, total = ram_sistema_mb()
    act, pico = ram_app_mb()
    r = lambda x: round(x, 1) if x is not None else ""  # noqa: E731
    return {"fecha": _ahora(), "sesion": _estado["sesion"], "app_abierta_min": minutos_abierta(),
            "cpu_pct": cpu_pct(), "ram_sistema_usada_mb": r(usada), "ram_sistema_total_mb": r(total),
            "ram_app_mb": r(act), "ram_app_pico_mb": r(pico), "temp_cpu_c": temperatura_c(),
            "frec_cpu_mhz": frecuencia_mhz(), "throttled": throttling(), "disco_libre_gb": disco_libre_gb()}


def _bucle():
    cpu_pct()
    while not _estado["parar"].wait(INTERVALO_S):
        _fila("monitor_sistema.csv", muestra_sistema())


def iniciar():
    """Llamar una vez al abrir la app."""
    if _estado["inicio"]:
        return
    _estado["inicio"] = time.time()
    _estado["sesion"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, total = ram_sistema_mb()
    _fila("sesiones.csv", {"sesion": _estado["sesion"], "evento": "inicio", "fecha": _ahora(), "equipo": equipo(),
                           "sistema": f"{platform.system()} {platform.release()} {platform.machine()}",
                           "python": platform.python_version(), "nucleos": os.cpu_count(),
                           "ram_total_mb": round(total) if total else "", "duracion_min": "",
                           "analisis": "", "imagenes": ""})
    _fila("monitor_sistema.csv", muestra_sistema())
    _estado["hilo"] = threading.Thread(target=_bucle, daemon=True)
    _estado["hilo"].start()


def cerrar():
    """Llamar al cerrar la app."""
    if not _estado["inicio"]:
        return
    _estado["parar"].set()
    _fila("monitor_sistema.csv", muestra_sistema())
    _fila("sesiones.csv", {"sesion": _estado["sesion"], "evento": "cierre", "fecha": _ahora(), "equipo": equipo(),
                           "sistema": "", "python": "", "nucleos": "", "ram_total_mb": "",
                           "duracion_min": minutos_abierta(), "analisis": _estado["n_analisis"],
                           "imagenes": _estado["n_imagenes"]})
    _estado["inicio"] = None


def imagen(lesion, nombre, ancho, alto, acople, tiempos, salida):
    """tiempos: dict con lectura_s, calibracion_s, rama1_s, parametros_s, clasificador_s, total_s."""
    _estado["n_imagenes"] += 1
    act, _ = ram_app_mb()
    t = {k: round(float(v), 3) for k, v in tiempos.items()}
    _fila("analisis_por_imagen.csv", {
        "fecha": _ahora(), "sesion": _estado["sesion"], "equipo": equipo(), "lesion": lesion, "imagen": nombre,
        "resolucion": f"{ancho}x{alto}", "acople": acople,
        "t_lectura_s": t.get("lectura_s", ""), "t_calibracion_color_s": t.get("calibracion_s", ""),
        "t_rama1_s": t.get("rama1_s", ""), "t_parametros_s": t.get("parametros_s", ""),
        "t_clasificador_s": t.get("clasificador_s", ""), "t_total_imagen_s": t.get("total_s", ""),
        "ram_app_mb": round(act, 1) if act else "", "temp_cpu_c": temperatura_c(),
        "resultado": salida.get("resultado_binario", ""), "clase": salida.get("clasificacion", ""),
        "p_maligno_pct": round(float(salida.get("probabilidad_malignidad", 0.0)), 1)})


def lesion(lesion_id, n_imagenes, t_carga_modelo_s, t_total_s, temp_max, general):
    _estado["n_analisis"] += 1
    act, pico = ram_app_mb()
    _fila("analisis_por_lesion.csv", {
        "fecha": _ahora(), "sesion": _estado["sesion"], "equipo": equipo(), "lesion": lesion_id,
        "n_imagenes": n_imagenes, "t_carga_modelo_s": round(t_carga_modelo_s, 3),
        "t_total_diagnostico_s": round(t_total_s, 3),
        "t_por_imagen_s": round((t_total_s - t_carga_modelo_s) / max(n_imagenes, 1), 3),
        "ram_app_mb": round(act, 1) if act else "", "ram_app_pico_mb": round(pico, 1) if pico else "",
        "temp_cpu_max_c": temp_max if temp_max is not None else "",
        "resultado": general.get("resultado_binario", ""), "clase": general.get("clasificacion", ""),
        "p_maligno_pct": round(float(general.get("probabilidad_malignidad", 0.0)), 1)})
