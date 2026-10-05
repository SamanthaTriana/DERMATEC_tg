"""
DERMATEC — Benchmark de la Raspberry Pi 5 (o de cualquier PC)
=============================================================
Mide, con las mismas funciones que usa la app V6, cuánto tarda cada etapa por imagen:
  calibración de color → Rama 1 v2 → parámetros (mm) → Rama 2 (Z.18 sin TTA + Z.19 con TTA).
Además mide cada red por separado para decidir entre el ensamble Z.18 + Z.19 y el respaldo Z.19 solo.

Uso (desde la carpeta DERMATEC_APP):
  .venv/bin/python herramientas/benchmark_pi.py CARPETA_CON_FOTOS [--n 10] [--hilos 4]
Si no se da carpeta, usa imágenes sintéticas (sirve para medir tiempos, no resultados).
Escribe benchmark_pi_resultados.json en la carpeta DERMATEC_APP.
"""
import argparse
import json
import platform
try:
    import resource          # Linux / Raspberry Pi
except ImportError:
    resource = None          # Windows
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "MODELO_DERMATEC_FINAL"))

import calibracion_color  # noqa: E402
import escala_acople  # noqa: E402
import parametros_lesion  # noqa: E402
import rama1_v2  # noqa: E402
from dermatec_inferencia import DermatecFinal  # noqa: E402


def ram_mb():
    """Memoria pico del proceso en MB (Linux: resource; Windows: API de Windows; si no, None)."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return None


def temperatura():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return None


def imagenes(carpeta, n):
    if carpeta:
        rutas = sorted(p for p in Path(carpeta).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))[:n]
        if not rutas:
            sys.exit(f"No hay imágenes en {carpeta}")
        return [(p.name, cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)) for p in rutas]
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        im = np.full((480, 640, 3), (150, 170, 205), np.uint8)
        cv2.ellipse(im, (320 + int(rng.integers(-60, 60)), 240), (70, 55), 0, 0, 360, (60, 70, 110), -1)
        out.append((f"sintetica_{i}.png", cv2.GaussianBlur(im, (0, 0), 3)))
    return out


def resumen(v):
    return {"media_s": round(st.mean(v), 3), "mediana_s": round(st.median(v), 3), "max_s": round(max(v), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta", nargs="?")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()

    print(f"Equipo: {platform.machine()} · Python {platform.python_version()} · hilos {a.hilos}")
    t0 = time.perf_counter()
    modelo = DermatecFinal(carpeta=RAIZ / "MODELO_DERMATEC_FINAL", hilos=a.hilos)
    t_carga = time.perf_counter() - t0
    print(f"Carga de los modelos: {t_carga:.2f} s · RAM {(ram_mb() or 0):.0f} MB")

    fotos = imagenes(a.carpeta, a.n)
    tiempos = {k: [] for k in ("color", "rama1", "parametros", "rama2_ensamble", "total")}
    por_red = {}
    resultados = []
    tmp = Path(tempfile.gettempdir())

    # Calentamiento (la primera inferencia siempre es más lenta)
    _, im0 = fotos[0]
    p0 = tmp / "dermatec_bench_warm.png"
    cv2.imwrite(str(p0), im0)
    modelo.predict(str(p0))

    for nombre, bgr in fotos:
        t = time.perf_counter()
        bgr2, _ = calibracion_color.aplicar(bgr)
        t1 = time.perf_counter()
        mascara, _info = rama1_v2.segmentar_v2(bgr2)
        t2 = time.perf_counter()
        esc = escala_acople.escala_para(bgr2.shape[1])
        params = parametros_lesion.calcular(bgr2, mascara, esc["px_por_mm"])
        t3 = time.perf_counter()
        ruta = tmp / "dermatec_bench.png"
        cv2.imwrite(str(ruta), bgr2)
        r = modelo.predict(str(ruta), mascara=mascara)
        t4 = time.perf_counter()
        for k, v in zip(tiempos, (t1 - t, t2 - t1, t3 - t2, t4 - t3, t4 - t)):
            tiempos[k].append(v)

        # Cada red por separado (misma entrada y misma TTA que en el config)
        from PIL import Image
        for nom, it, size, i_in, i_out, tta_m in modelo.modelos:
            with Image.open(ruta) as im:
                x = DermatecFinal._adecuar(im, size)
            usar = modelo.tta if tta_m is None else bool(tta_m)
            vistas = [x, x[:, :, ::-1], x[:, ::-1], x[:, ::-1, ::-1]] if usar else [x]
            ts = time.perf_counter()
            for v in vistas:
                it.set_tensor(i_in, np.ascontiguousarray(v))
                it.invoke()
            por_red.setdefault(f"{nom} ({'con' if usar else 'sin'} TTA, {len(vistas)} pasada(s))", []).append(time.perf_counter() - ts)

        n_ok = sum(1 for p in params.values() if p.get("valor") is not None)
        resultados.append({"imagen": nombre, "resultado": r["resultado_binario"], "clase": r["clasificacion"],
                           "p_maligno_%": round(r["probabilidad_malignidad"], 1), "aviso": bool(r["aviso_calidad"]),
                           "parametros_calculados": f"{n_ok}/{len(params)}", "total_s": round(t4 - t, 2)})
        print(f"{nombre[:40]:40s} {r['resultado_binario']:8s} {r['clasificacion'][:24]:24s} "
              f"Rama1 {t2 - t1:5.2f} s · Rama2 {t4 - t3:5.2f} s · total {t4 - t:5.2f} s")

    salida = {
        "equipo": platform.platform(), "python": platform.python_version(), "hilos": a.hilos,
        "temperatura_C_final": temperatura(), "ram_pico_MB": round(ram_mb() or 0),
        "carga_modelos_s": round(t_carga, 2), "imagenes": len(fotos),
        "etapas": {k: resumen(v) for k, v in tiempos.items()},
        "redes_por_separado": {k: resumen(v) for k, v in por_red.items()},
        "resultados": resultados,
    }
    z19 = [v for k, v in por_red.items() if k.startswith("Z.19")]
    if z19:
        salida["estimado_total_con_Z19_solo_s"] = round(
            st.mean(tiempos["total"]) - st.mean(tiempos["rama2_ensamble"]) + st.mean(z19[0]), 2)
    (RAIZ / "benchmark_pi_resultados.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ RESUMEN ================")
    print(f"Carga de modelos: {t_carga:.1f} s · RAM pico {salida['ram_pico_MB']} MB · temperatura {salida['temperatura_C_final']} °C")
    for k, v in salida["etapas"].items():
        print(f"  {k:16s} media {v['media_s']:.2f} s (máx {v['max_s']:.2f})")
    for k, v in salida["redes_por_separado"].items():
        print(f"  {k:36s} media {v['media_s']:.2f} s")
    if "estimado_total_con_Z19_solo_s" in salida:
        print(f"  Total estimado con Z.19 solo: {salida['estimado_total_con_Z19_solo_s']:.2f} s por imagen")
    print("Guardado en benchmark_pi_resultados.json")


if __name__ == "__main__":
    main()
