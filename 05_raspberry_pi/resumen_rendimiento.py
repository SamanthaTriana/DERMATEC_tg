"""
DERMATEC — Resumen de rendimiento para los anexos (L: Raspberry Pi, H: interfaz)
================================================================================
Lee los CSV de registros/ y escribe registros/RESUMEN_RENDIMIENTO.txt con:
  - horas de uso de la app (por sesión y total), RAM, CPU, temperatura y throttling
  - tiempo por imagen (total y por etapa): media, mediana, p95, mínimo y máximo
  - tiempo por diagnóstico (lote de N imágenes) y carga del modelo
Uso:  python herramientas/resumen_rendimiento.py            (Raspberry: .venv/bin/python ...)
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REG = RAIZ / "registros"


def leer(nombre):
    ruta = REG / nombre
    if not ruta.is_file():
        return []
    with open(ruta, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nums(filas, col):
    out = []
    for f in filas:
        try:
            out.append(float(f[col]))
        except (KeyError, TypeError, ValueError):
            pass
    return out


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]


def linea(nombre, v, u="s", dec=3):
    if not v:
        return f"  {nombre:<34} sin datos"
    return (f"  {nombre:<34} n={len(v):<4} media {st.mean(v):.{dec}f} {u} · mediana {st.median(v):.{dec}f} · "
            f"p95 {p95(v):.{dec}f} · mín {min(v):.{dec}f} · máx {max(v):.{dec}f}")


def main():
    mon, img, les, ses = (leer(n) for n in ("monitor_sistema.csv", "analisis_por_imagen.csv",
                                            "analisis_por_lesion.csv", "sesiones.csv"))
    L = ["DERMATEC — RESUMEN DE RENDIMIENTO", "=" * 34]
    equipos = sorted({f.get("equipo", "") for f in ses + img if f.get("equipo")})
    L.append("Equipo(s): " + (", ".join(equipos) or "sin datos"))
    ini = [f for f in ses if f.get("evento") == "inicio"]
    if ini:
        f = ini[-1]
        L.append(f"Sistema: {f.get('sistema')} · Python {f.get('python')} · {f.get('nucleos')} núcleos · "
                 f"RAM total {f.get('ram_total_mb')} MB")

    # ---- uso
    por_sesion = defaultdict(float)
    for f in mon:
        try:
            por_sesion[f["sesion"]] = max(por_sesion[f["sesion"]], float(f["app_abierta_min"]))
        except (KeyError, ValueError):
            pass
    L += ["", "USO DE LA APP"]
    if por_sesion:
        L.append(f"  Sesiones: {len(por_sesion)} · tiempo total abierta {sum(por_sesion.values()) / 60:.2f} h · "
                 f"sesión más larga {max(por_sesion.values()) / 60:.2f} h")
    else:
        L.append("  sin datos")
    L.append(f"  Diagnósticos (lesiones analizadas): {len(les)} · imágenes analizadas: {len(img)}")

    # ---- recursos
    L += ["", "RECURSOS (muestras cada 60 s + una por imagen)"]
    L.append(linea("CPU del sistema", nums(mon, "cpu_pct"), "%", 1))
    L.append(linea("RAM del sistema usada", nums(mon, "ram_sistema_usada_mb"), "MB", 0))
    L.append(linea("RAM de la app", nums(mon, "ram_app_mb") + nums(img, "ram_app_mb"), "MB", 0))
    pico = nums(mon, "ram_app_pico_mb") + nums(les, "ram_app_pico_mb")
    L.append(f"  {'RAM pico de la app':<34} {max(pico):.0f} MB" if pico else f"  {'RAM pico de la app':<34} sin datos")
    L.append(linea("Temperatura CPU", nums(mon, "temp_cpu_c") + nums(img, "temp_cpu_c"), "°C", 1))
    L.append(linea("Frecuencia CPU", nums(mon, "frec_cpu_mhz"), "MHz", 0))
    thr = sorted({f.get("throttled", "") for f in mon if f.get("throttled")})
    L.append(f"  {'Throttling (vcgencmd)':<34} {', '.join(thr) if thr else 'no disponible'}"
             + ("   (0x0 = nunca se limitó)" if thr == ["0x0"] else ""))
    disco = nums(mon, "disco_libre_gb")
    if disco:
        L.append(f"  {'Disco libre':<34} {min(disco):.1f}–{max(disco):.1f} GB")
    peso = sum(p.stat().st_size for p in (RAIZ / "MODELO_DERMATEC_FINAL").glob("*.tflite"))
    L.append(f"  {'Tamaño de los modelos (.tflite)':<34} {peso / 1e6:.1f} MB")

    # ---- tiempos por imagen
    L += ["", "TIEMPO POR IMAGEN"]
    for col, nom in (("t_total_imagen_s", "Total por imagen"), ("t_lectura_s", "Lectura"),
                     ("t_calibracion_color_s", "Calibración de color"), ("t_rama1_s", "Rama 1 (segmentación)"),
                     ("t_parametros_s", "Parámetros en mm"), ("t_clasificador_s", "Clasificador Z.18 + Z.19")):
        L.append(linea(nom, nums(img, col)))
    por_res = defaultdict(list)
    for f in img:
        try:
            por_res[f["resolucion"]].append(float(f["t_total_imagen_s"]))
        except (KeyError, ValueError):
            pass
    for r, v in sorted(por_res.items()):
        L.append(linea(f"Total, imágenes {r}", v))

    # ---- por diagnóstico
    L += ["", "TIEMPO POR DIAGNÓSTICO (lote de N imágenes)"]
    L.append(linea("Total del diagnóstico", nums(les, "t_total_diagnostico_s")))
    L.append(linea("Por imagen dentro del lote", nums(les, "t_por_imagen_s")))
    carga = [x for x in nums(les, "t_carga_modelo_s") if x > 0]
    L.append(linea("Carga del modelo (1.ª vez)", carga))
    por_n = defaultdict(list)
    for f in les:
        try:
            por_n[int(f["n_imagenes"])].append(float(f["t_total_diagnostico_s"]))
        except (KeyError, ValueError):
            pass
    for n, v in sorted(por_n.items()):
        L.append(linea(f"Lotes de {n} imagen(es)", v))

    texto = "\n".join(L) + "\n"
    REG.mkdir(exist_ok=True)
    (REG / "RESUMEN_RENDIMIENTO.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"Guardado en {REG / 'RESUMEN_RENDIMIENTO.txt'}")


if __name__ == "__main__":
    main()
