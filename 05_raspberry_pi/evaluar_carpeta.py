"""
DERMATEC — Evaluar una carpeta de capturas con el clasificador final (incluye control de calidad).
Uso (desde la carpeta principal de la app):
    EVALUAR_CARPETA.bat "C:\\CAPTURAS"
    o  .venv\\Scripts\\python.exe herramientas\\evaluar_carpeta.py "C:\\CAPTURAS"   -> igual que la app (Rama 1 v2)
    ... --j21      -> encuadre con la segmentación rápida J2
    ... --rapido   -> sin segmentación (solo control por incertidumbre)
Genera en esa carpeta: resultados_clasificador.csv y un resumen en pantalla.
"""
import csv, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "MODELO_DERMATEC_FINAL"))
sys.path.insert(0, str(RAIZ))
for _sub in ("rama", "calibracion"):
    sys.path.insert(0, str(RAIZ / _sub))
from dermatec_inferencia import DermatecFinal                     # noqa: E402
import calibracion_color                                          # noqa: E402
import cv2, tempfile, uuid                                        # noqa: E402
SIN_COLOR = "--sin-color" in sys.argv                             # desactiva la calibración de color

args = [a for a in sys.argv[1:] if not a.startswith("--")]
rapido = "--rapido" in sys.argv
raiz = Path(args[0] if args else ".")
imgs = sorted(p for p in raiz.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
if not imgs:
    sys.exit(f"No hay imágenes en {raiz}")
d = DermatecFinal(RAIZ / "MODELO_DERMATEC_FINAL")
if rapido:
    d._param_mod = None

# Rama 1 de la app (misma que usa la interfaz), salvo --j21 o --rapido
rama1 = None
if not rapido and "--j21" not in sys.argv:
    try:
        import importlib.util, cv2
        app_py = sorted(RAIZ.glob("dermatec_app_v*.py"))[-1]
        spec = importlib.util.spec_from_file_location("dermatec_app", str(app_py))
        mod = importlib.util.module_from_spec(spec); sys.modules["dermatec_app"] = mod; spec.loader.exec_module(mod)
        rama1 = mod.ejecutar_rama1
        print("Encuadre con la Rama 1 de", app_py.name)
    except Exception as exc:
        print("Rama 1 de la app no disponible; se usa la segmentación J2.1:", repr(exc))
filas, t0 = [], time.time()
for i, ruta in enumerate(imgs, 1):
    mascara, tmp = None, None
    bgr = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)
    info = {"aplicada": False}
    if not SIN_COLOR:
        bgr, info = calibracion_color.aplicar(bgr)
    if rama1 is not None:
        try:
            mascara = rama1(bgr)["mascara_final"]
        except Exception as exc:
            print(f"\n  Rama 1 falló en {ruta.name}: {exc!r}")
    entrada = ruta
    if info.get("aplicada"):
        tmp = Path(tempfile.gettempdir()) / f"DERMATEC_CAL_{uuid.uuid4().hex}.png"
        tmp.write_bytes(cv2.imencode(".png", bgr)[1].tobytes()); entrada = tmp
    try:
        r = d.predict(entrada, mascara=mascara)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    p = np.asarray(r["_p"]); seg = r.get("segmentacion", {})
    filas.append({"archivo": ruta.name, "carpeta": str(ruta.parent.relative_to(raiz)),
                  "resultado": r["resultado_binario"], "sin_control": r["resultado_sin_control_calidad"],
                  "clase": r["clase_mas_probable"], "p_maligno": round(1 - float(p[0]), 4),
                  "incertidumbre_bits": round(r["incertidumbre_entropia_bits"], 3),
                  "fraccion_campo_lesion": round(seg.get("fraccion_campo_lesion", float("nan")), 3),
                  "bordes_tocados": seg.get("bordes_tocados", ""),
                  "motivos": " | ".join(r["control_calidad"]["motivos"]),
                  **{f"p_{n.split()[0]}": round(float(v), 4) for n, v in zip(["Nevo", "Melanoma", "BCC", "SCC"], p)}})
    print(f"\r{i}/{len(imgs)}", end="")
salida = raiz / "resultados_clasificador.csv"
with open(salida, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
n = len(filas)
print(f"\n\nCalibración de color: {'NO' if SIN_COLOR or not calibracion_color._cargar()[0] else 'sí'}")
print(f"Modelo: {d.nombre} | umbral {d.threshold:.3f} | imágenes: {n} ({time.time() - t0:.0f} s)")
for k, v in Counter(f["resultado"] for f in filas).most_common():
    print(f"  {k:16s} {v:3d} ({100 * v / n:.1f} %)")
av = sum(bool(f["motivos"]) for f in filas)
print(f"Con aviso de baja confianza (repetir captura): {av} ({100 * av / n:.1f} %)")
print(f"P(maligno) mediana {np.median([f['p_maligno'] for f in filas]):.2f} | "
      f"incertidumbre mediana {np.median([f['incertidumbre_bits'] for f in filas]):.2f} bits")
print("CSV:", salida)
