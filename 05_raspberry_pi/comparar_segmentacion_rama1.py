"""
DERMATEC — Comparar variantes de la Rama 1 con capturas tomadas CON EL ACOPLE (sin tocar la app)
===============================================================================================
Variantes:
  A  Rama 1 actual          : B2.23 → GMM Lab → DILATE_5 sobre la imagen completa (ajustada con capturas cercanas)
  B  Segmentación rápida J2 : segment_lesion() de dermatec_parametros.py (pensada para encuadre de dermatoscopio)
  D  Rama 1 v2 (rama1_v2.py, la nueva por defecto en la app)
  C  (descartada) dos etapas   : B localiza la lesión → recorte cuadrado con margen → Rama 1 (A) sobre el recorte
                              → la máscara vuelve a la imagen completa. Así la Rama 1 trabaja en el régimen para el
                              que se ajustó (lesión grande en el cuadro) y procesa menos píxeles (más rápida).

Uso (en la carpeta principal de la app):
    .venv\\Scripts\\python.exe herramientas\\comparar_segmentacion_rama1.py "C:\\CAPTURAS_ACOPLE"
    ... herramientas\\comparar_segmentacion_rama1.py "C:\\CAPTURAS_ACOPLE" --mascaras "C:\\MASCARAS_ACOPLE"   (Dice si hay máscaras)
Salida en <carpeta>\\comparacion_rama1\\ : una imagen por captura (contornos A rojo, B azul, D amarillo) y un CSV.
"""
import argparse, csv, importlib.util, sys, time
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent.parent          # carpeta principal de la app
for _d in [AQUI, AQUI / "MODELO_DERMATEC_FINAL"] + sorted(AQUI.glob("MODELO_DERMATEC*")):
    sys.path.insert(0, str(_d))
try:
    import dermatec_parametros as dp                              # noqa: E402
except ModuleNotFoundError:
    try:                                                          # respaldo: paquete J2.1 si está instalado
        sys.path.insert(0, str(AQUI / "DERMATEC_J2_1_VISUAL"))
        from src import dermatec_j21_inferencia as dp             # noqa: E402
    except Exception:
        sys.exit("❌ Falta dermatec_parametros.py. Cópialo (del zip que te envié) dentro de MODELO_DERMATEC_FINAL "
                 "o junto a este script y vuelve a correr.")

app_py = sorted(AQUI.glob("dermatec_app_v*.py"))[-1]
spec = importlib.util.spec_from_file_location("dermatec_app", str(app_py))
app = importlib.util.module_from_spec(spec); sys.modules["dermatec_app"] = app; spec.loader.exec_module(app)


def seg_rapida(bgr):
    """B: segmentación J2 (RGB, reescalada a ≤384 px) → máscara del tamaño original."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    peq = dp.resize_for_seg(rgb)
    s = dp.segment_lesion(peq)
    m = (cv2.resize(s["mask"].astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)
    return m.astype(np.uint8) * 255, bool(int(s["valid"]))


def ejecutar_rama1_dos_etapas(bgr, margen=0.9, lado_min=0.40):
    """C: localizar con B, recortar un cuadrado alrededor de la lesión y correr la Rama 1 validada en el recorte.
    margen: fracción del tamaño de la lesión que se agrega a cada lado; lado_min: lado mínimo del recorte
    como fracción del lado corto de la imagen."""
    h, w = bgr.shape[:2]
    m0, ok = seg_rapida(bgr)
    ys, xs = np.nonzero(m0)
    if ok and len(xs) > 50:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        lado = max(x1 - x0, y1 - y0) * (1 + 2 * margen)
    else:                                                          # sin localización: centro de la imagen
        cx, cy, lado = w / 2, h / 2, 0.8 * min(h, w)
    lado = int(np.clip(lado, lado_min * min(h, w), min(h, w)))
    x0 = int(np.clip(cx - lado / 2, 0, w - lado)); y0 = int(np.clip(cy - lado / 2, 0, h - lado))
    rec = bgr[y0:y0 + lado, x0:x0 + lado]
    r = app.ejecutar_rama1(rec)
    mascara = np.zeros((h, w), np.uint8); mascara[y0:y0 + lado, x0:x0 + lado] = r["mascara_final"]
    datos = app._rama1_roi_y_parametros(bgr, mascara)              # parámetros sobre la imagen completa
    return {"mascara_final": mascara, "valida": datos["valida"], "calidad": datos["calidad"],
            "parametros": datos["parametros"], "recorte": (x0, y0, lado)}


def encuadre(m):
    m = m > 0; h, w = m.shape; b = max(2, int(0.01 * min(h, w)))
    return float(m.mean()), int(m[:b].any()) + int(m[-b:].any()) + int(m[:, :b].any()) + int(m[:, -b:].any())


def dice(a, b):
    a, b = a > 0, b > 0
    return 2 * (a & b).sum() / max(1, a.sum() + b.sum())


ap = argparse.ArgumentParser()
ap.add_argument("carpeta"); ap.add_argument("--mascaras", default=None); ap.add_argument("--max", type=int, default=0)
a = ap.parse_args()
raiz = Path(a.carpeta); sal = raiz / "comparacion_rama1"; sal.mkdir(exist_ok=True)
imgs = sorted(p for p in raiz.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[: a.max or None]
filas = []
for i, ruta in enumerate(imgs, 1):
    bgr = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)
    res = {}
    t = time.perf_counter(); rA = app.ejecutar_rama1_v1(bgr); res["A"] = (rA["mascara_final"], time.perf_counter() - t)
    t = time.perf_counter(); mB, _ = seg_rapida(bgr); res["B"] = (mB, time.perf_counter() - t)
    t = time.perf_counter(); rD = app.ejecutar_rama1_v2(bgr); res["D"] = (rD["mascara_final"], time.perf_counter() - t)
    gt = None
    if a.mascaras:
        cand = [p for p in Path(a.mascaras).glob(ruta.stem + ".*")]
        if cand:
            gt = cv2.imdecode(np.fromfile(str(cand[0]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    fila = {"archivo": ruta.name}
    vis = bgr.copy()
    for k, color in [("A", (0, 0, 255)), ("B", (255, 128, 0)), ("D", (0, 255, 255))]:
        m, seg = res[k]; frac, bordes = encuadre(m)
        fila.update({f"{k}_seg": round(seg, 2), f"{k}_fraccion": round(frac, 3), f"{k}_bordes": bordes})
        if gt is not None: fila[f"{k}_dice"] = round(dice(m, gt), 3)
        cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, color, 2)
    cv2.putText(vis, "A v1 (rojo)  B rapida J2 (azul)  D Rama 1 v2 (amarillo)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", np.hstack([bgr, vis])); buf.tofile(str(sal / (ruta.stem + "_comparacion.jpg")))
    filas.append(fila); print(f"\r{i}/{len(imgs)}", end="")

with open(sal / "comparacion_rama1.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for x in filas for k in x}, key=lambda k: (k != "archivo", k)))
    w.writeheader(); w.writerows(filas)
print(f"\n\nImágenes: {len(filas)} | salida: {sal}")
for k, nombre in [("A", "Rama 1 v1"), ("B", "Rápida J2"), ("D", "Rama 1 v2")]:
    seg = np.mean([x[f"{k}_seg"] for x in filas]); frac = np.median([x[f"{k}_fraccion"] for x in filas])
    toca = np.mean([x[f"{k}_bordes"] >= 2 for x in filas]) * 100
    txt = f"  {k} {nombre:14s} {seg:5.2f} s/img | fracción mediana {frac:.2f} | toca ≥2 bordes {toca:4.0f} %"
    if a.mascaras and f"{k}_dice" in filas[0]:
        txt += f" | Dice medio {np.mean([x[f'{k}_dice'] for x in filas if f'{k}_dice' in x]):.3f}"
    print(txt)
print("Revisa las imágenes *_comparacion.jpg: el contorno correcto debe rodear solo la lesión.")
