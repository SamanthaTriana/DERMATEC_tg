"""
DERMATEC — Análisis de la entrada de la CNN: canales RGB, histogramas e iluminante (Shades of Gray)
===================================================================================================
Compara uno o varios grupos de imágenes (por ejemplo, una muestra del conjunto de entrenamiento y las
capturas del capilógrafo) tal como las ve la red:
  - adecuación a 288×288 (misma función de la app),
  - medias y desviaciones por canal R, G, B,
  - iluminante estimado (Minkowski p = 6) y ganancia de Shades of Gray,
  - histogramas por canal ANTES y DESPUÉS de Shades of Gray.

Uso (no requiere TensorFlow):
    python analisis_entrada_rgb.py --grupo Entrenamiento "D:\\muestra_train" --grupo Capilografo "C:\\CAPTURAS"
    python analisis_entrada_rgb.py --grupo Capturas C:\\CAPTURAS --max 200 --salida resultados_rgb

Salidas (carpeta --salida): analisis_rgb_por_imagen.csv, analisis_rgb_resumen.csv e histogramas_rgb.png
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entrada_cnn import adecuar_imagen, shades_of_gray_np      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--grupo", nargs=2, action="append", metavar=("NOMBRE", "CARPETA"), required=True)
ap.add_argument("--max", type=int, default=300, help="máximo de imágenes por grupo (muestreo fijo)")
ap.add_argument("--salida", default="resultados_rgb")
a = ap.parse_args()

sal = Path(a.salida); sal.mkdir(parents=True, exist_ok=True)
bins = np.arange(257)
filas, hist = [], {}
for nombre, carpeta in a.grupo:
    rutas = sorted(p for p in Path(carpeta).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if len(rutas) > a.max:
        rutas = [rutas[i] for i in np.random.default_rng(0).choice(len(rutas), a.max, replace=False)]
    h_antes, h_desp = np.zeros((3, 256)), np.zeros((3, 256))
    for r in rutas:
        x = adecuar_imagen(r)
        y, ilum, gan = shades_of_gray_np(x)
        for c in range(3):
            h_antes[c] += np.histogram(x[..., c], bins=bins)[0]
            h_desp[c] += np.histogram(y[..., c], bins=bins)[0]
        m, s, my = x.reshape(-1, 3).mean(0), x.reshape(-1, 3).std(0), y.reshape(-1, 3).mean(0)
        filas.append({"grupo": nombre, "archivo": r.name,
                      "media_R": m[0], "media_G": m[1], "media_B": m[2], "std_R": s[0], "std_G": s[1], "std_B": s[2],
                      "brillo": float(m.mean()), "ilum_R": ilum[0, 0], "ilum_G": ilum[0, 1], "ilum_B": ilum[0, 2],
                      "ganancia_R": gan[0, 0], "ganancia_G": gan[0, 1], "ganancia_B": gan[0, 2],
                      "SoG_media_R": my[0], "SoG_media_G": my[1], "SoG_media_B": my[2]})
    hist[nombre] = (h_antes / max(1, h_antes.sum(1, keepdims=True).max()), h_desp / max(1, h_desp.sum(1, keepdims=True).max()), len(rutas))
    print(f"{nombre}: {len(rutas)} imágenes")

with open(sal / "analisis_rgb_por_imagen.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader()
    w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()} for fila in filas)

claves = [k for k in filas[0] if k not in ("grupo", "archivo")]
resumen = []
for nombre, _ in a.grupo:
    g = [f for f in filas if f["grupo"] == nombre]
    fila = {"grupo": nombre, "n": len(g)}
    for k in claves:
        v = np.array([f[k] for f in g], dtype=float)
        fila[k + "_media"], fila[k + "_std"] = round(float(v.mean()), 3), round(float(v.std()), 3)
    resumen.append(fila)
with open(sal / "analisis_rgb_resumen.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(resumen[0])); w.writeheader(); w.writerows(resumen)
for r in resumen:
    print(f"  {r['grupo']:15s} n={r['n']:4d} | medias RGB ({r['media_R_media']:.0f}, {r['media_G_media']:.0f}, {r['media_B_media']:.0f})"
          f" | ganancia SoG ({r['ganancia_R_media']:.2f}, {r['ganancia_G_media']:.2f}, {r['ganancia_B_media']:.2f})")

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(hist), 2, figsize=(12, 3.2 * len(hist)), squeeze=False)
    colores = ["#C0392B", "#27AE60", "#2E86C1"]
    for i, (nombre, (ha, hd, n)) in enumerate(hist.items()):
        for j, (h, tit) in enumerate([(ha, "antes de Shades of Gray"), (hd, "después de Shades of Gray")]):
            for c, col, lab in zip(range(3), colores, "RGB"):
                ax[i, j].plot(np.arange(256), h[c], color=col, lw=1.4, label=lab)
            ax[i, j].set_title(f"{nombre} (n = {n}) — {tit}", fontsize=10)
            ax[i, j].set_xlim(0, 255); ax[i, j].set_xlabel("Intensidad"); ax[i, j].legend(fontsize=8)
            ax[i, j].spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(sal / "histogramas_rgb.png", dpi=150)
    print("Figura:", sal / "histogramas_rgb.png")
except ImportError:
    print("(matplotlib no instalado: se omiten los histogramas; los CSV sí se guardaron)")
print("CSV:", sal / "analisis_rgb_por_imagen.csv", "y", sal / "analisis_rgb_resumen.csv")
