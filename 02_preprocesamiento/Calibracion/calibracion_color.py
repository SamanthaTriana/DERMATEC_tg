"""
DERMATEC — Calibración de color del capilógrafo (con el acople final y la perilla de luz fija)
=============================================================================================
Se construye UNA vez con fotos de la carta de grises (CARTA_CALIBRACION_DERMATEC.pdf) y se aplica igual a todas
las capturas. Corrige dos cosas medibles:
  1. Iluminación no uniforme del LED (campo plano / "flat-field"): con la foto del blanco (G1) se mide cuánto más
     oscuras salen las esquinas que el centro y se compensa.
  2. Balance de grises: con los grises medios (G5–G7) se calculan ganancias por canal para que un gris salga gris
     (R = G = B).
Ambas correcciones se hacen en espacio lineal (se deshace la gamma sRGB, se corrige y se vuelve a aplicar).
No se ajusta la curva de tonos mientras la cámara tenga exposición automática (ver LEEME).

Uso:
  python calibracion_color.py FOTO_G1_BLANCO.jpg [FOTO_G5.jpg FOTO_G6.jpg FOTO_G7.jpg ...]
  (o desde la app: MENÚ → Calibrar color)
Genera calibracion_color.json y calibracion_color_campo.npz junto a este archivo.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

AQUI = Path(__file__).resolve().parent
ARCHIVO = AQUI / "calibracion_color.json"
CAMPO = AQUI / "calibracion_color_campo.npz"
_cache = {"mtime": None, "cfg": None, "campo": None}


def _leer(ruta):
    im = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"No se pudo leer {ruta}")
    return im


def _a_lineal(x):          # sRGB (0–1) → lineal
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _a_srgb(x):            # lineal → sRGB (0–1)
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def construir(ruta_blanco, rutas_grises=(), acople="8 cm"):
    """Crea la calibración a partir de la foto del blanco y (opcional) fotos de grises medios."""
    blanco = _leer(ruta_blanco).astype(np.float32) / 255.0
    lin = _a_lineal(blanco)
    campo = cv2.GaussianBlur(lin, (0, 0), 25)                       # iluminación suave (sin la textura del papel)
    h, w = campo.shape[:2]
    centro = campo[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40].reshape(-1, 3).mean(0)
    ganancia_campo = np.clip(centro[None, None, :] / np.maximum(campo, 1e-4), 1.0, 1.6)
    peq = cv2.resize(ganancia_campo, (80, 60), interpolation=cv2.INTER_AREA).astype(np.float32)

    muestras = [_a_lineal(_leer(r).astype(np.float32) / 255.0) for r in rutas_grises] or [lin]
    medias = []
    for m in muestras:
        mh, mw = m.shape[:2]
        c = m[int(.3 * mh):int(.7 * mh), int(.3 * mw):int(.7 * mw)].reshape(-1, 3).mean(0)   # BGR
        medias.append(c)
    medias = np.mean(medias, axis=0)
    ganancias = (medias[1] / medias).astype(float)                   # verde como referencia (BGR)

    esquinas = float(np.mean([campo[20, 20].mean(), campo[20, -20].mean(), campo[-20, 20].mean(),
                              campo[-20, -20].mean()]) / centro.mean())
    cfg = {"activa": True, "acople": acople, "fecha": datetime.now().isoformat(timespec="seconds"),
           "ganancias_bgr": [round(g, 4) for g in ganancias], "tamano_referencia": [w, h],
           "iluminacion_esquinas_vs_centro": round(esquinas, 3),
           "fuente": {"blanco": Path(ruta_blanco).name, "grises": [Path(r).name for r in rutas_grises]},
           "nota": "Campo plano + balance de grises en espacio lineal. Sin curva de tonos (exposición automática)."}
    ARCHIVO.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(CAMPO, ganancia=peq)
    _cache["mtime"] = None
    return cfg


def _cargar():
    if not ARCHIVO.is_file() or not CAMPO.is_file():
        return None, None
    mt = (ARCHIVO.stat().st_mtime, CAMPO.stat().st_mtime)
    if _cache["mtime"] != mt:
        _cache["cfg"] = json.loads(ARCHIVO.read_text(encoding="utf-8"))
        _cache["campo"] = np.load(CAMPO)["ganancia"]
        _cache["mtime"] = mt
    return _cache["cfg"], _cache["campo"]


def aplicar(bgr):
    """Devuelve (imagen corregida, info). Si no hay calibración activa devuelve la original."""
    cfg, campo = _cargar()
    if not cfg or not cfg.get("activa", True):
        return bgr, {"aplicada": False}
    h, w = bgr.shape[:2]
    g = cv2.resize(campo, (w, h), interpolation=cv2.INTER_LINEAR)
    lin = _a_lineal(bgr.astype(np.float32) / 255.0)
    lin = lin * g * np.asarray(cfg["ganancias_bgr"], np.float32)[None, None, :]
    out = (_a_srgb(lin) * 255.0 + 0.5).astype(np.uint8)
    return out, {"aplicada": True, "acople": cfg.get("acople"), "fecha": cfg.get("fecha")}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    c = construir(sys.argv[1], sys.argv[2:])
    print(json.dumps(c, indent=2, ensure_ascii=False))
