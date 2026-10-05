"""
DERMATEC — Parámetros cuantitativos de la lesión (Rama 1)
=========================================================
Se calculan sobre la imagen que llega y la máscara de la Rama 1, SIN compararlos con ningún dataset.
Las medidas geométricas se expresan en milímetros con la escala del acople (calibracion_escala.json).

Grupos:
  Geometría (mm)      área, perímetro, diámetros mayor / menor / equivalente   → criterio D del ABCD
  Forma (0–1)         circularidad, solidez, excentricidad, índice de irregularidad del borde
  Asimetría (0–1)     respecto a los dos ejes principales                       → criterio A del ABCD
  Borde               nitidez: cambio de luminosidad L* por mm en el contorno     → criterio B del ABCD
  Color (CIELAB)      luminosidad media, ΔE*ab lesión–piel, variabilidad de color → criterio C del ABCD
  Textura             entropía (bits) y uniformidad del histograma de grises

Referencias: Stolz et al. (1994), regla ABCD de dermatoscopía; CIE (1976) espacio L*a*b* y ΔE*ab.
"""
import cv2
import numpy as np

ORDEN = [
    "area_mm2", "perimetro_mm", "diametro_mayor_mm", "diametro_menor_mm", "diametro_equivalente_mm",
    "circularidad", "solidez", "excentricidad", "irregularidad_borde",
    "asimetria_eje_mayor", "asimetria_eje_menor", "asimetria_media",
    "nitidez_borde",
    "luminosidad_L", "delta_e_lesion_piel", "variabilidad_color",
    "entropia", "uniformidad",
]

DEF = {
    "area_mm2": ("Geometría", "Área", "mm²", "Superficie ocupada por la lesión."),
    "perimetro_mm": ("Geometría", "Perímetro", "mm", "Longitud del contorno de la lesión."),
    "diametro_mayor_mm": ("Geometría", "Diámetro mayor", "mm",
                          "Mayor distancia entre dos puntos del borde (criterio D del ABCD: > 6 mm es signo de alerta)."),
    "diametro_menor_mm": ("Geometría", "Diámetro menor", "mm", "Ancho mínimo de la lesión (perpendicular al mayor)."),
    "diametro_equivalente_mm": ("Geometría", "Diámetro equivalente", "mm", "Diámetro de un círculo con la misma área."),
    "circularidad": ("Forma", "Circularidad", "0–1", "4π·área/perímetro². 1 = círculo perfecto; valores bajos = forma irregular."),
    "solidez": ("Forma", "Solidez", "0–1", "Área / área de su envolvente convexa. Valores bajos = entrantes o lobulaciones."),
    "excentricidad": ("Forma", "Excentricidad", "0–1", "Elongación de la elipse ajustada. 0 = redonda; cercano a 1 = alargada."),
    "irregularidad_borde": ("Forma", "Irregularidad del borde", "≥ 1",
                            "Perímetro²/(4π·área). 1 = borde liso circular; mayor = borde más irregular."),
    "asimetria_eje_mayor": ("Asimetría", "Asimetría (eje mayor)", "0–1",
                            "Fracción de la lesión que no coincide al doblarla sobre su eje mayor. 0 = simétrica."),
    "asimetria_eje_menor": ("Asimetría", "Asimetría (eje menor)", "0–1",
                            "Fracción que no coincide al doblarla sobre su eje menor. 0 = simétrica."),
    "asimetria_media": ("Asimetría", "Asimetría media", "0–1", "Promedio de ambos ejes (criterio A del ABCD)."),
    "nitidez_borde": ("Borde", "Nitidez del borde", "ΔL*/mm",
                      "Cambio medio de luminosidad por milímetro en el contorno. Alto = borde abrupto (criterio B del ABCD)."),
    "luminosidad_L": ("Color", "Luminosidad media (L*)", "0–100", "Brillo medio de la lesión. 0 = negro, 100 = blanco."),
    "delta_e_lesion_piel": ("Color", "Contraste lesión–piel (ΔE*ab)", "ΔE",
                            "Diferencia de color entre la lesión y la piel que la rodea. > 2,3 ya es perceptible."),
    "variabilidad_color": ("Color", "Variabilidad de color", "ΔE",
                           "Dispersión del color dentro de la lesión. Alta = varios tonos (criterio C del ABCD)."),
    "entropia": ("Textura", "Entropía", "bits (0–8)", "Heterogeneidad de tonos dentro de la lesión. Alta = textura compleja."),
    "uniformidad": ("Textura", "Uniformidad", "0–1", "Energía del histograma. Alta = lesión homogénea."),
}


def _feret_max(contorno):
    h = cv2.convexHull(contorno).reshape(-1, 2).astype(np.float64)
    if len(h) < 2:
        return 0.0
    d = np.sqrt(((h[:, None, :] - h[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def _asimetria(mask):
    """Asimetría respecto a los ejes principales (momentos de segundo orden)."""
    m = mask.astype(np.uint8)
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] == 0:
        return np.nan, np.nan
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    ang = 0.5 * np.degrees(np.arctan2(2 * M["mu11"], M["mu20"] - M["mu02"]))
    h, w = m.shape
    lado = int(np.ceil(np.hypot(h, w))) + 4
    lienzo = np.zeros((lado, lado), np.uint8)
    ox, oy = lado // 2 - int(round(cx)), lado // 2 - int(round(cy))
    ys, xs = np.nonzero(m)
    lienzo[ys + oy, xs + ox] = 1
    R = cv2.getRotationMatrix2D((lado / 2, lado / 2), ang, 1.0)
    rot = cv2.warpAffine(lienzo, R, (lado, lado), flags=cv2.INTER_NEAREST)
    area = max(1, int(rot.sum()))
    a_mayor = float(np.logical_xor(rot, rot[::-1, :]).sum()) / (2 * area)   # doblar sobre el eje mayor (horizontal)
    a_menor = float(np.logical_xor(rot, rot[:, ::-1]).sum()) / (2 * area)   # doblar sobre el eje menor (vertical)
    return a_mayor, a_menor


def calcular(bgr, mascara, px_por_mm):
    """bgr: imagen original; mascara: máscara final de la Rama 1 (0/255); px_por_mm: escala del acople.
    Devuelve {clave: {grupo, nombre, valor, unidad, descripcion}} en el orden de ORDEN."""
    salida = {k: {"grupo": DEF[k][0], "nombre": DEF[k][1], "valor": None, "unidad": DEF[k][2], "descripcion": DEF[k][3]}
              for k in ORDEN}
    m = (np.asarray(mascara) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape != bgr.shape[:2]:
        m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return salida
    m = (lab_cc == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    s = float(px_por_mm) if px_por_mm else np.nan
    A = float(m.sum())
    P = float(cv2.arcLength(cv2.approxPolyDP(c, 1.0, True), True))   # evita el sobreconteo del contorno en escalera
    v = {}
    v["area_mm2"] = A / s ** 2
    v["perimetro_mm"] = P / s
    v["diametro_mayor_mm"] = _feret_max(c) / s
    v["diametro_menor_mm"] = float(min(cv2.minAreaRect(c)[1])) / s
    v["diametro_equivalente_mm"] = float(np.sqrt(4 * A / np.pi)) / s
    v["circularidad"] = float(np.clip(4 * np.pi * A / max(P * P, 1e-9), 0, 1))
    v["irregularidad_borde"] = max(1.0, float(P * P / (4 * np.pi * max(A, 1))))
    v["solidez"] = A / max(float(cv2.contourArea(cv2.convexHull(c))), 1.0)
    if len(c) >= 5:
        (_, _), (e1, e2), _ = cv2.fitEllipse(c)
        a, b = max(e1, e2), min(e1, e2)
        v["excentricidad"] = float(np.sqrt(max(0.0, 1 - (b / a) ** 2))) if a > 0 else None
    am, an = _asimetria(m)
    v["asimetria_eje_mayor"], v["asimetria_eje_menor"] = am, an
    v["asimetria_media"] = (am + an) / 2

    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)          # L 0–100, a*, b* reales
    les = lab[m > 0]
    anillo_px = max(5, int(round(0.5 * s))) if np.isfinite(s) else 7                 # piel: anillo de ~0,5 mm
    anillo = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * anillo_px + 1,) * 2))
    piel = (anillo > 0) & (m == 0)
    media = les.mean(0)
    v["luminosidad_L"] = float(media[0])
    v["variabilidad_color"] = float(np.sqrt(((les - media) ** 2).sum(1).mean()))
    if piel.sum() > 30:
        v["delta_e_lesion_piel"] = float(np.linalg.norm(media - lab[piel].mean(0)))
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    g = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    pts = c.reshape(-1, 2)
    v["nitidez_borde"] = float(g[pts[:, 1], pts[:, 0]].mean() * s)                    # ΔL* por píxel → por mm
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[m > 0]
    p = np.bincount(gris, minlength=256).astype(np.float64)
    p /= max(p.sum(), 1)
    nz = p[p > 0]
    v["entropia"] = float(-(nz * np.log2(nz)).sum()) + 0.0
    v["uniformidad"] = float((p ** 2).sum())
    for k, val in v.items():
        if val is not None and np.isfinite(val):
            salida[k]["valor"] = float(val)
    return salida


def formatear(valor, unidad):
    if valor is None:
        return "—"
    if unidad in ("mm", "mm²", "ΔE", "ΔL*/mm", "0–100"):
        return f"{valor:.2f}".replace(".", ",")
    if unidad.startswith("bits"):
        return f"{valor:.2f}".replace(".", ",")
    return f"{valor:.3f}".replace(".", ",")
