"""
DERMATEC — Rama 1 v2 (robusta al encuadre con acople largo y al encuadre cercano)
===============================================================================
Etapas (clásicas, sin GPU; referencias en el LEEME):
  0. Reducción a 400 px de ancho (la segmentación no necesita más resolución; la máscara vuelve al tamaño original).
  1. Remoción de vello tipo DullRazor (Lee et al., 1997): black-hat morfológico con elementos lineales en 12
     orientaciones → máscara de vello → inpainting (Telea, 2004).
  2. Máscara de piel válida: se excluye lo que no es piel (papel/borde del acople: brillante y poco saturado;
     zonas muy oscuras del borde) — las lesiones nunca son blancas.
  3. Corrección de sombreado (Cavalcanti & Scharcanski, 2011): se ajusta una superficie cuadrática a cada canal
     Lab sobre la piel (ajuste robusto iterativo que excluye la lesión) y se resta → iluminación plana.
  4. Mapa de "lesionalidad": distancia de color a la piel sana en Lab corregido, contando solo lo más oscuro
     (ΔL<0) o más pigmentado (Δb, Δa > 0), normalizada por la dispersión robusta de la piel (MAD).
  5. Umbral de Otsu (Celebi et al., 2009) acotado por abajo por un umbral estadístico de la piel (mediana + k·MAD).
  6. Selección del componente por puntaje (contraste medio × √área × prior de centro), no por el más grande.
  7. Refinamiento del borde con GrabCut (Rother et al., 2004) iniciado con el componente elegido.
  8. Suavizado, relleno de huecos, reescalado y DILATE_5 (igual que la Rama 1 validada).
"""
import time
import cv2
import numpy as np

ANCHO = 400


def quitar_vello(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bh = np.zeros_like(g)
    L = 17
    for ang in range(0, 180, 15):
        k = np.zeros((L, L), np.uint8)
        c = L // 2
        dx, dy = np.cos(np.deg2rad(ang)) * c, np.sin(np.deg2rad(ang)) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
        bh = np.maximum(bh, cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    # umbral adaptativo al contraste de la imagen
    t = max(10, float(np.percentile(bh, 97)))
    m = (bh >= t).astype(np.uint8)
    # quedarse con estructuras alargadas (vello), no con puntos (retícula/pigmento)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w_, h_, a = st[i, 2], st[i, 3], st[i, 4]
        if max(w_, h_) >= 18 and a / max(1, w_ * h_) < 0.45:
            keep[i] = True
    m = keep[lab].astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    limpia = cv2.inpaint(bgr, m, 5, cv2.INPAINT_TELEA) if m.any() else bgr.copy()
    return limpia, m


def mascara_piel(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    papel = (v > 170) & (s < 45)
    papel = cv2.morphologyEx(papel.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # solo regiones grandes de papel (no reflejos puntuales)
    n, lab, st, _ = cv2.connectedComponentsWithStats(papel, 8)
    grande = np.zeros(n, bool); grande[1:] = st[1:, 4] > 0.01 * papel.size
    papel = grande[lab].astype(np.uint8)
    papel = cv2.dilate(papel, np.ones((15, 15), np.uint8))
    oscuro = v < 25
    return (papel == 0) & ~oscuro


def ajustar_superficie(canal, valido, iters=3):
    h, w = canal.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w; yy /= h
    A_full = np.stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], -1).reshape(-1, 6)
    z = canal.reshape(-1)
    sel = valido.reshape(-1).copy()
    paso = max(1, int(sel.sum() // 20000))
    for _ in range(iters):
        idx = np.flatnonzero(sel)[::paso]
        coef, *_ = np.linalg.lstsq(A_full[idx], z[idx], rcond=None)
        fit = A_full @ coef
        r = z - fit
        mad = 1.4826 * np.median(np.abs(r[idx])) + 1e-6
        sel = valido.reshape(-1) & (np.abs(r) < 2.5 * mad)
    return fit.reshape(h, w)


def mapa_lesion(bgr, piel):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[..., 0] * 100 / 255, lab[..., 1] - 128, lab[..., 2] - 128
    # piel de referencia: excluye el tercio más oscuro para que la lesión no sesgue el ajuste
    ref = piel & (L > np.percentile(L[piel], 30))
    dL = ajustar_superficie(L, ref) - L          # >0: más oscuro que la piel
    da = a - ajustar_superficie(a, ref)
    db = b - ajustar_superficie(b, ref)
    def z(x):
        v = x[ref]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-3
        return (x - med) / mad
    zL, za, zb = z(dL), z(da), z(db)
    # lesionalidad: oscurecimiento dominante + pigmentación (b, a) cuando acompaña oscurecimiento
    d = np.maximum(zL, 0) + 0.5 * np.maximum(zb, 0) * (zL > 0) + 0.25 * np.maximum(za, 0) * (zL > 0)
    d[~piel] = 0
    return cv2.GaussianBlur(d, (0, 0), 2.0), dL


def segmentar_v2(bgr_orig, refinar=True, devolver_debug=False, quitar_vello_fn=None):
    """Devuelve (máscara uint8 0/255 del tamaño original, info). quitar_vello_fn: función opcional
    bgr -> (limpia, mascara_vello, ...) para usar otra remoción de vello (p. ej. B2.23 de la app)."""
    t0 = time.perf_counter()
    H, W = bgr_orig.shape[:2]
    esc = ANCHO / W
    bgr = cv2.resize(bgr_orig, (ANCHO, int(round(H * esc))), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    if quitar_vello_fn is None:
        limpia, vello = quitar_vello(bgr)
    else:
        out = quitar_vello_fn(bgr); limpia, vello = out[0], out[1]
    t_vello = time.perf_counter()
    piel = mascara_piel(limpia)
    d, dL = mapa_lesion(limpia, piel)

    v = d[piel]
    otsu, _ = cv2.threshold(np.clip(v * 10, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(otsu / 10.0, 3.0)
    m = ((d >= t) & piel).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    mejor, pmejor, cand = 0, -1, []
    diag = np.hypot(w, h)
    for i in range(1, n):
        area = st[i, 4]
        if area < 0.002 * h * w:
            continue
        comp = lab == i
        contraste = float(d[comp].mean())
        dc = np.hypot(cen[i][0] - w / 2, cen[i][1] - h / 2) / (0.5 * diag)
        x, y, bw, bh = st[i, :4]
        bordes = int(x <= 1) + int(y <= 1) + int(x + bw >= w - 1) + int(y + bh >= h - 1)
        compacidad = area / max(1.0, bw * bh)
        p = contraste * np.sqrt(area) * np.exp(-1.5 * dc * dc) * (0.35 ** bordes) * (0.5 + compacidad)
        cand.append((p, i))
        if p > pmejor:
            pmejor, mejor = p, i
    if mejor == 0:
        sel = np.zeros((h, w), np.uint8)
    else:
        sel = (lab == mejor).astype(np.uint8)
        if refinar:
            gc = np.full((h, w), cv2.GC_BGD, np.uint8)
            anillo = cv2.dilate(sel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
            gc[anillo > 0] = cv2.GC_PR_BGD
            gc[sel > 0] = cv2.GC_PR_FGD
            nucleo = cv2.erode(sel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            gc[nucleo > 0] = cv2.GC_FGD
            gc[~piel] = cv2.GC_BGD
            try:
                bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
                cv2.grabCut(limpia, gc, None, bg, fg, 3, cv2.GC_INIT_WITH_MASK)
                r = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
                n2, l2, s2, _ = cv2.connectedComponentsWithStats(r, 8)
                if n2 > 1:
                    # componente de GrabCut que más se solapa con la semilla
                    sol = [((l2 == j) & (sel > 0)).sum() for j in range(1, n2)]
                    r = (l2 == 1 + int(np.argmax(sol))).astype(np.uint8)
                    # aceptar el refinamiento solo si no explota ni colapsa
                    if 0.5 * sel.sum() <= r.sum() <= 2.2 * sel.sum():
                        sel = r
            except cv2.error:
                pass
        sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cs, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sel = np.zeros_like(sel); cv2.drawContours(sel, cs, -1, 1, -1)
    sel = (cv2.GaussianBlur(sel.astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.uint8)   # borde suave
    mask_pre = cv2.resize(sel * 255, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_pre = (cv2.GaussianBlur(mask_pre, (0, 0), 1.5) > 127).astype(np.uint8) * 255
    mask = cv2.dilate(mask_pre, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))    # DILATE_5
    info = {"tiempo_s": time.perf_counter() - t0, "tiempo_vello_s": t_vello - t0, "umbral": t, "candidatos": len(cand),
            "mascara_pre_dilate": mask_pre,
            "imagen_limpia": cv2.resize(limpia, (W, H), interpolation=cv2.INTER_LINEAR),
            "mascara_vello": cv2.resize(vello, (W, H), interpolation=cv2.INTER_NEAREST),
            "mapa_lesion": cv2.resize(np.clip(d / 10.0, 0, 1).astype(np.float32), (W, H)),
            "contraste_medio": float(d[sel > 0].mean()) if sel.any() else 0.0,
            "fraccion_vello": float((vello > 0).mean())}
    if devolver_debug:
        info.update({"d": d, "piel": piel, "vello": vello, "limpia": limpia})
    return mask, info
