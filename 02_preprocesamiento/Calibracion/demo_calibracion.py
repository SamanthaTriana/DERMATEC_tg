"""Calibración de color del capilógrafo (campo plano + balance de grises) a partir de fotos de la carta.
Uso:  python demo_calibracion.py fotos_carta/blanco_G1.jpg fotos_carta/gris_1.jpg fotos_carta/gris_2.jpg fotos_carta/gris_3.jpg
Genera calibracion_color.json y calibracion_color_campo.npz junto a este archivo y guarda antes_despues.png.
La escala (px/mm) está en calibracion_escala.json y en la app se calibra con MENÚ → Calibrar escala."""
import sys
import cv2, numpy as np
import calibracion_color as cc

blanco, grises = sys.argv[1], sys.argv[2:]
cfg = cc.construir(blanco, grises)
print("Ganancias BGR:", cfg["ganancias_bgr"], "| iluminación esquinas/centro:", cfg["iluminacion_esquinas_vs_centro"])
img = cc._leer(blanco)
corr = cc.aplicar(img)[0]
cv2.imwrite("antes_despues.png", np.hstack([img, corr]))
print("Guardado antes_despues.png (izquierda: original, derecha: corregida)")
