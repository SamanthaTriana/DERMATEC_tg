"""
DERMATEC — Rama CNN: cómo "ve" la imagen el clasificador final (Z.18 EfficientNetV2-B0 y Z.19 MobileNetV3-Large)
=================================================================================================================
Resume en un solo archivo el código de entrada usado en el entrenamiento final (Kaggle) y en el despliegue
(PC y Raspberry Pi). Copiado del notebook final sin cambios de lógica.

 captura (480×640, corregida en color con 03_calibracion en la app)
 └─> adecuar_imagen(): RGB → 384×384 bilineal (PIL) → float32 [0, 255]
 └─> [dentro del modelo] ShadesOfGray(p=6) → preprocesamiento del backbone → CNN
 (en entrenamiento, antes del modelo: aumentar(), con zoom_capilografo())

¿Se normaliza la imagen?
 - NO fuera del modelo: la app solo redimensiona y entrega valores 0–255.
 - SÍ dentro del modelo: (1) constancia de color Shades of Gray (norma de Minkowski p = 6, conserva el brillo
   medio); (2) el preprocesamiento propio del backbone (EfficientNetV2 y MobileNetV3 lo traen integrado).
   Como ambos viajan dentro del .tflite, PC y Pi hacen lo mismo.
 - NO se usan: eliminación de vello, CLAHE, máscara, ROI ni inpainting (auditoría X.1/X.2).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

IMG_SIZE = 384


# 1) Adecuación (idéntica en entrenamiento, app de PC y Raspberry Pi)
def adecuar_imagen(datos_o_ruta, size: int = IMG_SIZE) -> np.ndarray:
    """RGB → resize cuadrado bilineal (PIL). Devuelve float32 (size, size, 3) en [0, 255].
    En entrenamiento el resultado se guardó una vez en caché como JPEG q95."""
    im = Image.open(io.BytesIO(datos_o_ruta) if isinstance(datos_o_ruta, (bytes, bytearray)) else datos_o_ruta)
    if im.format == "JPEG":
        im.draft("RGB", (size * 2, size * 2))  # decodificación JPEG rápida a escala reducida
    return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)


# 2) Shades of Gray — versión NumPy equivalente a la capa Keras (ver shades_of_gray.py)
def shades_of_gray_np(x: np.ndarray, p: int = 6):
    """x: (H, W, 3) o (N, H, W, 3) en [0, 255]. Devuelve (imagen corregida, iluminante, ganancia)."""
    x = np.asarray(x, dtype=np.float32)
    lote = x if x.ndim == 4 else x[None]
    xn = np.clip(lote / 255.0, 1e-6, 1.0)
    ilum = np.power(np.mean(np.power(xn, p), axis=(1, 2), keepdims=True), 1.0 / p)
    ganancia = np.mean(ilum, axis=-1, keepdims=True) / (ilum + 1e-6)
    y = np.clip(lote * ganancia, 0.0, 255.0)
    return (y if x.ndim == 4 else y[0]), ilum.reshape(-1, 3), ganancia.reshape(-1, 3)


# 3) Aumentación en entrenamiento (TensorFlow; se importa solo si se usa)
# Base: 8 simetrías, recorte 80–100 %, brillo ±0,10, contraste 0,85–1,15, saturación 0,8–1,2.
# Modelos finales Z.18/Z.19: con prob. ZOOM_P el recorte suave se reemplaza por zoom_capilografo(): recorte
# cercano al centro de área ZOOM_AREA·lado² (zoom ≈ 1,3×–2,6×) y, con prob. DEGRADAR_P, reducción al 40–70 %
# + JPEG 55–90 (+ desenfoque 3×3 y tinte 0,9–1,1 por canal).
ZOOM_P, ZOOM_AREA, ZOOM_JITTER, DEGRADAR_P, DEGRADAR_BLUR_COLOR = 0.5, (0.15, 0.60), 0.15, 0.5, True


def construir_aumentacion(img_size: int = IMG_SIZE, robustez: bool = True):
    """Devuelve la función aumentar(img) de TensorFlow usada en entrenamiento."""
    import tensorflow as tf

    def zoom_capilografo(img):
        area = tf.random.uniform([], ZOOM_AREA[0], ZOOM_AREA[1])
        lado = tf.cast(tf.sqrt(area) * img_size, tf.int32)
        libre = img_size - lado
        j = tf.cast(ZOOM_JITTER * img_size, tf.int32)
        c = libre // 2
        oy = tf.clip_by_value(c + tf.random.uniform([], -j, j + 1, tf.int32), 0, libre)
        ox = tf.clip_by_value(c + tf.random.uniform([], -j, j + 1, tf.int32), 0, libre)
        img = tf.image.resize(tf.image.crop_to_bounding_box(img, oy, ox, lado, lado), [img_size, img_size])

        def degradar(x):
            f = tf.random.uniform([], 0.4, 0.7)
            pequeno = tf.cast(f * img_size, tf.int32)
            x = tf.image.resize(tf.image.resize(x, [pequeno, pequeno], antialias=True), [img_size, img_size])
            x = tf.image.random_jpeg_quality(tf.clip_by_value(x / 255.0, 0.0, 1.0), 55, 90) * 255.0
            if DEGRADAR_BLUR_COLOR:
                x = tf.cond(tf.random.uniform([]) < 0.5, lambda: tf.nn.avg_pool2d(x[None], 3, 1, "SAME")[0], lambda: x)
                x = tf.clip_by_value(x * tf.random.uniform([1, 1, 3], 0.9, 1.1), 0.0, 255.0)
            return x
        return tf.cond(tf.random.uniform([]) < DEGRADAR_P, lambda: degradar(img), lambda: img)

    def recorte_suave(img):
        lado = tf.cast(tf.random.uniform([], 0.80, 1.0) * img_size, tf.int32)
        return tf.image.resize(tf.image.random_crop(img, [lado, lado, 3]), [img_size, img_size])

    def aumentar(img):
        img = tf.image.rot90(img, tf.random.uniform([], 0, 4, tf.int32))
        img = tf.image.random_flip_left_right(img)
        if robustez:
            img = tf.cond(tf.random.uniform([]) < ZOOM_P, lambda: zoom_capilografo(img), lambda: recorte_suave(img))
        else:
            img = recorte_suave(img)
        img.set_shape([img_size, img_size, 3])
        img = img / 255.0
        img = tf.image.random_brightness(img, 0.10)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(tf.clip_by_value(img, 0.0, 1.0), 0.8, 1.2)
        return tf.clip_by_value(img, 0.0, 1.0) * 255.0

    return aumentar


if __name__ == "__main__":
    import sys
    x = adecuar_imagen(sys.argv[1])
    y, ilum, gan = shades_of_gray_np(x)
    print("entrada:", x.shape, x.dtype, f"[{x.min():.0f}, {x.max():.0f}]", "| medias RGB", x.reshape(-1, 3).mean(0).round(1))
    print("iluminante estimado (R, G, B):", ilum[0].round(3), "| ganancia:", gan[0].round(3))
    print("tras Shades of Gray, medias RGB:", y.reshape(-1, 3).mean(0).round(1))
