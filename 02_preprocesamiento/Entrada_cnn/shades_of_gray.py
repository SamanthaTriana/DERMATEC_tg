import keras
from keras import layers


@keras.saving.register_keras_serializable(package="dermatec")
class ShadesOfGray(layers.Layer):
    """Constancia de color Shades of Gray (Barata et al., 2015). Entrada y salida en [0, 255]."""
    def __init__(self, p=6, **kw):
        kw.setdefault("dtype", "float32")
        super().__init__(**kw)
        self.p = p

    def call(self, x):
        x = keras.ops.cast(x, "float32")
        xn = keras.ops.clip(x / 255.0, 1e-6, 1.0)
        ilum = keras.ops.power(keras.ops.mean(keras.ops.power(xn, self.p), axis=(1, 2), keepdims=True), 1.0 / self.p)
        ganancia = keras.ops.mean(ilum, axis=-1, keepdims=True) / (ilum + 1e-6)
        return keras.ops.clip(x * ganancia, 0.0, 255.0)

    def get_config(self):
        return {**super().get_config(), "p": self.p}
