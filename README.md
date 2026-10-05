# DERMATEC — Prototipo de dermatoscopio electrónico con IA para apoyo en diagnóstico médico

Trabajo de grado, Ingeniería Electrónica, Universidad Industrial de Santander (2026).
Autores: Samantha Lucía Triana Toloza y Bryan Steven Ayala Riveros.
Director: Jeison Arley Castillo Bohórquez · Codirector: Rodolfo Villamizar Mejía.

DERMATEC adapta un capilógrafo digital comercial con un acople impreso en 3D (distancia de trabajo de 8 cm) y una Raspberry Pi 5. Cada captura se corrige en color y escala; luego se segmenta la lesión y se calculan 18 parámetros en milímetros (regla ABCD), y un ensamble de dos redes convolucionales (EfficientNetV2-B0 + MobileNetV3-Large) estima la probabilidad de malignidad y clasifica la lesión en nevo, melanoma, carcinoma basocelular o carcinoma escamocelular. Es una herramienta de apoyo al especialista, no de diagnóstico.

## Estructura

| Carpeta | Contenido |
|---|---|
| `01_dataset` | Enlace al conjunto DERMATEC_KAGGLE_V2 (17.296 imágenes), hash de verificación, manifiesto con la partición por paciente, auditoría, licencias de las fuentes y código para armar y verificar el conjunto |
| `02_preprocesamiento` | `Parametros_segmentacion_roi`: segmentación y 18 parámetros · `Entrada_cnn`: adecuación de la imagen para la red (384 px, Shades of Gray, aumentación) · `Calibracion`: corrección de color con carta de grises y escala del acople |
| `03_modelos` | `Candidato_final`: notebook de entrenamiento de Z.18 / Z.19 y métricas · `Exportacion`: modelos TFLite, configuración calibrada y script de inferencia |
| `04_interfaz` | Interfaz DERMATEC App para Windows y para Linux (solo el código de la app; las piezas se copian de 02 y 03) |
| `05_raspberry_pi` | `DERMATEC_APP.zip`: aplicación completa para la Raspberry Pi 5, lista para instalar |

## Cómo reproducir

1. **Datos:** descargar el conjunto desde el enlace de `01_dataset` y comprobarlo con `verificar_dataset.py` (hash, conteos y particiones). Para armarlo desde las fuentes originales, usar el notebook de `01_dataset`.
2. **Entrenamiento:** abrir el notebook de `03_modelos/Candidato_final` en Kaggle o Colab con el conjunto cargado; la variable `EXPERIMENTO` selecciona Z.18 o Z.19.
3. **Inferencia:** `python demo_inferencia.py <carpeta_de_imagenes>` dentro de `03_modelos/Exportacion` (requiere `tensorflow`, `pillow`, `numpy`, `opencv-python-headless`).
4. **Procesamiento para el especialista y calibración:** los scripts de `02_preprocesamiento` se pueden correr por separado sobre una carpeta de imágenes.
5. **Aplicación:**
   - Raspberry Pi 5 (Raspberry Pi OS 64 bits): descomprimir `05_raspberry_pi/DERMATEC_APP.zip`, ejecutar `bash pi/instalar_pi.sh` y abrir desde el ícono DERMATEC.
   - Windows: en `04_interfaz/Windows_pc`, copiar las piezas indicadas en `COLOCAR_AQUI.txt`, ejecutar `INSTALAR_WINDOWS.bat` y luego `EJECUTAR_DERMATEC.bat`.

Las capturas del capilógrafo deben tomarse con el acople de 8 cm y la calibración de color activa; las imágenes del conjunto de datos se usan tal cual.

## Resultados del clasificador (conjunto de prueba, 2.146 imágenes)

Exactitud 0,870 · Macro-F1 0,821 · Sensibilidad maligna 0,934 · Especificidad 0,920 · AUC 0,970.

## Datos y licencias

Las imágenes provienen de fuentes públicas (ISIC, HAM10000, DERM12345, HIBA, MILK10k) bajo sus propias licencias; ver `01_dataset`. Este repositorio no está destinado a uso clínico.
