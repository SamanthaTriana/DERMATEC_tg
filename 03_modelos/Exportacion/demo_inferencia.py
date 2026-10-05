"""Clasifica con el ensamble final (Z.18 + Z.19) todas las imágenes de una carpeta.
Uso:  python demo_inferencia.py <carpeta_imagenes>
Colab: !pip install tensorflow opencv-python-headless pillow  y luego  !python demo_inferencia.py ejemplos
Las imágenes deben ser capturas con el acople de 8 cm y corregidas en color (02_procesamiento/Calibracion);
imágenes de otro aumento o sin calibrar no son comparables. Usa MODELO_DERMATEC_FINAL (tflite + config_dermatec.json: 384 px, sesgo por clase y umbral 0,545)."""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODELO_DERMATEC_FINAL"))
from dermatec_inferencia import DermatecFinal

modelo = DermatecFinal(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODELO_DERMATEC_FINAL"))
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.*"))):
    r = modelo.predict(f)
    print(f"{os.path.basename(f)}: {r['resultado_binario']} ({r['clase_mas_probable']}) | P(maligno) = {r['probabilidad_malignidad']:.1f} %"
          + (f" | {r['aviso_calidad']}" if r.get('aviso_calidad') else ""))
