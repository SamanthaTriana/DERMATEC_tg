"""
DERMATEC — Clasificador final para App V6 y Raspberry Pi 5
===========================================================
Modelo: el/los .tflite listados en config_dermatec.json (versión final: ensamble Z.18 + Z.19,
EfficientNetV2-B0 + MobileNetV3-Large a 384 px; se promedian las probabilidades; TTA por modelo según el config).

Regla de decisión (congelada en Validation):
    p = promedio de los modelos (cada uno con o sin TTA de 4 volteos, según su calibración)
    P(maligno) = 1 − P(Nevo);  MALIGNA si P ≥ umbral  (umbral elegido para sens ≥ 0,95 en Validation)
    salida jerárquica: BENIGNA → Nevo; MALIGNA → argmax(log p + sesgo) entre Melanoma, BCC y SCC

Control de calidad de la captura. Si falla, se añade el aviso "baja confianza: repetir la captura"
(el resultado se muestra igual; con calidad.modo = "bloqueo" pasaría a NO CONCLUYENTE):
  1. Fuera de dominio: entropía de la predicción > percentil 95 de la entropía en Validation
     (el modelo está más dudoso que en el 95 % de las imágenes dermatoscópicas conocidas).
  2. Encuadre: con la máscara de la Rama 1, la lesión ocupa más del 60 % del campo o toca
     2 o más bordes (la lesión no cabe completa: capilógrafo demasiado cerca).

Misma interfaz que el antiguo DermatecJ21/DermatecZ23: predict() y predict_lesion_results().
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

NOMBRES = ["Nevo", "Melanoma", "Carcinoma basocelular", "Carcinoma escamocelular"]
SOSPECHOSO = "Nevo — sospechoso, revisar"
NO_CONCLUYENTE = "NO CONCLUYENTE"


class DermatecFinal:
    def __init__(self, carpeta=None, hilos=4, tta=None):
        self.carpeta = Path(carpeta) if carpeta else Path(__file__).resolve().parent
        self.cfg = json.loads((self.carpeta / "config_dermatec.json").read_text(encoding="utf-8"))
        self.nombre = self.cfg.get("nombre", "DERMATEC")
        self.sesgo = np.asarray(self.cfg["sesgo"], dtype=np.float64)
        self.threshold = float(self.cfg["umbral_p_maligno"])
        self.tta = bool(self.cfg.get("tta", True)) if tta is None else tta
        cal = self.cfg.get("calidad", {})
        self.h_max = float(cal.get("entropia_max_bits", self.cfg.get("entropia_p95_bits", 99.0)))
        self.area_max = float(cal.get("area_lesion_max", 0.60))
        self.bordes_max = int(cal.get("bordes_tocados_max", 1))
        # "advertencia" (por defecto): siempre da MALIGNA/BENIGNA y añade un aviso de baja confianza.
        # "bloqueo": reemplaza el resultado por NO CONCLUYENTE.
        self.modo_calidad = str(cal.get("modo", "advertencia")).lower()
        self.modelos = []
        for m in self.cfg["modelos"]:
            ruta = self.carpeta / m["archivo"]
            if not ruta.is_file():
                raise FileNotFoundError(f"Falta el modelo {ruta}")
            it = Interpreter(model_path=str(ruta), num_threads=hilos)
            it.allocate_tensors()
            ent = it.get_input_details()[0]
            self.modelos.append((m.get("id", ruta.stem), it, int(ent["shape"][1]), ent["index"],
                                 it.get_output_details()[0]["index"], m.get("tta")))
        self._param_mod, self._param_ref = self._cargar_parametros()

    # ------------------------------------------------------------------
    # Parámetros dermatoscópicos (funciones de J2.1 copiadas en dermatec_parametros.py, sin modelos)
    # ------------------------------------------------------------------
    def _cargar_parametros(self):
        """1) módulo local dermatec_parametros.py + parametros_referencia.json (sin TensorFlow);
        2) si no están, el paquete DERMATEC_J2_1_VISUAL junto a la app (versiones anteriores)."""
        try:
            if str(self.carpeta) not in sys.path:
                sys.path.insert(0, str(self.carpeta))
            mod = importlib.import_module("dermatec_parametros")
            ref = json.loads((self.carpeta / "parametros_referencia.json").read_text(encoding="utf-8"))
            return mod, ref.get("dermatoscopic_parameters", {})
        except Exception as exc_local:
            raiz = self.carpeta.parent
            cfg = raiz / "DERMATEC_J2_1_VISUAL" / "config" / "DERMATEC_J2_1_FINAL_CONFIG.json"
            try:
                if str(raiz) not in sys.path:
                    sys.path.insert(0, str(raiz))
                mod = importlib.import_module("DERMATEC_J2_1_VISUAL.src.dermatec_j21_inferencia")
                ref = json.loads(cfg.read_text(encoding="utf-8")).get("dermatoscopic_parameters", {})
                return mod, ref
            except Exception as exc:
                print("Parámetros dermatoscópicos no disponibles:", repr(exc_local), repr(exc))
                return None, {}

    def _indice(self, clave, valor):
        r = self._param_ref.get(clave)
        if r is None or valor is None or not np.isfinite(valor):
            return None
        p05, p95 = float(r["p05"]), float(r["p95"])
        if abs(p95 - p05) < 1e-12:
            return 50.0
        return float(np.clip((valor - p05) / (p95 - p05) * 100.0, 0, 100))

    def _parametros(self, rgb, mascara=None):
        """Devuelve (parámetros, info de segmentación, máscara o None).
        Si llega la máscara de la Rama 1, los parámetros se calculan con ELLA (misma lesión que se muestra en la app);
        si no, con la segmentación rápida del paquete J2.1."""
        if self._param_mod is None:
            return {}, {"valida": False, "calidad": 0.0}, None
        m = self._param_mod
        if mascara is not None:
            try:
                mk = (np.asarray(mascara) > 0).astype(np.uint8)
                if mk.ndim == 3:
                    mk = mk[..., 0]
                if mk.shape != rgb.shape[:2]:
                    import cv2
                    mk = (cv2.resize(mk, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                if mk.sum() >= 50:     # la app calcula los parámetros con unidades (parametros_lesion.py)
                    return {}, {"valida": True, "calidad": 1.0, "fuente_mascara": "Rama 1"}, mk
            except Exception as exc:
                print("Error calculando parámetros con la máscara de la Rama 1:", repr(exc))
        try:
            import cv2
            crop = m.crop_black_border(rgb)
            seg = m.segment_lesion(m.resize_for_seg(crop))
            h, w = crop.shape[:2]
            mask = (cv2.resize(seg["mask"], (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
            valida = bool(int(seg["valid"]))
            feats = m.extract_features(crop, mask) if valida else {}
        except Exception as exc:
            print("Error calculando parámetros:", repr(exc))
            return {}, {"valida": False, "calidad": 0.0}, None
        return self._formatear(feats), {"valida": valida, "calidad": float(seg.get("quality", 0.0))}, (mask if valida else None)

    def _formatear(self, feats):
        salida = {}
        for clave, r in self._param_ref.items():
            v = feats.get(clave, np.nan)
            v = float(v) if v is not None and np.isfinite(v) else None
            salida[clave] = {
                "nombre": r.get("display_name", clave),
                "valor": v,
                "indice_relativo_0_100": self._indice(clave, v),
                "nota": "Índice relativo respecto al conjunto de entrenamiento. No es probabilidad diagnóstica.",
            }
        return salida

    @staticmethod
    def _encuadre(mask):
        """Fracción del campo ocupada por la lesión y número de bordes que toca (margen 1 %)."""
        h, w = mask.shape
        b = max(2, int(0.01 * min(h, w)))
        bordes = int(mask[:b].any()) + int(mask[-b:].any()) + int(mask[:, :b].any()) + int(mask[:, -b:].any())
        return float(mask.mean()), bordes

    # ------------------------------------------------------------------
    # Clasificador
    # ------------------------------------------------------------------
    @staticmethod
    def _adecuar(im, size):
        """Igual que en entrenamiento: RGB → resize cuadrado bilineal (PIL), float32 [0, 255]."""
        if im.format == "JPEG":
            im.draft("RGB", (size * 2, size * 2))
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)[None]

    def probabilidades(self, ruta):
        cache, probs = {}, {}
        for nombre, it, size, i_in, i_out, tta_m in self.modelos:
            if size not in cache:
                with Image.open(ruta) as im:
                    cache[size] = self._adecuar(im, size)
            x = cache[size]
            usar_tta = self.tta if tta_m is None else bool(tta_m)      # TTA por modelo si el config lo indica
            vistas = [x, x[:, :, ::-1], x[:, ::-1], x[:, ::-1, ::-1]] if usar_tta else [x]
            salida = []
            for v in vistas:
                it.set_tensor(i_in, np.ascontiguousarray(v)); it.invoke()
                salida.append(it.get_tensor(i_out)[0].astype(np.float64))
            probs[nombre] = np.mean(salida, axis=0)
        return np.mean(list(probs.values()), axis=0), {k: v.tolist() for k, v in probs.items()}

    def _decidir(self, p, motivos=()):
        pred = int(np.argmax(np.log(p + 1e-9) + self.sesgo))
        p_mal = float(1.0 - p[0])
        maligna = p_mal >= self.threshold
        entropia = float(-(p * np.log2(p + 1e-12)).sum())
        motivos = list(motivos)
        if entropia > self.h_max:
            motivos.append(f"imagen fuera del dominio de entrenamiento (incertidumbre {entropia:.2f} bits "
                           f"> {self.h_max:.2f})")
        # Salida jerárquica coherente: BENIGNA → Nevo; MALIGNA → la más probable entre Melanoma, BCC y SCC
        # (con el sesgo calibrado en Validation). El binario manda y la clase nunca lo contradice.
        lp = np.log(p + 1e-9) + self.sesgo
        clase = NOMBRES[1 + int(np.argmax(lp[1:]))] if maligna else NOMBRES[0]
        confiable = not motivos
        bloquear = (not confiable) and self.modo_calidad == "bloqueo"
        return {
            "modelo": self.nombre,
            "resultado_binario": NO_CONCLUYENTE if bloquear else ("MALIGNA" if maligna else "BENIGNA"),
            "probabilidad_malignidad": p_mal * 100.0,
            "threshold_malignidad": self.threshold,
            "clasificacion": "No concluyente — repetir la captura o remitir a dermatología" if bloquear else clase,
            "aviso_calidad": "" if confiable else "⚠ Baja confianza: repetir la captura (" + "; ".join(motivos) + ")",
            "clase_mas_probable": clase,
            "resultado_sin_control_calidad": "MALIGNA" if maligna else "BENIGNA",
            "probabilidades_clases": {n: float(v * 100.0) for n, v in zip(NOMBRES, p)},
            "incertidumbre_entropia_bits": entropia,
            "control_calidad": {"aprobado": confiable, "motivos": motivos},
            "advertencia": "DERMATEC es una herramienta de apoyo para evaluación inicial y no sustituye el diagnóstico médico.",
        }

    def predict(self, image_path, mascara=None):
        """mascara (opcional): máscara binaria de la lesión de la Rama 1 (B2.23 → GMM → DILATE_5).
        En la app se pasa siempre; si no se da, se usa la segmentación del paquete J2.1."""
        p, por_modelo = self.probabilidades(image_path)
        with Image.open(image_path) as im:
            rgb = np.asarray(im.convert("RGB"))
        params, seg, mask = self._parametros(rgb, mascara)
        if mascara is not None:
            m = (np.asarray(mascara) > 0).astype(np.uint8)
            if m.ndim == 3:
                m = m[..., 0]
            if m.any():
                mask = m
                seg["fuente_mascara_encuadre"] = "Rama 1"
        motivos = []
        if mask is not None:
            area, bordes = self._encuadre(mask)
            seg.update({"fraccion_campo_lesion": area, "bordes_tocados": bordes})
            if area > self.area_max or bordes > self.bordes_max:
                motivos.append(f"encuadre: la lesión ocupa {100 * area:.0f} % del campo y toca {bordes} borde(s); "
                               "aleje el equipo para que la lesión quede completa con piel alrededor")
        out = self._decidir(p, motivos)
        out["probabilidades_por_modelo"] = por_modelo
        out["_p"] = p.tolist()
        out["_qc"] = bool(out["control_calidad"]["aprobado"])
        out["parametros_dermatoscopicos"], out["segmentacion"] = params, seg
        return out

    def predict_lesion_results(self, resultados):
        """Promedia las imágenes que pasan el control de calidad; si ninguna pasa, promedia todas y deja el aviso."""
        todas = [r for r in resultados if "_p" in r]
        if not todas:
            raise RuntimeError("Resultados sin probabilidades del modelo.")
        buenas = [r for r in todas if r.get("_qc", True)]
        usadas = buenas or todas
        motivos = [] if buenas else ["ninguna imagen de la lesión pasó el control de calidad"]
        gen = self._decidir(np.mean([np.asarray(r["_p"]) for r in usadas], axis=0), motivos)
        n_mal = sum(str(r.get("resultado_binario", "")).upper() == "MALIGNA" for r in todas)
        n_nc = sum(str(r.get("resultado_binario", "")).upper() == NO_CONCLUYENTE for r in todas)
        gen.update({
            "imagenes_analizadas": len(todas),
            "imagenes_usadas_en_resultado": len(buenas),
            "imagenes_malignas": n_mal,
            "imagenes_benignas": len(todas) - n_mal - n_nc,
            "imagenes_no_concluyentes": n_nc,
            "consistencia_maligna": float(n_mal / len(todas) * 100.0),
            "conteo_clases": {c: sum(r.get("clasificacion") == c for r in todas) for c in NOMBRES},
            "metodo_agregacion": "PROMEDIO_PROBABILIDADES de imágenes que pasan control de calidad + REGLA_CALIBRADA",
            "nota_agregacion": ("El resultado general promedia las probabilidades de las imágenes que pasan el "
                                "control de calidad y aplica el sesgo y el umbral calibrados en Validation."),
        })
        return gen


DermatecZ23 = DermatecFinal       # compatibilidad con scripts anteriores


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description="Prueba rápida del clasificador DERMATEC")
    ap.add_argument("imagen")
    a = ap.parse_args()
    t0 = time.perf_counter(); d = DermatecFinal(); t1 = time.perf_counter()
    r = d.predict(a.imagen); t2 = time.perf_counter()
    for k in ("_p", "_qc", "parametros_dermatoscopicos"):
        r.pop(k, None)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"Carga {t1 - t0:.2f} s | predicción {1000 * (t2 - t1):.0f} ms")
