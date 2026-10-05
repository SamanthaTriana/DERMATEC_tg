# ================================================================
# DERMATEC — APP V6 (Rama 1 v2 + clasificador final Z.18 + Z.19 + control de calidad de captura)
# ================================================================
#
# Objetivo de esta versión:
# - Mantener SOLO 2 ventanas principales:
#       1) Entorno de captura (cámara LIVE + N imágenes)
#       2) Análisis de lesión (resultado general + resultados individuales)
# - Proteger el trabajo frente a cierres/cambios accidentales.
# - Organizar información por Paciente -> Lesión -> Imágenes -> Análisis.
# - Guardar análisis estructurado en JSON.
# - Exportar un informe PDF consolidado por paciente.
# - Mantener una arquitectura de cámara separada para facilitar la
#   migración posterior a Raspberry Pi / Linux.
#
# IMPORTANTE:
# - En Windows se conserva la ruta DirectShow/PyGrabber que ya funcionó
#   en DERMATEC V4.x, incluyendo el pin físico "Estático" cuando exista.
# - En Linux/Raspberry Pi se deja un backend V4L2/OpenCV funcional para
#   video y captura por software. El SNAP físico del dermatoscopio en
#   Linux debe validarse con el hardware/driver final.
# - V6: la inferencia usa el modelo final (TFLite, ensamble Z.18 + Z.19) de MODELO_DERMATEC_FINAL.
#   Salida: BENIGNA (Nevo) / MALIGNA + clase maligna más probable. Si la captura es dudosa
#   (incertidumbre alta o lesión mal encuadrada) se muestra un aviso ámbar, sin bloquear el resultado.
# - Rama 1 v2 (rama1_v2.py) y parámetros con unidades (parametros_lesion.py); escala en mm por acople
#   (escala_acople.py + calibracion_escala.json).
# - Si el modelo no expone todavía una función multiimagen dedicada,
#   V5.1 usa una agregación TEMPORAL por promedio de probabilidades.
#   Esta función está aislada para sustituirla después por el agregador
#   validado del modelo sin rediseñar la interfaz.
#
# ================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from ctypes import wstring_at

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import uuid
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont


# ================================================================
# 1. IMPORTS ESPECÍFICOS DE WINDOWS — DIRECTSHOW
# ================================================================

ES_WINDOWS = sys.platform.startswith("win")
ES_LINUX = sys.platform.startswith("linux")

if ES_WINDOWS:
    try:
        from comtypes import GUID, client
        from pygrabber.dshow_graph import (
            FilterGraph,
            SampleGrabber,
            SampleGrabberCallback,
        )
        from pygrabber.dshow_core import qedit
        from pygrabber.dshow_ids import (
            clsids,
            MediaTypes,
            MediaSubtypes,
        )
        DIRECTSHOW_DISPONIBLE = True
    except Exception as exc:
        DIRECTSHOW_DISPONIBLE = False
        DIRECTSHOW_IMPORT_ERROR = repr(exc)
else:
    DIRECTSHOW_DISPONIBLE = False
    DIRECTSHOW_IMPORT_ERROR = "DirectShow solo se usa en Windows."


# ================================================================
# 2. CONFIGURACIÓN GENERAL
# ================================================================

APP_VERSION = "6.0"
APP_NAME = "DERMATEC"

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import parametros_lesion          # noqa: E402  parámetros con unidades (Rama 1)
import escala_acople              # noqa: E402  escala px/mm del acople
import calibracion_color          # noqa: E402  calibración de color (campo plano + balance de grises)
DATA_ROOT = BASE_DIR / "DERMATEC_DATA"


def dibujar_regla_mm(img, ancho_original_px):
    """Dibuja una regla milimetrada a lo largo del borde inferior de una imagen PIL ya redimensionada para
    mostrar: marca cada 1 mm, marca larga y número cada 5 mm. Usa la escala del acople activo
    (calibracion_escala.json). Solo afecta la vista: la imagen guardada y la que analiza la red no cambian."""
    try:
        esc = escala_acople.escala_para(int(ancho_original_px))
        ppm = esc["px_por_mm"] * img.width / float(ancho_original_px)
        if ppm < 3:
            return img
        d = ImageDraw.Draw(img)
        x0, x1, y0 = 8, img.width - 8, img.height - 10
        alto = max(5, int(img.height * 0.018))
        n = int((x1 - x0) / ppm)
        for ancho_linea, color in ((4, (0, 0, 0)), (2, (255, 255, 255))):
            d.line([(x0, y0), (x0 + n * ppm, y0)], fill=color, width=ancho_linea)
            for k in range(n + 1):
                xk = x0 + k * ppm
                h = alto * 2 if k % 5 == 0 else alto
                d.line([(xk, y0), (xk, y0 - h)], fill=color, width=ancho_linea)
        for k in range(0, n + 1, 5):
            txt = f"{k}" if k else "0 mm"
            tx, ty = x0 + k * ppm + 3, y0 - alto * 2 - 12
            d.text((tx + 1, ty + 1), txt, fill=(0, 0, 0))
            d.text((tx, ty), txt, fill=(255, 255, 255))
        if esc.get("estado") != "calibrada":
            txt = "escala estimada"
            tx, ty = x1 - 6 * len(txt) - 4, y0 - alto * 2 - 26
            d.text((tx + 1, ty + 1), txt, fill=(0, 0, 0))
            d.text((tx, ty), txt, fill=(255, 210, 90))
    except Exception:
        pass
    return img
DATA_ROOT.mkdir(parents=True, exist_ok=True)

J21_ROOT = BASE_DIR / "DERMATEC_J2_1_VISUAL"          # solo funciones de parámetros
Z23_ROOT = BASE_DIR / "MODELO_DERMATEC_FINAL"         # clasificador final

ARCHIVOS_Z23 = [
    Z23_ROOT / "config_dermatec.json",
    Z23_ROOT / "dermatec_inferencia.py",
]

# Paleta clínica moderna
COLOR_BG = "#F3F7F8"
COLOR_CARD = "#FFFFFF"
COLOR_TEAL = "#239B8F"
COLOR_TEAL_DARK = "#147A71"
COLOR_TEAL_SOFT = "#DFF4F1"
COLOR_BLUE_SOFT = "#EAF3F8"
COLOR_TEXT = "#20363F"
COLOR_MUTED = "#6C7D84"
COLOR_BORDER = "#D7E2E5"
COLOR_GREEN = "#2E7D5B"
COLOR_GREEN_SOFT = "#E6F5EC"
COLOR_RED = "#AD4141"
COLOR_RED_SOFT = "#F9E8E8"
COLOR_AMBER = "#9A5B00"
COLOR_AMBER_SOFT = "#FDF1D8"
COLOR_AMBER = "#A66A19"
COLOR_AMBER_SOFT = "#FFF3DA"
COLOR_DARK_PANEL = "#152126"


# ================================================================
# 3. UTILIDADES GENERALES
# ================================================================


def ahora_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def marca_tiempo() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def limpiar_nombre_archivo(texto: str, fallback: str = "Sin_nombre") -> str:
    texto = (texto or "").strip()
    if not texto:
        texto = fallback
    texto = re.sub(r"[<>:\"/\\|?*]", "_", texto)
    texto = re.sub(r"\s+", "_", texto)
    texto = texto.strip("._ ")
    return texto[:80] or fallback


def leer_json(ruta: Path, default: Any = None) -> Any:
    try:
        with ruta.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def escribir_json(ruta: Path, data: Any) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temp = ruta.with_suffix(ruta.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp.replace(ruta)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def abrir_carpeta(ruta: Path) -> None:
    ruta.mkdir(parents=True, exist_ok=True)
    try:
        if ES_WINDOWS:
            os.startfile(str(ruta))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(ruta)])
        else:
            subprocess.Popen(["xdg-open", str(ruta)])
    except Exception as exc:
        messagebox.showerror("DERMATEC", f"No fue posible abrir la carpeta.\n\n{exc}")


def abrir_archivo(ruta: Path) -> None:
    try:
        if ES_WINDOWS:
            os.startfile(str(ruta))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(ruta)])
        else:
            subprocess.Popen(["xdg-open", str(ruta)])
    except Exception as exc:
        messagebox.showerror("DERMATEC", f"No fue posible abrir el archivo.\n\n{exc}")


def cargar_fuente(tamano: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidatos = []
    if ES_WINDOWS:
        candidatos += [
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        if bold:
            candidatos = [
                Path("C:/Windows/Fonts/seguisb.ttf"),
                Path("C:/Windows/Fonts/arialbd.ttf"),
            ] + candidatos
    else:
        candidatos += [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        ]
        if bold:
            candidatos = [
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
            ] + candidatos

    for ruta in candidatos:
        if ruta.is_file():
            try:
                return ImageFont.truetype(str(ruta), tamano)
            except Exception:
                pass
    return ImageFont.load_default()


def envolver_texto(draw: ImageDraw.ImageDraw, texto: str, fuente, ancho_px: int) -> list[str]:
    palabras = str(texto or "").split()
    if not palabras:
        return [""]
    lineas: list[str] = []
    actual = palabras[0]
    for palabra in palabras[1:]:
        prueba = actual + " " + palabra
        caja = draw.textbbox((0, 0), prueba, font=fuente)
        if caja[2] - caja[0] <= ancho_px:
            actual = prueba
        else:
            lineas.append(actual)
            actual = palabra
    lineas.append(actual)
    return lineas


# ================================================================
# 4. MODELOS DE ESTADO
# ================================================================


@dataclass
class PacienteActivo:
    id: str
    nombre: str
    carpeta: Path


@dataclass
class LesionActiva:
    id: str
    nombre: str
    carpeta: Path


@dataclass
class EstadoSesion:
    paciente: Optional[PacienteActivo] = None
    lesion: Optional[LesionActiva] = None
    imagenes: list[Path] = field(default_factory=list)

    resultados_individuales: list[dict[str, Any]] = field(default_factory=list)
    resultado_general: Optional[dict[str, Any]] = None

    observaciones: dict[str, str] = field(
        default_factory=lambda: {
            "localizacion_anatomica": "",
            "texto": "",
        }
    )

    analisis_generado: bool = False
    analisis_guardado: bool = False
    analisis_sucio: bool = False
    analisis_desactualizado: bool = False

    firma_imagenes_analizadas: list[str] = field(default_factory=list)
    ruta_ultimo_pdf: Optional[Path] = None


# ================================================================
# 5. GESTIÓN DE DATOS: PACIENTES / LESIONES / IMÁGENES / ANÁLISIS
# ================================================================


class GestorDatos:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # IDs
    # ------------------------------------------------------------

    @staticmethod
    def _siguiente_id(carpetas: list[Path], prefijo: str) -> str:
        mayor = 0
        patron = re.compile(rf"^{re.escape(prefijo)}-(\d+)")
        for carpeta in carpetas:
            m = patron.match(carpeta.name)
            if m:
                mayor = max(mayor, int(m.group(1)))
        return f"{prefijo}-{mayor + 1:03d}"

    # ------------------------------------------------------------
    # Pacientes
    # ------------------------------------------------------------

    def listar_pacientes(self) -> list[PacienteActivo]:
        salida: list[PacienteActivo] = []
        for carpeta in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            meta = leer_json(carpeta / "paciente.json", {}) or {}
            pid = str(meta.get("id") or carpeta.name.split("__", 1)[0])
            nombre = str(meta.get("nombre") or carpeta.name.split("__", 1)[-1].replace("_", " "))
            salida.append(PacienteActivo(pid, nombre, carpeta))
        return salida

    def crear_paciente(self, nombre: str) -> PacienteActivo:
        pid = self._siguiente_id([p for p in self.root.iterdir() if p.is_dir()], "P")
        nombre_limpio = limpiar_nombre_archivo(nombre, f"Paciente_{pid[-3:]}")
        carpeta = self.root / f"{pid}__{nombre_limpio}"
        carpeta.mkdir(parents=True, exist_ok=False)
        (carpeta / "informes").mkdir(exist_ok=True)

        escribir_json(
            carpeta / "paciente.json",
            {
                "id": pid,
                "nombre": nombre.strip() or f"Paciente {pid[-3:]}",
                "creado": ahora_iso(),
                "app_version": APP_VERSION,
            },
        )
        return PacienteActivo(pid, nombre.strip() or f"Paciente {pid[-3:]}", carpeta)

    # ------------------------------------------------------------
    # Lesiones
    # ------------------------------------------------------------

    def listar_lesiones(self, paciente: PacienteActivo) -> list[LesionActiva]:
        salida: list[LesionActiva] = []
        for carpeta in sorted([p for p in paciente.carpeta.iterdir() if p.is_dir() and p.name.startswith("L-")]):
            meta = leer_json(carpeta / "lesion.json", {}) or {}
            lid = str(meta.get("id") or carpeta.name.split("__", 1)[0])
            nombre = str(meta.get("nombre") or carpeta.name.split("__", 1)[-1].replace("_", " "))
            salida.append(LesionActiva(lid, nombre, carpeta))
        return salida

    def crear_lesion(self, paciente: PacienteActivo, nombre: str) -> LesionActiva:
        existentes = [p for p in paciente.carpeta.iterdir() if p.is_dir() and p.name.startswith("L-")]
        lid = self._siguiente_id(existentes, "L")
        nombre_limpio = limpiar_nombre_archivo(nombre, f"Lesion_{lid[-3:]}")
        carpeta = paciente.carpeta / f"{lid}__{nombre_limpio}"
        carpeta.mkdir(parents=True, exist_ok=False)
        (carpeta / "imagenes").mkdir(exist_ok=True)
        (carpeta / "analisis_historial").mkdir(exist_ok=True)

        escribir_json(
            carpeta / "lesion.json",
            {
                "id": lid,
                "nombre": nombre.strip() or f"Lesión {lid[-3:]}",
                "creado": ahora_iso(),
                "app_version": APP_VERSION,
            },
        )
        escribir_json(carpeta / "imagenes_manifest.json", {"imagenes": []})
        return LesionActiva(lid, nombre.strip() or f"Lesión {lid[-3:]}", carpeta)

    # ------------------------------------------------------------
    # Imágenes
    # ------------------------------------------------------------

    @staticmethod
    def _carpeta_imagenes(lesion: LesionActiva) -> Path:
        ruta = lesion.carpeta / "imagenes"
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    def leer_manifest(self, lesion: LesionActiva) -> dict[str, Any]:
        ruta = lesion.carpeta / "imagenes_manifest.json"
        data = leer_json(ruta, {"imagenes": []})
        if not isinstance(data, dict):
            data = {"imagenes": []}
        data.setdefault("imagenes", [])
        return data

    def escribir_manifest(self, lesion: LesionActiva, data: dict[str, Any]) -> None:
        escribir_json(lesion.carpeta / "imagenes_manifest.json", data)

    def listar_imagenes(self, lesion: LesionActiva) -> list[Path]:
        carpeta = self._carpeta_imagenes(lesion)
        extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        return sorted([p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in extensiones])

    def _hashes_existentes(self, lesion: LesionActiva) -> set[str]:
        manifest = self.leer_manifest(lesion)
        hashes = {str(x.get("sha256")) for x in manifest.get("imagenes", []) if x.get("sha256")}
        return hashes

    def guardar_frame(
        self,
        lesion: LesionActiva,
        frame_bgr: np.ndarray,
        fuente: str,
    ) -> tuple[Optional[Path], bool]:
        if frame_bgr is None:
            return None, False

        ok, buffer = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        )
        if not ok:
            raise RuntimeError("No fue posible codificar la captura como JPEG.")

        datos = buffer.tobytes()
        hash_img = sha256_bytes(datos)
        if hash_img in self._hashes_existentes(lesion):
            return None, True

        carpeta = self._carpeta_imagenes(lesion)
        fuente_corta = limpiar_nombre_archivo(fuente, "CAPTURA").upper()
        nombre = f"IMG_{marca_tiempo()}_{fuente_corta}.jpg"
        destino = carpeta / nombre
        destino.write_bytes(datos)

        manifest = self.leer_manifest(lesion)
        manifest["imagenes"].append(
            {
                "archivo": destino.name,
                "fuente": fuente,
                "sha256": hash_img,
                "fecha": ahora_iso(),
            }
        )
        self.escribir_manifest(lesion, manifest)
        return destino, False

    def importar_imagen(self, lesion: LesionActiva, origen: Path) -> tuple[Optional[Path], bool]:
        datos = origen.read_bytes()
        hash_img = sha256_bytes(datos)
        if hash_img in self._hashes_existentes(lesion):
            return None, True

        carpeta = self._carpeta_imagenes(lesion)
        ext = origen.suffix.lower() if origen.suffix else ".jpg"
        base = limpiar_nombre_archivo(origen.stem, "imagen")
        destino = carpeta / f"IMP_{marca_tiempo()}_{base}{ext}"
        shutil.copy2(origen, destino)

        manifest = self.leer_manifest(lesion)
        manifest["imagenes"].append(
            {
                "archivo": destino.name,
                "fuente": "IMPORTADA",
                "archivo_original": origen.name,
                "sha256": hash_img,
                "fecha": ahora_iso(),
            }
        )
        self.escribir_manifest(lesion, manifest)
        return destino, False

    def eliminar_imagen(self, lesion: LesionActiva, ruta: Path) -> None:
        try:
            if ruta.is_file():
                ruta.unlink()
        finally:
            manifest = self.leer_manifest(lesion)
            manifest["imagenes"] = [x for x in manifest.get("imagenes", []) if x.get("archivo") != ruta.name]
            self.escribir_manifest(lesion, manifest)

    def fuente_imagen(self, lesion: LesionActiva, nombre: str) -> str:
        manifest = self.leer_manifest(lesion)
        for item in manifest.get("imagenes", []):
            if item.get("archivo") == nombre:
                return str(item.get("fuente") or "DESCONOCIDA")
        return "DESCONOCIDA"

    # ------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------

    @staticmethod
    def ruta_analisis(lesion: LesionActiva) -> Path:
        return lesion.carpeta / "analisis.json"

    def cargar_analisis(self, lesion: LesionActiva) -> Optional[dict[str, Any]]:
        data = leer_json(self.ruta_analisis(lesion), None)
        return data if isinstance(data, dict) else None

    def guardar_analisis(self, lesion: LesionActiva, data: dict[str, Any]) -> Path:
        ruta = self.ruta_analisis(lesion)
        if ruta.is_file():
            historial = lesion.carpeta / "analisis_historial"
            historial.mkdir(exist_ok=True)
            destino = historial / f"analisis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(ruta, destino)
        escribir_json(ruta, data)
        return ruta

    # ------------------------------------------------------------
    # PDF consolidado
    # ------------------------------------------------------------

    def exportar_pdf_paciente(self, paciente: PacienteActivo) -> Path:
        lesiones = self.listar_lesiones(paciente)
        analisis_validos: list[tuple[LesionActiva, dict[str, Any]]] = []
        for lesion in lesiones:
            data = self.cargar_analisis(lesion)
            if data:
                analisis_validos.append((lesion, data))

        if not analisis_validos:
            raise RuntimeError("El paciente todavía no tiene análisis guardados para exportar.")

        carpeta_informes = paciente.carpeta / "informes"
        carpeta_informes.mkdir(exist_ok=True)
        pdf = carpeta_informes / f"Informe_{paciente.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        W, H = 1240, 1754  # A4 aproximado a 150 dpi
        margen = 70
        fuente_titulo = cargar_fuente(42, bold=True)
        fuente_h1 = cargar_fuente(28, bold=True)
        fuente_h2 = cargar_fuente(21, bold=True)
        fuente_normal = cargar_fuente(18, bold=False)
        fuente_peq = cargar_fuente(14, bold=False)
        fuente_peq_b = cargar_fuente(14, bold=True)

        paginas: list[Image.Image] = []

        # Portada/resumen
        pagina = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(pagina)
        d.rounded_rectangle((margen, 60, W - margen, 190), radius=22, fill="#E6F5F2")
        d.text((margen + 30, 88), "DERMATEC", font=fuente_titulo, fill="#176F68")
        d.text((margen + 30, 145), "Informe consolidado de análisis de lesiones", font=fuente_h2, fill="#20363F")

        y = 240
        d.text((margen, y), "Paciente", font=fuente_h1, fill="#20363F")
        y += 48
        d.text((margen, y), f"ID: {paciente.id}", font=fuente_normal, fill="#20363F")
        y += 34
        d.text((margen, y), f"Nombre/identificación: {paciente.nombre}", font=fuente_normal, fill="#20363F")
        y += 34
        d.text((margen, y), f"Fecha del informe: {datetime.now().strftime('%d/%m/%Y %H:%M')}", font=fuente_normal, fill="#20363F")
        y += 64

        d.text((margen, y), "Lesiones incluidas", font=fuente_h1, fill="#20363F")
        y += 52
        for lesion, data in analisis_validos:
            gen = data.get("resultado_general", {}) or {}
            decision = str(gen.get("resultado_binario", "—"))
            clase = str(gen.get("clasificacion", "—"))
            texto = f"{lesion.id} — {lesion.nombre}: {decision} / {clase}"
            d.text((margen + 15, y), "• " + texto, font=fuente_normal, fill="#20363F")
            y += 36

        y = H - 190
        nota = (
            "DERMATEC es una herramienta de apoyo para la evaluación inicial de lesiones cutáneas. "
            "Los resultados del modelo no sustituyen la valoración ni el diagnóstico médico."
        )
        for linea in envolver_texto(d, nota, fuente_peq, W - 2 * margen):
            d.text((margen, y), linea, font=fuente_peq, fill="#6C7D84")
            y += 24
        paginas.append(pagina)

        # Páginas por lesión
        for lesion, data in analisis_validos:
            gen = data.get("resultado_general", {}) or {}
            individuales = data.get("resultados_individuales", []) or []
            observ = data.get("observaciones", {}) or {}

            pagina = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(pagina)
            d.rectangle((0, 0, W, 120), fill="#E6F5F2")
            d.text((margen, 38), f"{lesion.id} — {lesion.nombre}", font=fuente_h1, fill="#176F68")

            y = 155
            d.text((margen, y), "Resultado general", font=fuente_h1, fill="#20363F")
            y += 48
            decision = str(gen.get("resultado_binario", "—"))
            color_dec = "#AD4141" if decision.upper() == "MALIGNA" else ("#9A5B00" if "NO CONCLUYENTE" in decision.upper() else "#2E7D5B")
            d.text((margen, y), decision, font=fuente_titulo, fill=color_dec)
            y += 58
            d.text((margen, y), f"Clasificación sugerida: {gen.get('clasificacion', '—')}", font=fuente_normal, fill="#20363F")
            y += 34
            if gen.get("aviso_calidad"):
                for linea in envolver_texto(d, str(gen["aviso_calidad"]), fuente_peq_b, W - 2 * margen):
                    d.text((margen, y), linea, font=fuente_peq_b, fill="#9A5B00")
                    y += 24
                y += 10
            pm = gen.get("probabilidad_malignidad")
            if pm is not None:
                d.text((margen, y), f"Probabilidad de malignidad agregada: {float(pm):.2f}%", font=fuente_normal, fill="#20363F")
                y += 34
            d.text((margen, y), f"Imágenes analizadas: {len(individuales)}", font=fuente_normal, fill="#20363F")
            y += 50

            # Hasta 6 miniaturas por página
            col_w = (W - 2 * margen - 30) // 3
            img_h = 210
            for idx, item in enumerate(individuales[:6]):
                fila = idx // 3
                col = idx % 3
                x = margen + col * (col_w + 15)
                y0 = y + fila * 310
                nombre_img = str(item.get("imagen", ""))
                ruta_img = lesion.carpeta / "imagenes" / nombre_img
                if ruta_img.is_file():
                    try:
                        foto = Image.open(ruta_img).convert("RGB")
                        foto.thumbnail((col_w - 10, img_h), Image.Resampling.LANCZOS)
                        fondo = Image.new("RGB", (col_w - 10, img_h), "#F3F7F8")
                        px = (fondo.width - foto.width) // 2
                        py = (fondo.height - foto.height) // 2
                        fondo.paste(foto, (px, py))
                        pagina.paste(fondo, (x, y0))
                    except Exception:
                        pass
                r = item.get("resultado", {}) or {}
                d.text((x, y0 + img_h + 8), f"Imagen {idx + 1}", font=fuente_peq_b, fill="#20363F")
                d.text((x, y0 + img_h + 30), f"{r.get('resultado_binario', '—')} · {r.get('clasificacion', '—')}", font=fuente_peq, fill="#20363F")
                if r.get("probabilidad_malignidad") is not None:
                    d.text((x, y0 + img_h + 52), f"Malignidad: {float(r['probabilidad_malignidad']):.1f}%", font=fuente_peq, fill="#6C7D84")

            obs_y = y + 2 * 310 + 30
            d.text((margen, obs_y), "Observaciones del profesional", font=fuente_h2, fill="#20363F")
            obs_y += 35
            loc = str(observ.get("localizacion_anatomica") or "No registrada")
            d.text((margen, obs_y), f"Localización anatómica: {loc}", font=fuente_peq_b, fill="#20363F")
            obs_y += 30
            texto_obs = str(observ.get("texto") or "Sin observaciones registradas.")
            for linea in envolver_texto(d, texto_obs, fuente_peq, W - 2 * margen):
                d.text((margen, obs_y), linea, font=fuente_peq, fill="#20363F")
                obs_y += 23

            d.text((margen, H - 90), f"DERMATEC V{APP_VERSION} · {data.get('fecha_guardado', '')}", font=fuente_peq, fill="#6C7D84")
            paginas.append(pagina)

            # Anexo de parámetros por lesión
            pagina = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(pagina)
            d.rectangle((0, 0, W, 120), fill="#EEF6FA")
            d.text((margen, 38), f"Parámetros cuantitativos — {lesion.id}", font=fuente_h1, fill="#245B73")
            y2 = 160
            d.text((margen, y2), "Medidas calculadas sobre cada imagen con la máscara de la Rama 1 (apoyo al profesional; no son probabilidad diagnóstica).", font=fuente_peq, fill="#6C7D84")
            y2 += 50

            for i, item in enumerate(individuales):
                res_i = (item.get("resultado", {}) or {})
                params = res_i.get("parametros_lesion", {}) or {}
                if not params:
                    continue
                d.text((margen, y2), f"Imagen {i + 1}: {item.get('imagen', '')}", font=fuente_h2, fill="#20363F")
                y2 += 30
                if res_i.get("escala"):
                    d.text((margen + 15, y2), escala_acople.texto_escala(res_i["escala"]), font=fuente_peq, fill="#6C7D84")
                    y2 += 26
                for clave, p in params.items():
                    if not isinstance(p, dict):
                        continue
                    valor_txt = parametros_lesion.formatear(p.get("valor"), p.get("unidad", ""))
                    d.text((margen + 15, y2), f"{p.get('grupo', '')} · {p.get('nombre', clave)}: {valor_txt}" + ("" if p.get('unidad', '') in ("0–1", "≥ 1", "0–100") or valor_txt == "—" else f" {p.get('unidad', '').split(' ')[0]}"), font=fuente_peq, fill="#20363F")
                    y2 += 24
                    if y2 > H - 120:
                        paginas.append(pagina)
                        pagina = Image.new("RGB", (W, H), "white")
                        d = ImageDraw.Draw(pagina)
                        d.rectangle((0, 0, W, 100), fill="#EEF6FA")
                        d.text((margen, 32), f"Parámetros — {lesion.id} (continuación)", font=fuente_h2, fill="#245B73")
                        y2 = 130
                y2 += 24
                if y2 > H - 150:
                    paginas.append(pagina)
                    pagina = Image.new("RGB", (W, H), "white")
                    d = ImageDraw.Draw(pagina)
                    d.rectangle((0, 0, W, 100), fill="#EEF6FA")
                    d.text((margen, 32), f"Parámetros — {lesion.id} (continuación)", font=fuente_h2, fill="#245B73")
                    y2 = 130
            paginas.append(pagina)

        paginas = [p.convert("RGB") for p in paginas]
        paginas[0].save(
            pdf,
            "PDF",
            resolution=150.0,
            save_all=True,
            append_images=paginas[1:],
        )
        return pdf



# ================================================================
# V5.2 — RAMA 1 EMBEBIDA
# ================================================================
# Flujo:
#   ORIGINAL -> B2.23 -> GMM_LAB_POST_B38 -> DILATE_5 -> ROI
#
# B2.23 y GMM/DILATE proceden directamente de los bloques validados
# del notebook de procesamiento (celdas 83 y 60).
#
# IMPORTANTE:
#   El clasificador NO recibe la imagen modificada por Rama 1.
#   El clasificador recibe la imagen ORIGINAL (resize 288 + Shades of Gray interno).
# ================================================================

from sklearn.mixture import GaussianMixture

# ================================================================
# V5.2 — I/O DE IMAGEN ROBUSTO PARA WINDOWS
# ================================================================
# OpenCV (cv2.imread/imwrite) puede fallar con rutas Windows que
# contienen caracteres Unicode. DERMATEC permite nombres definidos
# por el profesional, por lo que NO debemos depender de ASCII.
# Se usan bytes + imdecode/imencode para lectura/escritura.
# ================================================================

def _tamano_ventana(win, ancho, alto, fraccion=0.90):
    """Geometría 'AxB' que no supera el 90 % de la pantalla (útil en portátiles y en la Pi)."""
    try:
        ancho = min(ancho, int(win.winfo_screenwidth() * fraccion))
        alto = min(alto, int(win.winfo_screenheight() * fraccion))
    except Exception:
        pass
    return f"{ancho}x{alto}"


def leer_imagen_bgr_segura(ruta):
    ruta = Path(ruta)
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    if datos.size == 0:
        raise RuntimeError(f"No se pudo leer el archivo de imagen: {ruta}")
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise RuntimeError(f"No se pudo decodificar la imagen: {ruta}")
    return imagen


def guardar_imagen_segura(ruta, imagen, calidad=96):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    sufijo = ruta.suffix.lower()
    if sufijo in {".jpg", ".jpeg"}:
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(calidad)]
    elif sufijo == ".png":
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        ext = ".png"
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    ok, buffer = cv2.imencode(ext, imagen, params)
    if not ok:
        raise RuntimeError(f"No fue posible codificar la imagen: {ruta.name}")
    buffer.tofile(str(ruta))
    return ruta


SEED_GMM = 42
MAX_SAMPLES = 20000
N_COMPONENTS = 2
N_INIT = 3

def kernel_lineal(
    longitud,
    angulo
):

    if longitud % 2 == 0:

        longitud += 1


    kernel = np.zeros(
        (
            longitud,
            longitud
        ),
        dtype=np.uint8
    )


    centro = longitud // 2

    radio = centro

    theta = np.deg2rad(
        angulo
    )


    x1 = int(
        centro
        -
        radio *
        np.cos(theta)
    )

    y1 = int(
        centro
        -
        radio *
        np.sin(theta)
    )


    x2 = int(
        centro
        +
        radio *
        np.cos(theta)
    )

    y2 = int(
        centro
        +
        radio *
        np.sin(theta)
    )


    cv2.line(
        kernel,
        (x1, y1),
        (x2, y2),
        255,
        1
    )


    return kernel


# ============================================================
# 10. NORMALIZACIÓN ROBUSTA
# ============================================================

def normalizar(
    imagen,
    p1=1,
    p2=99
):

    imagen = imagen.astype(
        np.float32
    )


    a = np.percentile(
        imagen,
        p1
    )

    b = np.percentile(
        imagen,
        p2
    )


    if b <= a:

        return np.zeros_like(
            imagen,
            dtype=np.float32
        )


    return np.clip(
        (
            imagen - a
        )
        /
        (
            b - a
        ),
        0,
        1
    )


# ============================================================
# 11. RESPUESTA GABOR
# ============================================================

def respuesta_gabor(
    gray,
    frecuencias,
    angulos
):

    h, w = gray.shape

    respuesta = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    gray_float = gray.astype(
        np.float32
    ) / 255.0


    for frecuencia in frecuencias:

        lambd = 1.0 / frecuencia


        for theta in angulos:

            kernel = cv2.getGaborKernel(
                (
                    21,
                    21
                ),
                sigma=4.0,
                theta=theta,
                lambd=lambd,
                gamma=0.25,
                psi=0,
                ktype=cv2.CV_32F
            )


            r = cv2.filter2D(
                gray_float,
                cv2.CV_32F,
                kernel
            )


            r = np.abs(
                r
            )


            r = normalizar(
                r,
                85,
                99.7
            )


            respuesta = np.maximum(
                respuesta,
                r
            )


    return respuesta


# ============================================================
# 12. B2.23
# ============================================================

def B223(
    imagen
):

    h, w = imagen.shape[:2]


    # ========================================================
    # A. ESPACIOS DE COLOR
    # ========================================================

    gray = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )


    lab = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2LAB
    )


    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )


    # ========================================================
    # B. CLAHE SUAVE
    # ========================================================

    clahe = cv2.createCLAHE(
        clipLimit=1.8,
        tileGridSize=(
            8,
            8
        )
    )


    gray_c = clahe.apply(
        gray
    )


    # ========================================================
    # C. BLACK-HAT MULTIESCALA
    # ========================================================

    dark_max = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    dark_sum = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    dark_count = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    angulos = list(
        range(
            0,
            180,
            10
        )
    )


    escalas = [
        9,
        13,
        17,
        21,
        27,
        33
    ]


    canales_dark = [

        gray_c,

        lab[:, :, 0],

        hsv[:, :, 2]
    ]


    contador = 0


    for canal in canales_dark:

        for longitud in escalas:

            for angulo in angulos:

                kernel = kernel_lineal(
                    longitud,
                    angulo
                )


                bh = cv2.morphologyEx(
                    canal,
                    cv2.MORPH_BLACKHAT,
                    kernel
                )


                bh = normalizar(
                    bh,
                    75,
                    99.7
                )


                dark_max = np.maximum(
                    dark_max,
                    bh
                )


                dark_sum += bh


                dark_count += (
                    bh > 0.58
                ).astype(
                    np.float32
                )


                contador += 1


    dark_mean = (
        dark_sum
        /
        float(contador)
    )


    dark_persistence = (
        dark_count
        /
        float(contador)
    )


    # ========================================================
    # D. GABOR
    # ========================================================

    gabor = respuesta_gabor(
        gray_c,
        frecuencias=[
            0.07,
            0.10,
            0.14,
            0.18
        ],
        angulos=[
            np.deg2rad(x)
            for x in range(
                0,
                180,
                15
            )
        ]
    )


    # ========================================================
    # E. RESPUESTA DE LÍNEA
    # ========================================================

    line_max = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    line_persistence = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    line_count = 0


    for longitud in [
        11,
        17,
        23,
        29
    ]:

        for angulo in angulos:

            kernel = kernel_lineal(
                longitud,
                angulo
            )


            # CLOSE transforma pequeños huecos
            # en estructuras continuas.

            cerrado = cv2.morphologyEx(
                gray_c,
                cv2.MORPH_CLOSE,
                kernel
            )


            respuesta = (
                cerrado.astype(
                    np.float32
                )
                -
                gray_c.astype(
                    np.float32
                )
            )


            respuesta = np.abs(
                respuesta
            )


            respuesta = normalizar(
                respuesta,
                85,
                99.7
            )


            line_max = np.maximum(
                line_max,
                respuesta
            )


            line_persistence += (
                respuesta > 0.62
            ).astype(
                np.float32
            )


            line_count += 1


    line_persistence /= float(
        line_count
    )


    # ========================================================
    # F. PELO CLARO
    # ========================================================

    light_max = np.zeros(
        (
            h,
            w
        ),
        dtype=np.float32
    )


    for canal in [
        gray_c,
        lab[:, :, 0]
    ]:

        for longitud in [
            11,
            17,
            23,
            29
        ]:

            for angulo in angulos:

                kernel = kernel_lineal(
                    longitud,
                    angulo
                )


                th = cv2.morphologyEx(
                    canal,
                    cv2.MORPH_TOPHAT,
                    kernel
                )


                th = normalizar(
                    th,
                    80,
                    99.7
                )


                light_max = np.maximum(
                    light_max,
                    th
                )


    # ========================================================
    # G. SCORE HÍBRIDO
    # ========================================================

    score_dark = (

        0.35 *
        dark_max

        +

        0.15 *
        dark_mean

        +

        0.15 *
        np.minimum(
            dark_persistence * 7,
            1
        )

        +

        0.20 *
        gabor

        +

        0.15 *
        line_max
    )


    score_light = (

        0.70 *
        light_max

        +

        0.30 *
        line_max
    )


    score = np.maximum(
        score_dark,
        0.65 *
        score_dark
        +
        0.35 *
        score_light
    )


    # ========================================================
    # H. UMBRAL ADAPTATIVO
    # ========================================================

    td = np.percentile(
        score,
        96.0
    )


    mascara = (
        score >= td
    ).astype(
        np.uint8
    ) * 255


    # ========================================================
    # I. CONTINUIDAD LINEAL
    # ========================================================

    mascara_continua = np.zeros_like(
        mascara
    )


    for longitud in [
        13,
        19,
        25
    ]:

        for angulo in [
            0,
            15,
            30,
            45,
            60,
            75,
            90,
            105,
            120,
            135,
            150,
            165
        ]:

            kernel = kernel_lineal(
                longitud,
                angulo
            )


            tmp = cv2.morphologyEx(
                mascara,
                cv2.MORPH_CLOSE,
                kernel
            )


            # Solo conservar recuperación
            # donde ya existe evidencia fuerte.

            evidencia = (
                score >=
                np.percentile(
                    score,
                    93
                )
            ).astype(
                np.uint8
            ) * 255


            tmp = cv2.bitwise_and(
                tmp,
                evidencia
            )


            mascara_continua = cv2.bitwise_or(
                mascara_continua,
                tmp
            )


    mascara = cv2.bitwise_or(
        mascara,
        mascara_continua
    )


    # ========================================================
    # J. FILTRADO GEOMÉTRICO
    # ========================================================

    n, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mascara,
            8
        )
    )


    filtrada = np.zeros_like(
        mascara
    )


    for i in range(
        1,
        n
    ):

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]


        ww = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]


        hh = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]


        mayor = max(
            ww,
            hh
        )


        menor = max(
            1,
            min(
                ww,
                hh
            )
        )


        elongacion = (
            mayor /
            menor
        )


        # Pelo claramente alargado.

        if (
            mayor >= 14
            and
            elongacion >= 2.0
            and
            area >= 12
        ):

            filtrada[
                labels == i
            ] = 255

            continue


        # Pelo largo aunque esté fragmentado.

        if (
            mayor >= 25
            and
            elongacion >= 1.65
            and
            area >= 15
        ):

            filtrada[
                labels == i
            ] = 255


    mascara = filtrada


    # ========================================================
    # K. RECUPERAR SEGMENTOS DELGADOS
    # ========================================================

    mascara_delgada = np.zeros_like(
        mascara
    )


    for angulo in [
        0,
        15,
        30,
        45,
        60,
        75,
        90,
        105,
        120,
        135,
        150,
        165
    ]:

        kernel = kernel_lineal(
            9,
            angulo
        )


        tmp = cv2.morphologyEx(
            mascara,
            cv2.MORPH_CLOSE,
            kernel
        )


        mascara_delgada = cv2.bitwise_or(
            mascara_delgada,
            tmp
        )


    # Evidencia fuerte solamente.

    evidencia_fuerte = (
        score >=
        np.percentile(
            score,
            95
        )
    ).astype(
        np.uint8
    ) * 255


    mascara_delgada = cv2.bitwise_and(
        mascara_delgada,
        evidencia_fuerte
    )


    mascara = cv2.bitwise_or(
        mascara,
        mascara_delgada
    )


    # ========================================================
    # L. ENGROSAMIENTO ADAPTATIVO
    # ========================================================

    # Primero una dilatación moderada.

    mascara = cv2.dilate(
        mascara,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                3,
                3
            )
        ),
        iterations=1
    )


    # ========================================================
    # M. LIMPIEZA FINAL
    # ========================================================

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                3,
                3
            )
        ),
        iterations=1
    )


    # ========================================================
    # N. RECONSTRUCCIÓN MULTIETAPA
    # ========================================================

    # --------------------------------------------------------
    # PASO 1
    # Reparación de la estructura central.
    # --------------------------------------------------------

    paso1 = cv2.inpaint(
        imagen,
        mascara,
        3,
        cv2.INPAINT_TELEA
    )


    # --------------------------------------------------------
    # PASO 2
    # Ampliar ligeramente la máscara.
    # --------------------------------------------------------

    mascara_expandida = cv2.dilate(
        mascara,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                5,
                5
            )
        ),
        iterations=1
    )


    # --------------------------------------------------------
    # Evitar que el segundo paso borre demasiado.
    #
    # La expansión solo se aplica alrededor de las zonas
    # donde realmente existe evidencia de pelo.
    # --------------------------------------------------------

    evidencia = (
        score >=
        np.percentile(
            score,
            92
        )
    ).astype(
        np.uint8
    ) * 255


    mascara_expandida = cv2.bitwise_and(
        mascara_expandida,
        evidencia
    )


    mascara_expandida = cv2.bitwise_or(
        mascara_expandida,
        mascara
    )


    # --------------------------------------------------------
    # PASO 2 — NS
    # --------------------------------------------------------

    paso2 = cv2.inpaint(
        paso1,
        mascara_expandida,
        7,
        cv2.INPAINT_NS
    )


    # --------------------------------------------------------
    # PASO 3 — TELEA
    #
    # Segunda reconstrucción únicamente sobre el núcleo.
    # --------------------------------------------------------

    salida = cv2.inpaint(
        paso2,
        mascara,
        4,
        cv2.INPAINT_TELEA
    )


    return (
        salida,
        mascara,
        score
    )

def gmm_lab_post_b38(
    image_bgr
):

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )


    h, w = lab.shape[:2]


    pixels = lab.reshape(
        -1,
        3
    ).astype(
        np.float32
    )


    rng = np.random.default_rng(
        SEED_GMM
    )


    if len(pixels) > MAX_SAMPLES:

        idx = rng.choice(
            len(pixels),
            size=MAX_SAMPLES,
            replace=False
        )

        train_pixels = pixels[
            idx
        ]

    else:

        train_pixels = pixels


    gmm = GaussianMixture(

        n_components=
            N_COMPONENTS,

        covariance_type=
            "full",

        random_state=
            SEED_GMM,

        n_init=
            N_INIT
    )


    gmm.fit(
        train_pixels
    )


    labels = gmm.predict(
        pixels
    )


    means = gmm.means_


    # EXACTAMENTE igual a 3.10.1.2:
    # componente con menor media L.

    lesion_component = int(
        np.argmin(
            means[:, 0]
        )
    )


    mask = (
        labels
        ==
        lesion_component
    ).astype(
        np.uint8
    )


    mask = (
        mask.reshape(
            h,
            w
        )
        *
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # POSTPROCESO EXACTO
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            5,
            5
        )
    )


    # CLOSE 5x5 x2

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )


    # OPEN 5x5 x1

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # Largest connected component

    num_labels, labels_cc, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )


    if num_labels > 1:

        areas = stats[
            1:,
            cv2.CC_STAT_AREA
        ]

        largest_idx = (
            1
            +
            np.argmax(
                areas
            )
        )


        clean_mask = np.zeros_like(
            mask
        )


        clean_mask[
            labels_cc
            ==
            largest_idx
        ] = 255


        mask = clean_mask


    # CLOSE 5x5 x1

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    return mask.astype(
        np.uint8
    )


# ============================================================
# 8. D5 EXACTO
# ============================================================

def aplicar_dilate_5(
    mask
):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            5,
            5
        )
    )


    return cv2.dilate(
        mask,
        kernel,
        iterations=1
    ).astype(
        np.uint8
    )


# ============================================================
# 9. MÉTRICAS
# ============================================================

def _rama1_roi_y_parametros(original_bgr, mask_final):
    """Calcula ROI, contorno y parámetros técnicos de la Rama 1."""
    mask = (mask_final > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return {
            "valida": False,
            "calidad": 0.0,
            "roi": np.zeros_like(original_bgr),
            "roi_mask": np.zeros_like(mask),
            "contorno": None,
            "parametros": {},
        }

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    clean = np.zeros_like(mask)
    clean[labels == idx] = 255

    area = float(cv2.countNonZero(clean))
    ys, xs = np.where(clean > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea) if contours else None

    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

    hull_area = float(cv2.contourArea(cv2.convexHull(contour))) if contour is not None else 0.0
    solidity = float(area / hull_area) if hull_area > 0 else 0.0
    extent = float(area / (bw * bh)) if bw > 0 and bh > 0 else 0.0
    aspect = float(max(bw, bh) / max(1, min(bw, bh)))
    equivalent_diameter = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0

    M = cv2.moments(clean, binaryImage=True)
    cx = float(M["m10"] / M["m00"]) if M["m00"] else 0.0
    cy = float(M["m01"] / M["m00"]) if M["m00"] else 0.0

    pad = max(4, int(0.03 * max(bw, bh)))
    xa, xb = max(0, x0 - pad), min(w, x1 + pad + 1)
    ya, yb = max(0, y0 - pad), min(h, y1 + pad + 1)

    roi = original_bgr[ya:yb, xa:xb].copy()
    roi_mask = clean[ya:yb, xa:xb].copy()
    roi_segmentada = np.zeros_like(roi)
    roi_segmentada[roi_mask > 0] = roi[roi_mask > 0]

    mask_fraction = float(area / (h * w)) if h and w else 0.0

    parametros = {
        "area_px": area,
        "area_fraction": mask_fraction,
        "bbox_width_px": float(bw),
        "bbox_height_px": float(bh),
        "perimeter_px": perimeter,
        "circularity": circularity,
        "solidity": solidity,
        "extent": extent,
        "aspect_ratio": aspect,
        "equivalent_diameter_px": equivalent_diameter,
        "centroid_x_px": cx,
        "centroid_y_px": cy,
        "roi_width_px": float(roi.shape[1]),
        "roi_height_px": float(roi.shape[0]),
        "componentes_conectados": int(n - 1),
    }

    calidad = float(np.clip(
        0.45 * min(1.0, area / (0.02 * h * w)) +
        0.30 * min(1.0, solidity) +
        0.25 * min(1.0, extent),
        0.0, 1.0
    ))

    return {
        "valida": bool(area > 0 and contour is not None),
        "calidad": calidad,
        "roi": roi_segmentada,
        "roi_mask": roi_mask,
        "contorno": contour,
        "parametros": parametros,
    }


# Versión de la Rama 1: "v2" (robusta a acople largo y cercano, ~0,2 s) o "v1" (B2.23 + GMM Lab + DILATE_5, ~6 s).
RAMA1_VERSION = os.environ.get("DERMATEC_RAMA1", "v2")


def ejecutar_rama1(original_bgr):
    """Ejecuta la Rama 1 y devuelve artefactos + parámetros (mismas claves en v1 y v2)."""
    if original_bgr is None:
        raise ValueError("La imagen original es None.")
    if RAMA1_VERSION == "v2":
        try:
            return ejecutar_rama1_v2(original_bgr)
        except ImportError:
            pass
    return ejecutar_rama1_v1(original_bgr)


def ejecutar_rama1_v2(original_bgr):
    """Rama 1 v2: vello (DullRazor) → piel válida → corrección de sombreado → mapa Lab → Otsu → selección
    por puntaje → GrabCut → DILATE_5. Ver rama1_v2.py."""
    import sys as _sys
    _aqui = str(Path(__file__).resolve().parent)
    if _aqui not in _sys.path:
        _sys.path.insert(0, _aqui)
    import rama1_v2
    t0 = time.perf_counter()
    mask_final, info = rama1_v2.segmentar_v2(original_bgr)
    t_seg = time.perf_counter()
    datos = _rama1_roi_y_parametros(original_bgr, mask_final)
    t_roi = time.perf_counter()
    parametros = dict(datos["parametros"])
    parametros.update({
        "rama1_version": "v2",
        "resolucion_original_px": f"{original_bgr.shape[1]}x{original_bgr.shape[0]}",
        "tiempo_b223_s": float(info["tiempo_vello_s"]),
        "tiempo_gmm_s": float(info["tiempo_s"] - info["tiempo_vello_s"]),
        "tiempo_dilate_s": 0.0,
        "tiempo_roi_parametros_s": float(t_roi - t_seg),
        "tiempo_total_rama1_s": float(t_roi - t0),
        "b223_mask_fraction": float(info["fraccion_vello"]),
        "gmm_mask_fraction": float(cv2.countNonZero(info["mascara_pre_dilate"]) / info["mascara_pre_dilate"].size),
        "final_mask_fraction": float(cv2.countNonZero(mask_final) / mask_final.size),
        "contraste_lesion_piel": float(info["contraste_medio"]),
    })
    return {
        "imagen_limpia_b223": info["imagen_limpia"],
        "mascara_b223": info["mascara_vello"],
        "score_b223": info["mapa_lesion"],
        "mascara_gmm": info["mascara_pre_dilate"],
        "mascara_final": mask_final,
        "roi": datos["roi"],
        "roi_mask": datos["roi_mask"],
        "contorno": datos["contorno"],
        "valida": datos["valida"],
        "calidad": datos["calidad"],
        "parametros": parametros,
    }


def ejecutar_rama1_v1(original_bgr):
    """Rama 1 v1 (original): B2.23 → GMM Lab → DILATE_5 sobre la imagen completa."""

    t0 = time.perf_counter()

    limpia, mascara_b223, score_b223 = B223(original_bgr)
    t_b223 = time.perf_counter()

    mascara_gmm = gmm_lab_post_b38(limpia)
    t_gmm = time.perf_counter()

    mask_final = aplicar_dilate_5(mascara_gmm)
    t_dilate = time.perf_counter()

    datos = _rama1_roi_y_parametros(original_bgr, mask_final)
    t_roi = time.perf_counter()

    parametros = dict(datos["parametros"])
    parametros.update({
        "resolucion_original_px": f"{original_bgr.shape[1]}x{original_bgr.shape[0]}",
        "tiempo_b223_s": float(t_b223 - t0),
        "tiempo_gmm_s": float(t_gmm - t_b223),
        "tiempo_dilate_s": float(t_dilate - t_gmm),
        "tiempo_roi_parametros_s": float(t_roi - t_dilate),
        "tiempo_total_rama1_s": float(t_roi - t0),
        "b223_mask_fraction": float(cv2.countNonZero(mascara_b223) / mascara_b223.size),
        "gmm_mask_fraction": float(cv2.countNonZero(mascara_gmm) / mascara_gmm.size),
        "final_mask_fraction": float(cv2.countNonZero(mask_final) / mask_final.size),
    })

    return {
        "imagen_limpia_b223": limpia,
        "mascara_b223": mascara_b223,
        "score_b223": score_b223,
        "mascara_gmm": mascara_gmm,
        "mascara_final": mask_final,
        "roi": datos["roi"],
        "roi_mask": datos["roi_mask"],
        "contorno": datos["contorno"],
        "valida": datos["valida"],
        "calidad": datos["calidad"],
        "parametros": parametros,
    }


# ================================================================
# 6. GESTIÓN DE CÁMARA — BACKEND AISLADO
# ================================================================


class GestorCamara:
    """
    Windows:
        DirectShow + PyGrabber. Conserva el pin Capturar y el pin Estático
        cuando el dispositivo los expone.

    Linux/Raspberry Pi:
        OpenCV/V4L2 para preview/captura por software. El botón SNAP físico
        deberá validarse posteriormente con el driver real del dermatoscopio.
    """

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None],
        on_snap: Callable[[np.ndarray], None],
        on_status: Callable[[str, str, str, bool], None],
    ):
        self.on_frame = on_frame
        self.on_snap = on_snap
        self.on_status = on_status

        self.connected = False
        self.device_name: Optional[str] = None
        self.desired_device_name: Optional[str] = None
        self.resolution_text = "—"
        self.snap_fisico = False

        self.graph = None
        self.camera_filter = None
        self.sample_video = None
        self.callback_video = None
        self.sample_still = None
        self.callback_still = None

        self.cap = None
        self.thread_linux = None
        self.stop_linux = threading.Event()

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_counter = 0

    # ------------------------------------------------------------
    # Enumeración
    # ------------------------------------------------------------

    def listar_dispositivos(self) -> list[tuple[str, Any]]:
        if ES_WINDOWS and DIRECTSHOW_DISPONIBLE:
            try:
                g = FilterGraph()
                nombres = g.get_input_devices()
                return [(str(nombre), idx) for idx, nombre in enumerate(nombres)]
            except Exception:
                return []

        if ES_LINUX:
            # En la Raspberry Pi 5 hay muchos /dev/video* internos (ISP, decodificador). Se listan solo las
            # cámaras USB y, de cada una, el primer nodo (el siguiente suele ser de metadatos y no da imagen).
            dispositivos: list[tuple[str, Any]] = []
            vistos: set[str] = set()
            sysfs = Path("/sys/class/video4linux")
            nodos = sorted(sysfs.glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)) if sysfs.is_dir() else []
            for nodo in nodos:
                idx = int(re.sub(r"\D", "", nodo.name) or 0)
                try:
                    destino = str((nodo / "device").resolve())
                    nombre = (nodo / "name").read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    continue
                if "usb" not in destino:
                    continue
                clave = destino.rsplit(":", 1)[0]
                if clave in vistos:
                    continue
                vistos.add(clave)
                dispositivos.append((f"{nombre} (/dev/video{idx})", idx))
            if not dispositivos:  # respaldo: todos los nodos
                for ruta in sorted(Path("/dev").glob("video*")):
                    m = re.search(r"(\d+)$", ruta.name)
                    if m:
                        dispositivos.append((f"{ruta}", int(m.group(1))))
            return dispositivos

        return []

    # ------------------------------------------------------------
    # Conexión
    # ------------------------------------------------------------

    def conectar(self, identificador: Any, nombre: str, silencioso: bool = False) -> bool:
        self.desconectar()
        self.desired_device_name = nombre
        try:
            if ES_WINDOWS and DIRECTSHOW_DISPONIBLE:
                self._conectar_windows(int(identificador), nombre)
            elif ES_LINUX:
                self._conectar_linux(int(identificador), nombre)
            else:
                raise RuntimeError("No hay backend de cámara disponible para este sistema operativo.")
            return True
        except Exception as exc:
            self.connected = False
            self.device_name = None
            self.snap_fisico = False
            self.resolution_text = "—"
            self.on_status("desconectada", "—", "SNAP no disponible", False)
            if not silencioso:
                messagebox.showerror("DERMATEC — Dispositivo", f"No fue posible conectar la cámara.\n\n{exc}")
            print("ERROR cámara:", repr(exc))
            return False

    # ------------------------------------------------------------
    # Windows DirectShow
    # ------------------------------------------------------------

    @staticmethod
    def _nombre_pin(pin) -> str:
        informacion = pin.QueryPinInfo()
        return wstring_at(informacion.achName)

    def _buscar_pin_salida(self, dispositivo, nombre_buscado: str):
        for pin in dispositivo.out_pins:
            try:
                actual = self._nombre_pin(pin)
            except Exception:
                continue
            if actual.strip().casefold() == nombre_buscado.strip().casefold():
                return pin
        return None

    @staticmethod
    def _obtener_pin_entrada(filtro):
        enumerador = filtro.EnumPins()
        pin, cantidad = enumerador.Next(1)
        while cantidad > 0:
            info = pin.QueryPinInfo()
            if info.dir == 0:
                return pin
            pin, cantidad = enumerador.Next(1)
        return None

    def _crear_null_renderer(self, nombre: str):
        filtro = client.CreateObject(
            GUID(clsids.CLSID_NullRender),
            interface=qedit.IBaseFilter,
        )
        self.graph.filter_graph.AddFilter(filtro, nombre)
        entrada = self._obtener_pin_entrada(filtro)
        if entrada is None:
            raise RuntimeError(f"No se encontró la entrada del filtro {nombre}.")
        return filtro, entrada

    def _callback_frame_windows(self, imagen):
        if imagen is None:
            return
        with self.lock:
            self.latest_frame = imagen.copy()
        self.frame_counter += 1
        self.on_frame(imagen)

    def _callback_snap_windows(self, imagen):
        if imagen is None:
            return
        try:
            if self.callback_still is not None:
                self.callback_still.grab_frame()
        except Exception as exc:
            print("Advertencia rearmando SNAP:", repr(exc))
        self.on_snap(imagen.copy())

    def _conectar_windows(self, indice: int, nombre: str) -> None:
        if not DIRECTSHOW_DISPONIBLE:
            raise RuntimeError(f"DirectShow/PyGrabber no está disponible. {DIRECTSHOW_IMPORT_ERROR}")

        self.graph = FilterGraph()
        dispositivos = self.graph.get_input_devices()
        if indice < 0 or indice >= len(dispositivos):
            raise RuntimeError("Índice de dispositivo no válido.")

        self.graph.add_video_input_device(indice)
        self.camera_filter = self.graph.get_input_device()

        pin_video = self._buscar_pin_salida(self.camera_filter, "Capturar")
        pin_still = self._buscar_pin_salida(self.camera_filter, "Estático")

        if pin_video is None and len(self.camera_filter.out_pins) >= 1:
            pin_video = self.camera_filter.out_pins[0]
        if pin_still is None and len(self.camera_filter.out_pins) >= 2:
            pin_still = self.camera_filter.out_pins[1]

        if pin_video is None:
            raise RuntimeError("El dispositivo no expone un pin de video utilizable.")

        # Rama de video
        self.sample_video = SampleGrabber(self.graph.capture_builder)
        self.callback_video = SampleGrabberCallback(self._callback_frame_windows)
        self.sample_video.set_callback(self.callback_video, 1)
        self.sample_video.set_media_type(MediaTypes.Video, MediaSubtypes.RGB24)
        self.graph.filter_graph.AddFilter(self.sample_video.instance, "DERMATEC Video Grabber")
        _, entrada_null_video = self._crear_null_renderer("DERMATEC Video Null")
        self.graph.graph_builder.Connect(pin_video, self.sample_video.get_in())
        self.graph.graph_builder.Connect(self.sample_video.get_out(), entrada_null_video)
        self.sample_video.initialize_after_connection()
        resolucion = self.sample_video.get_resolution()
        self.resolution_text = f"{resolucion[0]} × {resolucion[1]}"

        # Rama SNAP físico, solo si existe el segundo pin
        self.snap_fisico = False
        if pin_still is not None:
            try:
                self.sample_still = SampleGrabber(self.graph.capture_builder)
                self.callback_still = SampleGrabberCallback(self._callback_snap_windows)
                self.sample_still.set_callback(self.callback_still, 1)
                self.sample_still.set_media_type(MediaTypes.Video, MediaSubtypes.RGB24)
                self.graph.filter_graph.AddFilter(self.sample_still.instance, "DERMATEC Still Grabber")
                _, entrada_null_still = self._crear_null_renderer("DERMATEC Still Null")
                self.graph.graph_builder.Connect(pin_still, self.sample_still.get_in())
                self.graph.graph_builder.Connect(self.sample_still.get_out(), entrada_null_still)
                self.sample_still.initialize_after_connection()
                self.snap_fisico = True
            except Exception as exc:
                print("SNAP físico no disponible en este dispositivo:", repr(exc))
                self.sample_still = None
                self.callback_still = None
                self.snap_fisico = False

        self.graph.run()
        time.sleep(0.2)

        if self.callback_still is not None:
            try:
                self.callback_still.grab_frame()
            except Exception:
                pass

        self.connected = True
        self.device_name = nombre
        self.on_status(
            nombre,
            self.resolution_text,
            "SNAP físico listo" if self.snap_fisico else "Captura por software",
            True,
        )

        try:
            self.callback_video.grab_frame()
        except Exception:
            pass

    # ------------------------------------------------------------
    # Linux / Raspberry V4L2
    # ------------------------------------------------------------

    def _conectar_linux(self, indice: int, nombre: str) -> None:
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        self.cap = cv2.VideoCapture(indice, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"No fue posible abrir /dev/video{indice}.")
        # Misma resolución que en el PC (640 × 480): con ella se calibraron el color y la escala, y con ella
        # se tomaron las capturas de prueba. Otra resolución cambia el encuadre (p. ej. 16:9 recorta el campo).
        res = os.environ.get("DERMATEC_RES", "640x480").lower().split("x")
        try:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(res[0]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(res[1]))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.resolution_text = f"{ancho} × {alto}" if ancho and alto else "Resolución automática"
        print(f"Cámara Linux /dev/video{indice}: {self.resolution_text}")
        self.snap_fisico = False
        self.connected = True
        self.device_name = nombre
        self.stop_linux.clear()
        estado_snap = self._iniciar_snap_linux(indice)

        def worker():
            while not self.stop_linux.is_set() and self.cap is not None:
                ok, frame = self.cap.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                with self.lock:
                    self.latest_frame = frame.copy()
                self.on_frame(frame)

        self.thread_linux = threading.Thread(target=worker, daemon=True)
        self.thread_linux.start()
        self.on_status(nombre, self.resolution_text, estado_snap, True)

    # ------------------------------------------------------------
    # Botón SNAP físico en Linux: el driver uvcvideo lo publica como un teclado (/dev/input/eventN)
    # ------------------------------------------------------------

    @staticmethod
    def _buscar_eventos_camara(indice: int) -> list[tuple[str, str]]:
        try:
            usb = Path(f"/sys/class/video4linux/video{indice}/device").resolve()
            usb_dev = str(usb.parent)  # dispositivo USB (sin la interfaz :1.0)
        except Exception:
            usb_dev = ""
        try:
            texto = Path("/proc/bus/input/devices").read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        encontrados, otros = [], []
        for bloque in texto.strip().split("\n\n"):
            nombre = re.search(r'N: Name="([^"]*)"', bloque)
            sysfs = re.search(r"S: Sysfs=(\S+)", bloque)
            evento = re.search(r"Handlers=.*?\b(event\d+)\b", bloque)
            if not (nombre and evento):
                continue
            n = nombre.group(1)
            ruta_sys = "/sys" + (sysfs.group(1) if sysfs else "")
            if usb_dev and ruta_sys.startswith(usb_dev):
                encontrados.append((f"/dev/input/{evento.group(1)}", n))
            elif re.search(r"uvc|camera|microscop|webcam", n, re.I):
                otros.append((f"/dev/input/{evento.group(1)}", n))
        return encontrados or otros

    def _iniciar_snap_linux(self, indice: int) -> str:
        eventos = self._buscar_eventos_camara(indice)
        if not eventos:
            print("SNAP físico: la cámara no publica un botón en /dev/input.")
            return "SNAP físico no detectado (use CAPTURAR o tecla C)"
        abiertos = []
        for ruta, nombre in eventos:
            try:
                fd = os.open(ruta, os.O_RDONLY | os.O_NONBLOCK)
                abiertos.append((fd, ruta, nombre))
            except PermissionError:
                print(f"SNAP físico: sin permiso para {ruta}. Ejecute: sudo usermod -aG input $USER y reinicie.")
            except Exception as exc:
                print(f"SNAP físico: no se pudo abrir {ruta}: {exc!r}")
        if not abiertos:
            return "SNAP sin permiso (grupo input)"
        print("SNAP físico escuchando en:", ", ".join(f"{r} ({n})" for _, r, n in abiertos))

        import select
        import struct
        detener = threading.Event()          # uno por conexión, para no dejar escuchas duplicadas
        self._snap_stop = detener
        formato = "llHHi"
        tam = struct.calcsize(formato)

        def escuchar():
            ultimo = 0.0
            try:
                while not detener.is_set():
                    listos, _, _ = select.select([fd for fd, _, _ in abiertos], [], [], 0.3)
                    for fd in listos:
                        try:
                            datos = os.read(fd, tam * 32)
                        except BlockingIOError:
                            continue
                        for k in range(0, len(datos) - tam + 1, tam):
                            _, _, tipo, codigo, valor = struct.unpack(formato, datos[k:k + tam])
                            if tipo == 1 and valor == 1 and time.time() - ultimo > 0.6:  # EV_KEY, tecla presionada
                                ultimo = time.time()
                                print(f"SNAP físico (código {codigo})")
                                frame = self.obtener_ultimo_frame()
                                if frame is not None:
                                    self.on_snap(frame)
            finally:
                for fd, _, _ in abiertos:
                    try:
                        os.close(fd)
                    except Exception:
                        pass

        threading.Thread(target=escuchar, daemon=True).start()
        self.snap_fisico = True
        return "SNAP físico listo"

    # ------------------------------------------------------------
    # Operación
    # ------------------------------------------------------------

    def solicitar_frame_windows(self) -> None:
        if ES_WINDOWS and self.connected and self.callback_video is not None:
            try:
                self.callback_video.grab_frame()
            except Exception as exc:
                print("Advertencia solicitando frame:", repr(exc))

    def obtener_ultimo_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def desconectar(self) -> None:
        self.connected = False

        if self.graph is not None:
            try:
                self.graph.stop()
            except Exception:
                pass
        self.graph = None
        self.camera_filter = None
        self.sample_video = None
        self.callback_video = None
        self.sample_still = None
        self.callback_still = None

        self.stop_linux.set()
        if getattr(self, "_snap_stop", None) is not None:
            self._snap_stop.set()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.thread_linux = None

        with self.lock:
            self.latest_frame = None

        self.device_name = None
        self.snap_fisico = False
        self.resolution_text = "—"


# ================================================================
# 7. DIÁLOGOS AUXILIARES
# ================================================================


class DialogoTresOpciones(tk.Toplevel):
    def __init__(
        self,
        parent,
        titulo: str,
        mensaje: str,
        opcion_1: str,
        opcion_2: str,
        opcion_3: str = "Cancelar",
    ):
        super().__init__(parent)
        self.resultado: Optional[str] = None
        self.title(titulo)
        self.configure(bg=COLOR_BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        cuerpo = ttk.Frame(self, padding=22)
        cuerpo.pack(fill="both", expand=True)

        ttk.Label(
            cuerpo,
            text=titulo,
            style="DialogTitle.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        ttk.Label(
            cuerpo,
            text=mensaje,
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 20))

        botones = ttk.Frame(cuerpo)
        botones.pack(fill="x")

        ttk.Button(botones, text=opcion_1, style="Primary.TButton", command=lambda: self._salir("opcion_1")).pack(side="left", padx=(0, 8))
        ttk.Button(botones, text=opcion_2, command=lambda: self._salir("opcion_2")).pack(side="left", padx=8)
        ttk.Button(botones, text=opcion_3, command=lambda: self._salir("cancelar")).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", lambda: self._salir("cancelar"))
        self.bind("<Escape>", lambda e: self._salir("cancelar"))
        self.update_idletasks()
        self._centrar(parent)

    def _centrar(self, parent):
        self.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
        y = parent.winfo_rooty() + max(0, (parent.winfo_height() - self.winfo_height()) // 2)
        self.geometry(f"+{x}+{y}")

    def _salir(self, resultado: str):
        self.resultado = resultado
        self.grab_release()
        self.destroy()

    @classmethod
    def preguntar(cls, parent, titulo, mensaje, opcion_1, opcion_2, opcion_3="Cancelar") -> Optional[str]:
        dlg = cls(parent, titulo, mensaje, opcion_1, opcion_2, opcion_3)
        parent.wait_window(dlg)
        return dlg.resultado


class DialogoSeleccion(tk.Toplevel):
    def __init__(self, parent, titulo: str, items: list[tuple[str, Any]]):
        super().__init__(parent)
        self.resultado = None
        self.items = items
        self.title(titulo)
        self.geometry("560x420")
        self.minsize(460, 340)
        self.transient(parent)
        self.grab_set()

        cuerpo = ttk.Frame(self, padding=16)
        cuerpo.pack(fill="both", expand=True)
        ttk.Label(cuerpo, text=titulo, style="DialogTitle.TLabel").pack(anchor="w", pady=(0, 10))

        self.listbox = tk.Listbox(
            cuerpo,
            font=("Segoe UI", 11),
            activestyle="none",
            selectbackground=COLOR_TEAL,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
        )
        self.listbox.pack(fill="both", expand=True)
        for etiqueta, _ in items:
            self.listbox.insert("end", etiqueta)
        if items:
            self.listbox.selection_set(0)

        barra = ttk.Frame(cuerpo)
        barra.pack(fill="x", pady=(12, 0))
        ttk.Button(barra, text="Cancelar", command=self._cancelar).pack(side="right")
        ttk.Button(barra, text="Abrir", style="Primary.TButton", command=self._aceptar).pack(side="right", padx=(0, 8))

        self.listbox.bind("<Double-1>", lambda e: self._aceptar())
        self.protocol("WM_DELETE_WINDOW", self._cancelar)

    def _aceptar(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.resultado = self.items[sel[0]][1]
        self.grab_release()
        self.destroy()

    def _cancelar(self):
        self.resultado = None
        self.grab_release()
        self.destroy()

    @classmethod
    def elegir(cls, parent, titulo: str, items: list[tuple[str, Any]]):
        dlg = cls(parent, titulo, items)
        parent.wait_window(dlg)
        return dlg.resultado


# ================================================================
# 8. APLICACIÓN PRINCIPAL
# ================================================================


class DermatecApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DERMATEC — Entorno de captura")
        self.root.geometry("1440x860")
        self.root.minsize(1120, 700)
        self.root.configure(bg=COLOR_BG)

        self.gestor = GestorDatos(DATA_ROOT)
        self.estado = EstadoSesion()

        self.model = None
        self.j21_module = None
        self.processing = False
        self.app_activa = True

        self.analysis_window: Optional[tk.Toplevel] = None
        self.progress_window: Optional[tk.Toplevel] = None

        self.thumb_refs: list[ImageTk.PhotoImage] = []
        self.analysis_thumb_refs: list[ImageTk.PhotoImage] = []
        self.preview_ref = None

        self._configurar_estilos()
        self._crear_ui_captura()

        self.camera = GestorCamara(
            on_frame=self._on_camera_frame,
            on_snap=self._on_physical_snap,
            on_status=self._on_camera_status,
        )

        self.root.protocol("WM_DELETE_WINDOW", self.intentar_cerrar_aplicacion)
        self.root.bind("<Escape>", lambda e: self.intentar_cerrar_aplicacion())
        self.root.bind("<c>", lambda e: self.capturar_desde_interfaz())
        self.root.bind("<C>", lambda e: self.capturar_desde_interfaz())

        self.root.after(300, self._auto_conectar_preferida)
        self.root.after(33, self._ciclo_video)
        self.root.after(3000, self._vigilar_dispositivo)

    # ============================================================
    # ESTILOS
    # ============================================================

    def _configurar_estilos(self):
        estilo = ttk.Style(self.root)
        try:
            estilo.theme_use("clam")
        except Exception:
            pass

        estilo.configure("TFrame", background=COLOR_BG)
        estilo.configure("Card.TFrame", background=COLOR_CARD)
        estilo.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        estilo.configure("Card.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        estilo.configure("Muted.Card.TLabel", background=COLOR_CARD, foreground=COLOR_MUTED, font=("Segoe UI", 9))
        estilo.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 21, "bold"))
        estilo.configure("Subtitle.TLabel", background=COLOR_BG, foreground=COLOR_MUTED, font=("Segoe UI", 9))
        estilo.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 11, "bold"))
        estilo.configure("DialogTitle.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 15, "bold"))
        estilo.configure("Result.TLabel", background=COLOR_CARD, foreground=COLOR_TEXT, font=("Segoe UI", 28, "bold"))
        estilo.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(16, 10), foreground="white", background=COLOR_TEAL)
        estilo.map("Primary.TButton", background=[("active", COLOR_TEAL_DARK), ("disabled", "#A7C8C4")])
        estilo.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(14, 9))
        estilo.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 9), foreground="white", background=COLOR_RED)
        estilo.map("Danger.TButton", background=[("active", "#8E3333")])
        estilo.configure("Treeview", rowheight=30, font=("Segoe UI", 9), background="white", fieldbackground="white")
        estilo.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        estilo.configure("Horizontal.TProgressbar", troughcolor="#E6EEEE", background=COLOR_TEAL)

    # ============================================================
    # VENTANA 1 — ENTORNO DE CAPTURA
    # ============================================================

    def _crear_ui_captura(self):
        # Barra superior
        top = ttk.Frame(self.root, padding=(18, 12))
        top.pack(fill="x")

        izquierda = ttk.Frame(top)
        izquierda.pack(side="left", fill="x", expand=True)
        ttk.Label(izquierda, text="DERMATEC", style="Title.TLabel").pack(anchor="w")
        self.label_contexto = ttk.Label(
            izquierda,
            text="Paciente: No seleccionado   ·   Lesión: No seleccionada",
            style="Subtitle.TLabel",
        )
        self.label_contexto.pack(anchor="w", pady=(2, 0))

        derecha = ttk.Frame(top)
        derecha.pack(side="right")
        self.boton_menu = ttk.Button(derecha, text="MENÚ")
        self.boton_menu.pack(side="left", padx=(0, 8))
        self.boton_menu.configure(command=lambda: self.mostrar_menu(self.boton_menu))
        self.boton_dispositivo = ttk.Button(derecha, text="DISPOSITIVO", command=self.abrir_dispositivos)
        self.boton_dispositivo.pack(side="left")

        # Contenedor principal
        main = ttk.Frame(self.root, padding=(18, 4, 18, 8))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=8)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Video
        video_card = tk.Frame(main, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_card.grid_rowconfigure(0, weight=1)
        video_card.grid_columnconfigure(0, weight=1)

        self.label_video = tk.Label(
            video_card,
            bg=COLOR_DARK_PANEL,
            fg="#DCE7EA",
            text="Esperando dispositivo de video...",
            font=("Segoe UI", 12),
        )
        self.label_video.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Panel de miniaturas
        thumbs_card = tk.Frame(main, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
        thumbs_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        thumbs_card.grid_columnconfigure(0, weight=1)
        thumbs_card.grid_rowconfigure(1, weight=1)

        header_thumbs = tk.Frame(thumbs_card, bg=COLOR_CARD)
        header_thumbs.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        tk.Label(header_thumbs, text="IMÁGENES CAPTURADAS", bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        self.label_num_imagenes = tk.Label(header_thumbs, text="0", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9))
        self.label_num_imagenes.pack(side="right")

        self.canvas_thumbs = tk.Canvas(thumbs_card, bg=COLOR_CARD, highlightthickness=0)
        scroll = ttk.Scrollbar(thumbs_card, orient="vertical", command=self.canvas_thumbs.yview)
        self.frame_thumbs = tk.Frame(self.canvas_thumbs, bg=COLOR_CARD)
        self.window_thumbs = self.canvas_thumbs.create_window((0, 0), window=self.frame_thumbs, anchor="nw")
        self.canvas_thumbs.configure(yscrollcommand=scroll.set)
        self.canvas_thumbs.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        scroll.grid(row=1, column=1, sticky="ns", pady=(0, 8))
        self.frame_thumbs.bind("<Configure>", lambda e: self.canvas_thumbs.configure(scrollregion=self.canvas_thumbs.bbox("all")))
        self.canvas_thumbs.bind("<Configure>", lambda e: self.canvas_thumbs.itemconfigure(self.window_thumbs, width=e.width))

        # Estado cámara
        estado = tk.Frame(self.root, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
        estado.pack(fill="x", padx=18, pady=(0, 8))
        self.label_live = tk.Label(estado, text="○ LIVE", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9, "bold"))
        self.label_live.pack(side="left", padx=(14, 18), pady=9)
        self.label_device = tk.Label(estado, text="Dispositivo: no conectado", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9))
        self.label_device.pack(side="left", padx=(0, 18))
        self.label_resolution = tk.Label(estado, text="Resolución: —", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9))
        self.label_resolution.pack(side="left", padx=(0, 18))
        self.label_snap = tk.Label(estado, text="SNAP: —", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9))
        self.label_snap.pack(side="left")

        # Acciones
        acciones = ttk.Frame(self.root, padding=(18, 2, 18, 14))
        acciones.pack(fill="x")
        centro = ttk.Frame(acciones)
        centro.pack()

        self.btn_capturar = ttk.Button(centro, text="CAPTURAR", style="Secondary.TButton", command=self.capturar_desde_interfaz)
        self.btn_capturar.grid(row=0, column=0, padx=7)
        self.btn_nueva_lesion = ttk.Button(centro, text="NUEVA LESIÓN", style="Secondary.TButton", command=self.nueva_lesion)
        self.btn_nueva_lesion.grid(row=0, column=1, padx=7)
        self.btn_analizar = ttk.Button(centro, text="ANALIZAR LESIÓN", style="Primary.TButton", command=self.analizar_lesion)
        self.btn_analizar.grid(row=0, column=2, padx=7)

        self._actualizar_estado_botones()

    # ============================================================
    # MENÚ SUPERIOR
    # ============================================================

    def _parent_actual(self):
        if self.analysis_window is not None:
            try:
                if self.analysis_window.winfo_exists():
                    return self.analysis_window
            except Exception:
                pass
        return self.root

    def mostrar_menu(self, anchor_widget=None):
        parent = self._parent_actual()
        menu = tk.Menu(parent, tearoff=False, font=("Segoe UI", 10))
        menu.add_command(label="Nuevo paciente", command=self.nuevo_paciente)
        menu.add_command(label="Abrir paciente", command=self.abrir_paciente)
        menu.add_separator()
        menu.add_command(label="Nueva lesión", command=self.nueva_lesion)
        menu.add_command(label="Abrir lesión", command=self.abrir_lesion)
        menu.add_separator()
        # Una sola entrada de menú permite seleccionar N imágenes de una vez.
        # Se conserva el flujo de importación existente: cada archivo se
        # copia a la lesión activa, se evita duplicar archivos y después
        # se refresca la galería una sola vez.
        menu.add_command(
            label="Cargar imágenes",
            command=lambda: self.cargar_imagenes(True),
        )
        menu.add_separator()
        menu.add_command(label="Guardar análisis", command=self.guardar_analisis_actual)
        menu.add_command(label="Exportar informe del paciente", command=self.exportar_informe)
        menu.add_command(label="Imprimir informe", command=self.imprimir_informe)
        menu.add_command(label="Abrir carpeta del paciente", command=self.abrir_carpeta_paciente)
        menu.add_separator()
        menu.add_command(label="Calibrar escala (mm)", command=self.calibrar_escala)
        menu.add_command(label="Calibrar color", command=self.calibrar_color)
        menu.add_separator()
        menu.add_command(label="Salir", command=self.intentar_cerrar_aplicacion)

        anchor = anchor_widget or self.boton_menu
        x = anchor.winfo_rootx()
        y = anchor.winfo_rooty() + anchor.winfo_height()
        menu.tk_popup(x, y)

    def calibrar_color(self):
        parent = self._parent_actual()
        blanco = filedialog.askopenfilename(parent=parent, title="1/2 — Foto del cuadro BLANCO (G1) con el acople final",
                                            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if not blanco:
            return
        grises = filedialog.askopenfilenames(parent=parent, title="2/2 — Fotos de grises medios (G5, G6, G7) — opcional",
                                             filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        acople = escala_acople.leer().get("acople_activo", "")
        try:
            cfg = calibracion_color.construir(blanco, list(grises), acople=acople)
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No se pudo calibrar el color:\n{exc}", parent=parent)
            return
        g = cfg["ganancias_bgr"]
        messagebox.showinfo(
            "DERMATEC",
            f"Calibración de color guardada (acople {acople}).\n\n"
            f"Iluminación esquinas/centro: {cfg['iluminacion_esquinas_vs_centro']:.2f} (se compensa)\n"
            f"Ganancias R/G/B: {g[2]:.3f} / {g[1]:.3f} / {g[0]:.3f}\n\n"
            "Se aplicará a todas las capturas en los próximos análisis.",
            parent=parent,
        )

    def calibrar_escala(self):
        escala_acople.abrir_calibrador(
            self._parent_actual(),
            al_guardar=lambda: messagebox.showinfo(
                "DERMATEC", escala_acople.texto_escala(escala_acople.escala_para(640)) +
                "\n\nVuelva a analizar la lesión para recalcular los parámetros.", parent=self._parent_actual()),
        )

    # ============================================================
    # PACIENTES / LESIONES
    # ============================================================

    @staticmethod
    def _maximizar_ventana(win):
        """Maximiza una ventana Tk/Toplevel de forma segura en Windows."""
        try:
            win.update_idletasks()
            win.state("zoomed")
            win.update_idletasks()
        except Exception:
            try:
                win.attributes("-zoomed", True)
            except Exception:
                pass

    def _nombre_por_defecto_paciente(self) -> str:
        return f"Paciente {len(self.gestor.listar_pacientes()) + 1}"

    def _nombre_por_defecto_lesion(self) -> str:
        if not self.estado.paciente:
            return "Lesión 1"
        return f"Lesión {len(self.gestor.listar_lesiones(self.estado.paciente)) + 1}"

    def nuevo_paciente(self):
        if not self._permitir_cambio_contexto():
            return
        nombre = simpledialog.askstring(
            "DERMATEC — Nuevo paciente",
            "Nombre o identificación visible del paciente:",
            initialvalue=self._nombre_por_defecto_paciente(),
            parent=self._parent_actual(),
        )
        if nombre is None:
            return
        try:
            paciente = self.gestor.crear_paciente(nombre)
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No fue posible crear el paciente.\n\n{exc}", parent=self.root)
            return

        self.estado = EstadoSesion(paciente=paciente)
        self._actualizar_contexto()
        messagebox.showinfo(
            "DERMATEC",
            f"Paciente creado: {paciente.nombre}\n\nAhora cree la primera lesión para comenzar a capturar imágenes.",
            parent=self.root,
        )
        self.nueva_lesion(forzar=True)

    def abrir_paciente(self):
        if not self._permitir_cambio_contexto():
            return
        pacientes = self.gestor.listar_pacientes()
        if not pacientes:
            messagebox.showinfo("DERMATEC", "Todavía no existen pacientes guardados.", parent=self.root)
            return
        items = [(f"{p.id}   {p.nombre}", p) for p in pacientes]
        paciente = DialogoSeleccion.elegir(self._parent_actual(), "Abrir paciente", items)
        if paciente is None:
            return
        self.estado = EstadoSesion(paciente=paciente)
        self._actualizar_contexto()
        lesiones = self.gestor.listar_lesiones(paciente)
        if lesiones:
            self.abrir_lesion(forzar=True)
        else:
            self.nueva_lesion(forzar=True)

    def nueva_lesion(self, forzar: bool = False):
        if not self.estado.paciente:
            messagebox.showwarning("DERMATEC", "Primero cree o abra un paciente.", parent=self.root)
            return
        if not forzar and not self._permitir_cambio_contexto():
            return
        nombre = simpledialog.askstring(
            "DERMATEC — Nueva lesión",
            "Nombre visible de la lesión:",
            initialvalue=self._nombre_por_defecto_lesion(),
            parent=self._parent_actual(),
        )
        if nombre is None:
            return
        try:
            lesion = self.gestor.crear_lesion(self.estado.paciente, nombre)
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No fue posible crear la lesión.\n\n{exc}", parent=self.root)
            return
        paciente = self.estado.paciente
        self.estado = EstadoSesion(paciente=paciente, lesion=lesion)
        self._cargar_lesion_en_ui()

    def abrir_lesion(self, forzar: bool = False):
        if not self.estado.paciente:
            messagebox.showwarning("DERMATEC", "Primero cree o abra un paciente.", parent=self.root)
            return
        if not forzar and not self._permitir_cambio_contexto():
            return
        lesiones = self.gestor.listar_lesiones(self.estado.paciente)
        if not lesiones:
            messagebox.showinfo("DERMATEC", "Este paciente todavía no tiene lesiones guardadas.", parent=self.root)
            return
        items = [(f"{l.id}   {l.nombre}", l) for l in lesiones]
        lesion = DialogoSeleccion.elegir(self._parent_actual(), "Abrir lesión", items)
        if lesion is None:
            return
        paciente = self.estado.paciente
        self.estado = EstadoSesion(paciente=paciente, lesion=lesion)
        self._cargar_lesion_en_ui()

    def _cargar_lesion_en_ui(self):
        lesion = self.estado.lesion
        if not lesion:
            return
        self.estado.imagenes = self.gestor.listar_imagenes(lesion)
        guardado = self.gestor.cargar_analisis(lesion)
        if guardado:
            self.estado.resultados_individuales = guardado.get("resultados_individuales", []) or []
            self.estado.resultado_general = guardado.get("resultado_general")
            self.estado.observaciones = guardado.get("observaciones", self.estado.observaciones) or self.estado.observaciones
            self.estado.firma_imagenes_analizadas = guardado.get("firma_imagenes", []) or []
            self.estado.analisis_generado = bool(self.estado.resultado_general)
            self.estado.analisis_guardado = True
            self.estado.analisis_sucio = False
            firmas_actuales = self._firmas_imagenes_actuales()
            self.estado.analisis_desactualizado = firmas_actuales != self.estado.firma_imagenes_analizadas
        self._actualizar_contexto()
        self._refrescar_miniaturas()
        self._actualizar_estado_botones()

        # Si el cambio de paciente/lesión se inició desde la ventana de análisis,
        # volvemos al entorno de captura para no dejar una vista antigua abierta.
        if self.analysis_window is not None:
            try:
                if self.analysis_window.winfo_exists():
                    self._cerrar_ventana_analisis_y_mostrar_captura()
            except Exception:
                pass

    def _permitir_cambio_contexto(self) -> bool:
        if self.processing:
            messagebox.showwarning("DERMATEC", "Hay un análisis en curso. Espere a que termine.", parent=self.root)
            return False

        if self.estado.analisis_sucio:
            r = DialogoTresOpciones.preguntar(
                self._parent_actual(),
                "Análisis sin guardar",
                "La lesión actual contiene un análisis u observaciones que todavía no han sido guardados.\n\n¿Desea guardar antes de cambiar de paciente o lesión?",
                "Guardar y continuar",
                "Continuar sin guardar",
            )
            if r == "cancelar" or r is None:
                return False
            if r == "opcion_1":
                if not self.guardar_analisis_actual(silencioso=False):
                    return False
            elif r == "opcion_2":
                self._descartar_cambios_analisis()

        elif self._hay_imagenes_sin_analisis_actualizado():
            continuar = messagebox.askyesno(
                "DERMATEC — Análisis pendiente",
                "La lesión actual contiene imágenes nuevas o todavía no tiene un análisis actualizado.\n\n"
                "Las imágenes ya están guardadas en disco y NO se perderán. Sin embargo, cambiará de contexto sin generar el análisis correspondiente.\n\n"
                "¿Desea continuar?",
                parent=self.root,
            )
            if not continuar:
                return False
        return True

    # ============================================================
    # IMÁGENES
    # ============================================================

    def _requerir_lesion(self) -> bool:
        if not self.estado.paciente:
            messagebox.showwarning(
                "DERMATEC",
                "Aún no hay un paciente seleccionado.\n\nCree un paciente nuevo o abra un paciente existente antes de capturar imágenes.",
                parent=self.root,
            )
            return False

        if not self.estado.lesion:
            messagebox.showwarning(
                "DERMATEC",
                "El paciente está seleccionado, pero todavía no hay una lesión activa.\n\nCree o seleccione una lesión antes de capturar imágenes.",
                parent=self.root,
            )
            return False

        return True

    def cargar_imagenes(self, multiples: bool = True):
        if not self._requerir_lesion():
            return
        opciones = {
            "title": "Cargar imágenes dermatoscópicas",
            "filetypes": [
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        }
        if multiples:
            seleccion = filedialog.askopenfilenames(parent=self.root, **opciones)
            rutas = [Path(x) for x in seleccion]
        else:
            seleccion = filedialog.askopenfilename(parent=self.root, **opciones)
            rutas = [Path(seleccion)] if seleccion else []
        if not rutas:
            return

        agregadas = 0
        duplicadas = 0
        for ruta in rutas:
            try:
                destino, duplicada = self.gestor.importar_imagen(self.estado.lesion, ruta)
                if duplicada:
                    duplicadas += 1
                elif destino:
                    agregadas += 1
            except Exception as exc:
                messagebox.showerror("DERMATEC", f"No fue posible importar:\n{ruta}\n\n{exc}", parent=self.root)

        if agregadas:
            self._desactualizar_analisis_por_cambio_imagenes()
            self.estado.imagenes = self.gestor.listar_imagenes(self.estado.lesion)
            self._refrescar_miniaturas()
        if duplicadas:
            messagebox.showinfo(
                "DERMATEC",
                f"Se agregaron {agregadas} imágenes.\n{duplicadas} archivo(s) eran duplicados exactos y no se copiaron nuevamente.",
                parent=self.root,
            )

    def _guardar_captura(self, frame: np.ndarray, fuente: str):
        if not self._requerir_lesion():
            return
        try:
            destino, duplicada = self.gestor.guardar_frame(self.estado.lesion, frame, fuente)
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No fue posible guardar la captura.\n\n{exc}", parent=self.root)
            return

        if duplicada:
            self.root.after(0, lambda: messagebox.showinfo("DERMATEC", "La imagen capturada es idéntica a una ya guardada en esta lesión y no se duplicó.", parent=self.root))
            return

        if destino:
            self._desactualizar_analisis_por_cambio_imagenes()
            self.estado.imagenes = self.gestor.listar_imagenes(self.estado.lesion)
            self.root.after(0, self._refrescar_miniaturas)
            print(f"Captura guardada: {destino.name} ({fuente})")

    def capturar_desde_interfaz(self):
        if not self._requerir_lesion():
            return
        frame = self.camera.obtener_ultimo_frame()
        if frame is None:
            messagebox.showwarning("DERMATEC", "Todavía no hay un frame LIVE disponible.", parent=self.root)
            return
        self._guardar_captura(frame, "CAPTURA_SOFTWARE")

    def _on_physical_snap(self, frame: np.ndarray):
        if not self.estado.paciente:
            print("SNAP ignorado: no hay paciente seleccionado.")
            self.root.after(0, lambda: messagebox.showwarning(
                "DERMATEC",
                "Aún no hay un paciente seleccionado.\n\nCree un paciente nuevo o abra un paciente existente antes de capturar imágenes.",
                parent=self.root,
            ))
            return

        if not self.estado.lesion:
            print("SNAP ignorado: no hay lesión activa.")
            self.root.after(0, lambda: messagebox.showwarning(
                "DERMATEC",
                "El paciente está seleccionado, pero no hay una lesión activa.\n\nCree o seleccione una lesión antes de capturar imágenes.",
                parent=self.root,
            ))
            return
        frame_copia = frame.copy()
        try:
            self.root.after(0, lambda f=frame_copia: self._guardar_captura(f, "SNAP_FISICO"))
        except Exception:
            pass

    def _refrescar_miniaturas(self):
        for widget in self.frame_thumbs.winfo_children():
            widget.destroy()
        self.thumb_refs.clear()

        imagenes = self.estado.imagenes
        self.label_num_imagenes.configure(text=f"{len(imagenes)} imagen(es)")

        if not imagenes:
            tk.Label(
                self.frame_thumbs,
                text="Aún no hay imágenes\nde esta lesión.",
                bg=COLOR_CARD,
                fg=COLOR_MUTED,
                font=("Segoe UI", 9),
                justify="center",
            ).pack(pady=40)
            self._actualizar_estado_botones()
            return

        for idx, ruta in enumerate(imagenes, start=1):
            tarjeta = tk.Frame(self.frame_thumbs, bg="#F8FBFB", highlightbackground=COLOR_BORDER, highlightthickness=1)
            tarjeta.pack(fill="x", padx=4, pady=5)

            try:
                img = Image.open(ruta).convert("RGB")
                img.thumbnail((180, 120), Image.Resampling.LANCZOS)
                ph = ImageTk.PhotoImage(img)
                self.thumb_refs.append(ph)
                lbl = tk.Label(tarjeta, image=ph, bg="#F8FBFB", cursor="hand2")
                lbl.pack(pady=(6, 2))
                lbl.bind("<Button-1>", lambda e, p=ruta: self._abrir_detalle_imagen_captura(p))
            except Exception:
                tk.Label(tarjeta, text="Vista no disponible", bg="#F8FBFB", fg=COLOR_MUTED).pack(pady=20)

            tk.Label(tarjeta, text=f"Imagen {idx}", bg="#F8FBFB", fg=COLOR_TEXT, font=("Segoe UI", 9, "bold")).pack()
            fuente = self.gestor.fuente_imagen(self.estado.lesion, ruta.name) if self.estado.lesion else ""
            tk.Label(tarjeta, text=fuente.replace("_", " "), bg="#F8FBFB", fg=COLOR_MUTED, font=("Segoe UI", 8)).pack(pady=(0, 6))

        self._actualizar_estado_botones()

    def _abrir_detalle_imagen_captura(self, ruta: Path):
        win = tk.Toplevel(self.root)
        win.title(f"DERMATEC — {ruta.name}")
        win.geometry("900x700")
        win.transient(self.root)
        cont = ttk.Frame(win, padding=12)
        cont.pack(fill="both", expand=True)

        try:
            img = Image.open(ruta).convert("RGB")
            ancho0 = img.width
            img.thumbnail((830, 560), Image.Resampling.LANCZOS)
            dibujar_regla_mm(img, ancho0)
            ph = ImageTk.PhotoImage(img)
            lbl = ttk.Label(cont, image=ph)
            lbl.image = ph
            lbl.pack(fill="both", expand=True)
        except Exception as exc:
            ttk.Label(cont, text=f"No fue posible cargar la imagen.\n{exc}").pack(expand=True)

        barra = ttk.Frame(cont)
        barra.pack(fill="x", pady=(10, 0))
        ttk.Button(barra, text="Cerrar", command=win.destroy).pack(side="right")
        ttk.Button(barra, text="Eliminar captura", style="Danger.TButton", command=lambda: self._eliminar_imagen_desde_modal(win, ruta)).pack(side="left")

    def _eliminar_imagen_desde_modal(self, win: tk.Toplevel, ruta: Path):
        if not messagebox.askyesno(
            "DERMATEC — Eliminar imagen",
            "¿Desea eliminar esta imagen de la lesión?\n\nEsta acción elimina el archivo guardado.",
            parent=win,
        ):
            return
        try:
            self.gestor.eliminar_imagen(self.estado.lesion, ruta)
            self._desactualizar_analisis_por_cambio_imagenes()
            self.estado.imagenes = self.gestor.listar_imagenes(self.estado.lesion)
            win.destroy()
            self._refrescar_miniaturas()
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No fue posible eliminar la imagen.\n\n{exc}", parent=win)

    def _desactualizar_analisis_por_cambio_imagenes(self):
        if self.estado.analisis_generado or self.estado.analisis_guardado:
            self.estado.analisis_desactualizado = True
        self.estado.analisis_generado = False
        self.estado.resultados_individuales = []
        self.estado.resultado_general = None
        self.estado.analisis_sucio = False
        self.estado.firma_imagenes_analizadas = []

    # ============================================================
    # CÁMARA / DISPOSITIVOS
    # ============================================================

    def _on_camera_frame(self, frame: np.ndarray):
        # El frame ya queda guardado dentro de GestorCamara.
        pass

    def _on_camera_status(self, nombre: str, resolucion: str, snap: str, connected: bool):
        def actualizar():
            if connected:
                self.label_live.configure(text="● LIVE", fg=COLOR_GREEN)
                self.label_device.configure(text=f"Dispositivo: {nombre}", fg=COLOR_TEXT)
                self.label_resolution.configure(text=f"Resolución: {resolucion}", fg=COLOR_TEXT)
                self.label_snap.configure(text=snap, fg=COLOR_GREEN if "listo" in snap.lower() else COLOR_MUTED)
            else:
                self.label_live.configure(text="○ LIVE", fg=COLOR_MUTED)
                self.label_device.configure(text="Dispositivo: no conectado", fg=COLOR_MUTED)
                self.label_resolution.configure(text="Resolución: —", fg=COLOR_MUTED)
                self.label_snap.configure(text="SNAP: —", fg=COLOR_MUTED)
            self._actualizar_estado_botones()
        try:
            self.root.after(0, actualizar)
        except Exception:
            pass

    def _auto_conectar_preferida(self):
        if not self.app_activa:
            return
        dispositivos = self.camera.listar_dispositivos()
        if not dispositivos:
            self._on_camera_status("desconectada", "—", "SNAP no disponible", False)
            return

        elegido = None
        for nombre, identificador in dispositivos:
            if nombre.strip().casefold() == "usb 2.0 camera":
                elegido = (nombre, identificador)
                break
        if elegido:
            nombre, identificador = elegido
            self.camera.conectar(identificador, nombre, silencioso=True)
        else:
            self.label_device.configure(text=f"{len(dispositivos)} cámara(s) detectada(s). Use DISPOSITIVO para conectar.")

    def _vigilar_dispositivo(self):
        if not self.app_activa:
            return
        try:
            dispositivos = self.camera.listar_dispositivos()
            nombres = [x[0] for x in dispositivos]
            if self.camera.connected and self.camera.device_name not in nombres:
                deseado = self.camera.device_name
                print("Dispositivo desconectado físicamente:", deseado)
                self.camera.desired_device_name = deseado
                self.camera.desconectar()
                self._on_camera_status("desconectada", "—", "SNAP no disponible", False)
            elif (not self.camera.connected) and self.camera.desired_device_name:
                for nombre, identificador in dispositivos:
                    if nombre == self.camera.desired_device_name:
                        print("Dispositivo volvió a aparecer; intentando reconexión...")
                        self.camera.conectar(identificador, nombre, silencioso=True)
                        break
        except Exception as exc:
            print("Monitor de dispositivos:", repr(exc))
        self.root.after(3000, self._vigilar_dispositivo)

    def abrir_dispositivos(self):
        parent = self._parent_actual()
        win = tk.Toplevel(parent)
        win.title("DERMATEC — Dispositivos")
        win.geometry("620x500")
        win.transient(parent)

        cont = ttk.Frame(win, padding=18)
        cont.pack(fill="both", expand=True)
        ttk.Label(cont, text="Dispositivos de imagen", style="DialogTitle.TLabel").pack(anchor="w")
        ttk.Label(
            cont,
            text=(
                "Seleccione el dermatoscopio/cámara que desea utilizar. "
                "En Raspberry Pi el backend de video cambiará a V4L2/libcamera sin modificar la interfaz."
            ),
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(5, 12))

        lista = tk.Listbox(
            cont,
            font=("Segoe UI", 10),
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            selectbackground=COLOR_TEAL,
            selectforeground="white",
        )
        lista.pack(fill="both", expand=True)
        cache: list[tuple[str, Any]] = []

        estado = ttk.Label(cont, text="")
        estado.pack(anchor="w", pady=(8, 0))

        def refrescar():
            nonlocal cache
            cache = self.camera.listar_dispositivos()
            lista.delete(0, "end")
            for nombre, identificador in cache:
                texto = nombre
                if self.camera.connected and nombre == self.camera.device_name:
                    texto += "   ✓ CONECTADO"
                lista.insert("end", texto)
            estado.configure(text=f"{len(cache)} dispositivo(s) detectado(s).")
            if cache:
                lista.selection_set(0)

        def conectar():
            sel = lista.curselection()
            if not sel:
                messagebox.showwarning("DERMATEC", "Seleccione un dispositivo.", parent=win)
                return
            nombre, identificador = cache[sel[0]]
            if self.camera.conectar(identificador, nombre, silencioso=False):
                refrescar()

        def desconectar():
            self.camera.desired_device_name = None
            self.camera.desconectar()
            self._on_camera_status("desconectada", "—", "SNAP no disponible", False)
            refrescar()

        barra = ttk.Frame(cont)
        barra.pack(fill="x", pady=(12, 0))
        ttk.Button(barra, text="Actualizar", command=refrescar).pack(side="left")
        ttk.Button(barra, text="Desconectar", command=desconectar).pack(side="right")
        ttk.Button(barra, text="Conectar", style="Primary.TButton", command=conectar).pack(side="right", padx=(0, 8))
        refrescar()

    def _ciclo_video(self):
        if not self.app_activa:
            return
        if ES_WINDOWS:
            self.camera.solicitar_frame_windows()

        frame = self.camera.obtener_ultimo_frame()
        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                ancho = max(self.label_video.winfo_width() - 12, 320)
                alto = max(self.label_video.winfo_height() - 12, 240)
                img.thumbnail((ancho, alto), Image.Resampling.BILINEAR)
                dibujar_regla_mm(img, rgb.shape[1])
                ph = ImageTk.PhotoImage(img)
                self.preview_ref = ph
                self.label_video.configure(image=ph, text="")
            except Exception as exc:
                print("Error mostrando video:", repr(exc))
        else:
            if not self.camera.connected:
                self.label_video.configure(image="", text="Conecte un dispositivo desde DISPOSITIVO")
        self.root.after(33, self._ciclo_video)

    # ============================================================
    # V5.2 — GUARDAR ARTEFACTOS DE RAMA 1
    # ============================================================

    def _guardar_rama1(self, ruta_imagen: Path, rama1: dict[str, Any]) -> dict[str, Any]:
        base = self.estado.lesion.carpeta / "analisis_rama1_v5_2" / ruta_imagen.stem
        base.mkdir(parents=True, exist_ok=True)

        def save(name, arr):
            p = base / name
            if arr is None:
                return None
            guardar_imagen_segura(p, arr)
            return str(p.relative_to(self.estado.lesion.carpeta))

        p_limpia = save("B2_23_imagen_limpia.png", rama1["imagen_limpia_b223"])
        p_mask_b223 = save("B2_23_mascara.png", rama1["mascara_b223"])
        p_score = save("B2_23_score.png", np.uint8(np.clip(rama1["score_b223"] * 255.0, 0, 255)))
        p_gmm = save("GMM_LAB_POST_B38_mascara.png", rama1["mascara_gmm"])
        p_final = save("MASK_FINAL_DILATE_5.png", rama1["mascara_final"])
        p_roi = save("ROI_SEGMENTADA_ORIGINAL.png", rama1["roi"])
        p_roi_mask = save("ROI_MASK.png", rama1["roi_mask"])

        overlay = leer_imagen_bgr_segura(ruta_imagen)
        p_contour = None
        if overlay is not None:
            contours, _ = cv2.findContours(
                rama1["mascara_final"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(overlay, contours, -1, (38, 180, 156), 3)
            p_contour = save("CONTORNO_MASK_FINAL.png", overlay)

        return {
            "carpeta": str(base.relative_to(self.estado.lesion.carpeta)),
            "imagen_limpia_b223": p_limpia,
            "mascara_b223": p_mask_b223,
            "score_b223": p_score,
            "mascara_gmm": p_gmm,
            "mascara_final": p_final,
            "roi": p_roi,
            "roi_mask": p_roi_mask,
            "contorno": p_contour,
            "valida": bool(rama1["valida"]),
            "calidad": float(rama1["calidad"]),
            "parametros": rama1["parametros"],
        }

    # ============================================================
    # INFERENCIA MULTIIMAGEN
    # ============================================================

    def _cargar_modelo(self):
        if self.model is not None and self.j21_module is not None:
            return

        faltantes = [p for p in ARCHIVOS_Z23 if not p.is_file()]
        if not faltantes:
            try:
                cfg = json.loads((Z23_ROOT / "config_dermatec.json").read_text(encoding="utf-8"))
                faltantes = [Z23_ROOT / m["archivo"] for m in cfg.get("modelos", []) if not (Z23_ROOT / m["archivo"]).is_file()]
            except Exception as exc:
                raise FileNotFoundError(f"config_dermatec.json no se pudo leer: {exc!r}")
        if faltantes:
            raise FileNotFoundError(
                "Faltan archivos del modelo en MODELO_DERMATEC_FINAL:\n" + "\n".join(str(p) for p in faltantes) +
                "\n\nCopie ahí el contenido de PAQUETE_FINAL_DERMATEC (config_dermatec.json y los .tflite)."
            )

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "dermatec_inferencia", str(Z23_ROOT / "dermatec_inferencia.py")
        )
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        self.j21_module = modulo
        self.model = modulo.DermatecFinal(Z23_ROOT)

    def analizar_lesion(self):
        if not self._requerir_lesion():
            return
        self.estado.imagenes = self.gestor.listar_imagenes(self.estado.lesion)
        if not self.estado.imagenes:
            messagebox.showwarning(
                "DERMATEC",
                "La lesión todavía no contiene imágenes para analizar.",
                parent=self.root,
            )
            return
        if self.processing:
            return

        if self.estado.analisis_sucio:
            continuar = messagebox.askyesno(
                "DERMATEC — Reanalizar",
                "Existe un análisis actual con cambios sin guardar.\n\n"
                "Si continúa se generará un nuevo análisis a partir de las imágenes actuales. ¿Desea continuar?",
                parent=self.root,
            )
            if not continuar:
                return

        self.processing = True
        self._actualizar_estado_botones()
        self._abrir_progreso(len(self.estado.imagenes))

        imagenes = list(self.estado.imagenes)

        def worker():
            try:
                self._actualizar_progreso_threadsafe(
                    0, len(imagenes), "Cargando clasificador DERMATEC..."
                )
                self._cargar_modelo()

                resultados = []

                for i, ruta in enumerate(imagenes, start=1):
                    self._actualizar_progreso_threadsafe(
                        i - 1,
                        len(imagenes),
                        f"Analizando imagen {i} de {len(imagenes)} — Rama 1 + clasificador...",
                    )

                    bgr = leer_imagen_bgr_segura(ruta)
                    # Calibración de color del equipo (misma corrección fija para todas las capturas)
                    bgr, info_color = calibracion_color.aplicar(bgr)

                    # --------------------------------------------------
                    # RAMA 1 (v2): vello -> piel válida -> sombreado -> Otsu Lab
                    #              -> selección -> GrabCut -> DILATE_5 -> ROI
                    # --------------------------------------------------
                    rama1 = ejecutar_rama1(bgr)
                    escala = escala_acople.escala_para(bgr.shape[1])
                    try:
                        parametros = parametros_lesion.calcular(bgr, rama1["mascara_final"], escala["px_por_mm"])
                    except Exception as exc:
                        print("Parámetros no disponibles:", repr(exc))
                        parametros = {}
                    rama1_meta = self._guardar_rama1(ruta, rama1)

                    # --------------------------------------------------
                    # RAMA 2
                    # El clasificador recibe la captura (con la calibración de color del equipo, si existe);
                    # nunca la imagen modificada por la Rama 1.
                    # --------------------------------------------------
                    ruta_j21 = ruta
                    temporal_j21 = None
                    try:
                        # Si la ruta contiene Unicode, el clasificador recibe una copia
                        # temporal con ruta ASCII. La imagen y sus bytes son
                        # exactamente los de la original.
                        if info_color.get("aplicada"):
                            # La red recibe la imagen con la calibración de color aplicada (copia temporal PNG, sin pérdida)
                            temporal_j21 = Path(tempfile.gettempdir()) / f"DERMATEC_CAL_{uuid.uuid4().hex}.png"
                            ok, buf = cv2.imencode(".png", bgr)
                            temporal_j21.write_bytes(buf.tobytes())
                            ruta_j21 = temporal_j21
                        else:
                            # Si la ruta contiene Unicode, el clasificador recibe una copia temporal con ruta ASCII
                            # (mismos bytes que la original).
                            try:
                                str(ruta).encode("ascii")
                            except UnicodeEncodeError:
                                temporal_j21 = Path(tempfile.gettempdir()) / f"DERMATEC_J21_{uuid.uuid4().hex}.jpg"
                                temporal_j21.write_bytes(np.fromfile(str(ruta), dtype=np.uint8).tobytes())
                                ruta_j21 = temporal_j21

                        # La máscara validada de la Rama 1 se usa para el control de encuadre.
                        salida = self.model.predict(str(ruta_j21), mascara=rama1.get("mascara_final"))
                        salida = dict(salida) if isinstance(salida, dict) else {
                            "salida_modelo": salida
                        }
                    finally:
                        if temporal_j21 is not None:
                            try:
                                temporal_j21.unlink(missing_ok=True)
                            except Exception:
                                pass
                    salida["rama1"] = rama1_meta
                    salida["parametros_lesion"] = parametros
                    salida["escala"] = escala
                    salida["calibracion_color"] = info_color
                    salida.pop("parametros_dermatoscopicos", None)   # reemplazados por parametros_lesion

                    resultados.append({
                        "imagen": ruta.name,
                        "fuente": self.gestor.fuente_imagen(
                            self.estado.lesion, ruta.name
                        ),
                        "resultado": salida,
                    })

                    self._actualizar_progreso_threadsafe(
                        i,
                        len(imagenes),
                        f"Imagen {i} de {len(imagenes)} completada",
                    )

                # Agregación sobre las N predicciones del ensamble.
                # No se usa mayoría simple.
                general = self._agregar_resultados_multiimagen(resultados)

                firmas = [sha256_file(p) for p in imagenes]

                validas = sum(
                    1
                    for x in resultados
                    if bool((x["resultado"].get("rama1", {}) or {}).get("valida"))
                )

                general["rama1"] = {
                    "imagenes_procesadas": len(resultados),
                    "segmentaciones_validas": validas,
                    "porcentaje_validas": float(100.0 * validas / len(resultados)),
                    "pipeline": (
                        "ORIGINAL -> B2.23 -> GMM_LAB_POST_B38 -> "
                        "DILATE_5 -> ROI"
                    ),
                    "clasificador_input": "ORIGINAL",
                }

                def terminar():
                    self.processing = False
                    self._cerrar_progreso()
                    self.estado.resultados_individuales = resultados
                    self.estado.resultado_general = general
                    self.estado.analisis_generado = True
                    self.estado.analisis_guardado = False
                    self.estado.analisis_sucio = True
                    self.estado.analisis_desactualizado = False
                    self.estado.firma_imagenes_analizadas = firmas
                    self._actualizar_estado_botones()
                    self._abrir_ventana_analisis()

                self.root.after(0, terminar)

            except Exception as exc:
                print("ERROR ANÁLISIS:", repr(exc))
                error_texto = repr(exc)

                def fallo(error_texto=error_texto):
                    self.processing = False
                    self._cerrar_progreso()
                    self._actualizar_estado_botones()
                    messagebox.showerror(
                        "DERMATEC — Análisis",
                        f"No fue posible completar el análisis.\n\n{error_texto}",
                        parent=self.root,
                    )

                self.root.after(0, fallo)

        threading.Thread(target=worker, daemon=True).start()

    def _agregar_resultados_multiimagen(self, resultados: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Agregación de las N imágenes de una misma lesión.

        Regla principal:
        - El resultado binario general se determina por mayoría de las
          predicciones binarias individuales.
        - En empate, la probabilidad media de malignidad actúa como
          desempate mediante el umbral del clasificador.
        - La clasificación multiclase se determina por mayoría de las
          clases predichas individualmente.
        - Las probabilidades medias se conservan como información
          complementaria y NO reemplazan la regla de mayoría.
        """
        if not resultados:
            raise RuntimeError("No hay resultados individuales para agregar.")

        # Hook futuro: si el paquete incorpora un agregador oficial validado.
        if self.model is not None and hasattr(self.model, "predict_lesion_results"):
            try:
                oficial = self.model.predict_lesion_results(
                    [x["resultado"] for x in resultados]
                )
                if isinstance(oficial, dict):
                    oficial.setdefault("metodo_agregacion", "modelo_multiimagen")
                    return oficial
            except Exception as exc:
                print(
                    "Agregador oficial no pudo usarse; "
                    "se utilizará agregación de mayoría:",
                    repr(exc),
                )

        clases = [
            "Nevo",
            "Melanoma",
            "Carcinoma basocelular",
            "Carcinoma escamocelular",
        ]

        # ---------------------------
        # Probabilidades medias
        # ---------------------------
        p_mal = float(np.mean([
            float(x["resultado"].get("probabilidad_malignidad", 0.0))
            for x in resultados
        ]))

        medias = {}
        for clase in clases:
            medias[clase] = float(np.mean([
                float(
                    (x["resultado"].get("probabilidades_clases", {}) or {})
                    .get(clase, 0.0)
                )
                for x in resultados
            ]))

        # ---------------------------
        # Mayoría BINARIA
        # ---------------------------
        binarias = [
            str(x["resultado"].get("resultado_binario", "")).upper()
            for x in resultados
        ]

        n_malignas = sum(x == "MALIGNA" for x in binarias)
        n_benignas = sum(x == "BENIGNA" for x in binarias)

        if n_malignas > n_benignas:
            resultado_binario = "MALIGNA"
            regla_binaria = "MAYORÍA_BINARIA"
        elif n_benignas > n_malignas:
            resultado_binario = "BENIGNA"
            regla_binaria = "MAYORÍA_BINARIA"
        else:
            # Desempate únicamente si hay el mismo número de benignas
            # y malignas.
            threshold = float(
                getattr(self.model, "threshold", 0.495)
            ) * 100.0

            resultado_binario = (
                "MALIGNA" if p_mal >= threshold else "BENIGNA"
            )
            regla_binaria = "EMPATE_PROBABILIDAD_MEDIA"

        # ---------------------------
        # Mayoría MULTICLASE
        # ---------------------------
        clases_individuales = [
            str(x["resultado"].get("clasificacion", "")).strip()
            for x in resultados
        ]

        conteo_clases = {
            clase: clases_individuales.count(clase)
            for clase in clases
        }

        max_conteo = max(conteo_clases.values()) if conteo_clases else 0
        candidatas = [
            clase for clase, n in conteo_clases.items()
            if n == max_conteo
        ]

        if len(candidatas) == 1:
            clasificacion = candidatas[0]
            regla_multiclase = "MAYORÍA_MULTICLASE"
        else:
            # Desempate multiclase mediante probabilidad media.
            clasificacion = max(
                candidatas,
                key=lambda clase: medias.get(clase, 0.0)
            )
            regla_multiclase = "EMPATE_PROBABILIDAD_MEDIA"

        threshold = float(
            getattr(self.model, "threshold", 0.495)
        )

        return {
            "modelo": getattr(self.model, "nombre", "DERMATEC"),
            "rama1": f"Rama 1 {RAMA1_VERSION}",
            "resultado_binario": resultado_binario,
            "probabilidad_malignidad": p_mal,
            "threshold_malignidad": threshold,
            "clasificacion": clasificacion,
            "probabilidades_clases": medias,
            "imagenes_analizadas": len(resultados),
            "imagenes_malignas": n_malignas,
            "imagenes_benignas": n_benignas,
            "consistencia_maligna": float(
                n_malignas / len(resultados) * 100.0
            ),
            "conteo_clases": conteo_clases,
            "metodo_agregacion": (
                "MAYORIA_BINARIA + MAYORIA_MULTICLASE"
            ),
            "regla_binaria": regla_binaria,
            "regla_multiclase": regla_multiclase,
            "nota_agregacion": (
                "El resultado general se obtiene a partir de las "
                "predicciones individuales de las N imágenes. "
                "La mayoría determina el resultado binario; en empate "
                "se utiliza la probabilidad media como desempate."
            ),
        }

    # ============================================================
    # PROGRESO
    # ============================================================

    def _abrir_progreso(self, total: int):
        win = tk.Toplevel(self.root)
        win.title("DERMATEC — Analizando lesión")
        win.geometry("520x230")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        self.progress_window = win

        cont = ttk.Frame(win, padding=24)
        cont.pack(fill="both", expand=True)
        ttk.Label(cont, text="Analizando lesión", style="DialogTitle.TLabel").pack(anchor="w")
        self.progress_label = ttk.Label(cont, text=f"Preparando {total} imágenes...")
        self.progress_label.pack(anchor="w", pady=(10, 16))
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(cont, variable=self.progress_var, maximum=max(total, 1), mode="determinate")
        self.progress_bar.pack(fill="x")
        ttk.Label(cont, text="No cierre DERMATEC mientras el análisis está en curso.", style="Subtitle.TLabel").pack(anchor="w", pady=(12, 0))

    def _actualizar_progreso_threadsafe(self, valor: int, total: int, texto: str):
        def f():
            if self.progress_window and self.progress_window.winfo_exists():
                self.progress_var.set(valor)
                self.progress_bar.configure(maximum=max(total, 1))
                self.progress_label.configure(text=texto)
        try:
            self.root.after(0, f)
        except Exception:
            pass

    def _cerrar_progreso(self):
        if self.progress_window:
            try:
                self.progress_window.grab_release()
                self.progress_window.destroy()
            except Exception:
                pass
        self.progress_window = None

    # ============================================================
    # VENTANA 2 — ANÁLISIS DE LESIÓN
    # ============================================================

    def _abrir_ventana_analisis(self):
        if not self.estado.analisis_generado or not self.estado.resultado_general:
            messagebox.showwarning("DERMATEC", "No existe un análisis disponible.", parent=self.root)
            return

        if self.analysis_window and self.analysis_window.winfo_exists():
            self._maximizar_ventana(self.analysis_window)
            self.analysis_window.lift()
            self.analysis_window.focus_force()
            return

        self.root.withdraw()
        win = tk.Toplevel(self.root)
        self.analysis_window = win
        win.title("DERMATEC — Análisis de lesión")
        # La pantalla 2 debe conservar el mismo estado de trabajo de la
        # aplicación principal. En Windows, state("zoomed") evita que el
        # contenido quede recortado y elimina la necesidad de maximizarla
        # manualmente después de cada análisis.
        win.geometry("1360x740")
        win.minsize(1100, 650)
        win.configure(bg=COLOR_BG)
        win.protocol("WM_DELETE_WINDOW", self.intentar_cerrar_aplicacion)
        self._maximizar_ventana(win)

        # Encabezado
        top = ttk.Frame(win, padding=(18, 12))
        top.pack(fill="x")
        izq = ttk.Frame(top)
        izq.pack(side="left", fill="x", expand=True)
        ttk.Label(izq, text="DERMATEC — ANÁLISIS DE LESIÓN", style="Title.TLabel").pack(anchor="w")
        p = self.estado.paciente
        l = self.estado.lesion
        ttk.Label(
            izq,
            text=f"Paciente: {p.nombre if p else '—'}   ·   Lesión: {l.nombre if l else '—'}",
            style="Subtitle.TLabel",
        ).pack(anchor="w")
        der = ttk.Frame(top)
        der.pack(side="right")
        btn_menu_analisis = ttk.Button(der, text="MENÚ")
        btn_menu_analisis.pack(side="left", padx=(0, 8))
        btn_menu_analisis.configure(command=lambda: self.mostrar_menu(btn_menu_analisis))
        ttk.Button(der, text="DISPOSITIVO", command=self.abrir_dispositivos).pack(side="left")

        main = ttk.Frame(win, padding=(18, 4, 18, 8))
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=7)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        # Galería
        gal_card = tk.Frame(main, bg=COLOR_CARD, highlightbackground=COLOR_BORDER, highlightthickness=1)
        gal_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        gal_card.grid_columnconfigure(0, weight=1)
        gal_card.grid_rowconfigure(1, weight=1)
        gh = tk.Frame(gal_card, bg=COLOR_CARD)
        gh.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        tk.Label(gh, text="IMÁGENES ANALIZADAS", bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Label(gh, text=f"{len(self.estado.resultados_individuales)} imagen(es)", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(side="right")

        canvas = tk.Canvas(gal_card, bg=COLOR_CARD, highlightthickness=0)
        scroll = ttk.Scrollbar(gal_card, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=COLOR_CARD)
        win_id = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(0, 8))
        scroll.grid(row=1, column=1, sticky="ns", pady=(0, 8))
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))
        self.analysis_gallery_frame = frame
        self.analysis_thumb_refs.clear()
        self._poblar_galeria_analisis()

        # Resultado general
        res_card = tk.Frame(
            main,
            bg=COLOR_CARD,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        res_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        res_card.grid_columnconfigure(0, weight=1)
        res_card.grid_rowconfigure(0, weight=1)

        res_canvas = tk.Canvas(
            res_card,
            bg=COLOR_CARD,
            highlightthickness=0,
            bd=0,
        )
        res_scroll = ttk.Scrollbar(
            res_card,
            orient="vertical",
            command=res_canvas.yview,
        )
        res_inner = tk.Frame(res_canvas, bg=COLOR_CARD)

        res_window_id = res_canvas.create_window(
            (0, 0),
            window=res_inner,
            anchor="nw",
        )
        res_canvas.configure(yscrollcommand=res_scroll.set)

        res_canvas.grid(row=0, column=0, sticky="nsew")
        res_scroll.grid(row=0, column=1, sticky="ns")

        res_inner.bind(
            "<Configure>",
            lambda e: res_canvas.configure(
                scrollregion=res_canvas.bbox("all")
            ),
        )
        res_canvas.bind(
            "<Configure>",
            lambda e: res_canvas.itemconfigure(
                res_window_id,
                width=e.width,
            ),
        )

        self._poblar_resultado_general(res_inner)

        # Herramientas
        tools = ttk.Frame(win, padding=(18, 2, 18, 6))
        tools.pack(fill="x")
        centro = ttk.Frame(tools)
        centro.pack()
        ttk.Button(centro, text="PARÁMETROS", style="Secondary.TButton", command=self.abrir_parametros).grid(row=0, column=0, padx=8)
        ttk.Button(centro, text="SEGMENTACIÓN", style="Secondary.TButton", command=self.abrir_segmentacion).grid(row=0, column=1, padx=8)
        ttk.Button(centro, text="OBSERVACIONES", style="Secondary.TButton", command=self.abrir_observaciones).grid(row=0, column=2, padx=8)

        # Acciones inferiores
        bottom = ttk.Frame(win, padding=(18, 4, 18, 14))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="← VOLVER A CAPTURA", command=self.volver_a_captura).pack(side="left")
        ttk.Button(bottom, text="IMPRIMIR", command=self.imprimir_informe).pack(side="right")
        ttk.Button(bottom, text="EXPORTAR", command=self.exportar_informe).pack(side="right", padx=8)
        ttk.Button(bottom, text="GUARDAR ANÁLISIS", style="Primary.TButton", command=self.guardar_analisis_actual).pack(side="right", padx=8)

    def _poblar_galeria_analisis(self):
        frame = self.analysis_gallery_frame
        for w in frame.winfo_children():
            w.destroy()

        resultados = self.estado.resultados_individuales
        columnas = 3
        for idx, item in enumerate(resultados):
            row = idx // columnas
            col = idx % columnas
            card = tk.Frame(frame, bg="#F8FBFB", highlightbackground=COLOR_BORDER, highlightthickness=1)
            card.grid(row=row, column=col, sticky="nsew", padx=7, pady=7)
            frame.grid_columnconfigure(col, weight=1)

            ruta = self.estado.lesion.carpeta / "imagenes" / item.get("imagen", "")
            try:
                img = Image.open(ruta).convert("RGB")
                img.thumbnail((210, 135), Image.Resampling.LANCZOS)
                ph = ImageTk.PhotoImage(img)
                self.analysis_thumb_refs.append(ph)
                lbl = tk.Label(card, image=ph, bg="#F8FBFB", cursor="hand2")
                lbl.pack(padx=8, pady=(8, 4))
                lbl.bind("<Button-1>", lambda e, i=idx: self._abrir_detalle_resultado_individual(i))
            except Exception:
                tk.Label(card, text="Imagen no disponible", bg="#F8FBFB", fg=COLOR_MUTED, height=8).pack(fill="x", padx=8, pady=8)

            r = item.get("resultado", {}) or {}
            decision = str(r.get("resultado_binario", "—"))
            clase = str(r.get("clasificacion", "—"))
            pm = r.get("probabilidad_malignidad")
            color = COLOR_RED if decision.upper() == "MALIGNA" else (COLOR_AMBER if "NO CONCLUYENTE" in decision.upper() else COLOR_GREEN)
            tk.Label(card, text=f"Imagen {idx + 1}", bg="#F8FBFB", fg=COLOR_TEXT, font=("Segoe UI", 9, "bold")).pack()
            tk.Label(card, text=decision, bg="#F8FBFB", fg=color, font=("Segoe UI", 9, "bold")).pack()
            tk.Label(card, text=clase, bg="#F8FBFB", fg=COLOR_TEXT, font=("Segoe UI", 8)).pack()
            if r.get("aviso_calidad"):
                tk.Label(card, text="⚠ baja confianza", bg="#F8FBFB", fg=COLOR_AMBER, font=("Segoe UI", 8, "bold")).pack()
            if pm is not None:
                tk.Label(card, text=f"Malignidad: {float(pm):.1f}%", bg="#F8FBFB", fg=COLOR_MUTED, font=("Segoe UI", 8)).pack(pady=(0, 8))

    def _poblar_resultado_general(self, parent):
        gen = self.estado.resultado_general or {}
        tk.Label(parent, text="RESULTADO GENERAL", bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=18, pady=(18, 12))

        decision = str(gen.get("resultado_binario", "—"))
        color = COLOR_RED if decision.upper() == "MALIGNA" else (COLOR_AMBER if "NO CONCLUYENTE" in decision.upper() else COLOR_GREEN)
        fondo = COLOR_RED_SOFT if decision.upper() == "MALIGNA" else (COLOR_AMBER_SOFT if "NO CONCLUYENTE" in decision.upper() else COLOR_GREEN_SOFT)
        badge = tk.Label(parent, text=f"  {decision}  ", bg=fondo, fg=color, font=("Segoe UI", 22, "bold"), padx=12, pady=8)
        badge.pack(anchor="w", padx=18, pady=(0, 16))

        tk.Label(parent, text="Clasificación sugerida", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=18)
        tk.Label(parent, text=str(gen.get("clasificacion", "—")), bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 17, "bold"), wraplength=320, justify="left").pack(anchor="w", padx=18, pady=(2, 14))
        if gen.get("aviso_calidad"):
            tk.Label(parent, text=str(gen["aviso_calidad"]), bg=COLOR_AMBER_SOFT, fg=COLOR_AMBER, font=("Segoe UI", 9, "bold"), wraplength=320, justify="left", padx=8, pady=6).pack(anchor="w", fill="x", padx=18, pady=(0, 14))

        pm = gen.get("probabilidad_malignidad")
        if pm is not None:
            tk.Label(parent, text="Probabilidad de malignidad agregada", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=18)
            tk.Label(parent, text=f"{float(pm):.2f}%", bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=18, pady=(2, 14))

        tk.Label(parent, text="Imágenes analizadas", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=18)
        tk.Label(parent, text=str(gen.get("imagenes_analizadas", len(self.estado.resultados_individuales))), bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=18, pady=(2, 12))

        if gen.get("imagenes_malignas") is not None:
            tk.Label(parent, text="Consistencia entre imágenes", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=18)
            texto = f"{gen.get('imagenes_malignas')} de {gen.get('imagenes_analizadas')} imágenes clasificadas como malignas"
            tk.Label(parent, text=texto, bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 9), wraplength=320, justify="left").pack(anchor="w", padx=18, pady=(2, 14))

        tk.Label(parent, text="Consideraciones", bg=COLOR_CARD, fg=COLOR_MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=18)
        consideracion = self._consideracion_general(gen)
        tk.Label(parent, text=consideracion, bg=COLOR_CARD, fg=COLOR_TEXT, font=("Segoe UI", 9), wraplength=320, justify="left").pack(anchor="w", padx=18, pady=(4, 14))

        tk.Label(
            parent,
            text="Herramienta de apoyo. No sustituye el diagnóstico médico.",
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            font=("Segoe UI", 8),
            wraplength=320,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(4, 18))

    @staticmethod
    def _consideracion_general(gen: dict[str, Any]) -> str:
        n = int(gen.get("imagenes_analizadas", 0) or 0)
        mal = int(gen.get("imagenes_malignas", 0) or 0)
        ben = int(gen.get("imagenes_benignas", n - mal) or 0)

        if mal > ben:
            return (
                f"La mayoría de las imágenes ({mal} de {n}) fue clasificada "
                "como maligna. Revise las predicciones individuales, la "
                "segmentación y los parámetros cuantitativos antes de "
                "documentar la valoración profesional."
            )

        if ben > mal:
            return (
                f"La mayoría de las imágenes ({ben} de {n}) fue clasificada "
                "como benigna. Revise las predicciones individuales, la "
                "segmentación y los parámetros cuantitativos junto con la "
                "valoración clínica del profesional."
            )

        return (
            "Las predicciones binarias presentan un empate. El sistema "
            "utiliza la probabilidad media de malignidad como criterio de "
            "desempate. Revise las predicciones individuales, la "
            "segmentación y los parámetros cuantitativos junto con la "
            "valoración clínica."
        )

    def _abrir_detalle_resultado_individual(self, indice: int):
        item = self.estado.resultados_individuales[indice]
        ruta = self.estado.lesion.carpeta / "imagenes" / item.get("imagen", "")
        r = item.get("resultado", {}) or {}
        r1 = r.get("rama1", {}) or {}
        p1 = r1.get("parametros", {}) or {}

        win = tk.Toplevel(self.analysis_window or self.root)
        win.title(f"DERMATEC — Imagen {indice + 1}")
        win.geometry("1050x760")
        win.transient(self.analysis_window or self.root)

        cont = ttk.Frame(win, padding=16)
        cont.pack(fill="both", expand=True)
        cont.columnconfigure(0, weight=3)
        cont.columnconfigure(1, weight=2)
        cont.rowconfigure(0, weight=1)

        izq = ttk.Frame(cont)
        izq.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        try:
            img = Image.open(ruta).convert("RGB")
            ancho0 = img.width
            img.thumbnail((600, 580), Image.Resampling.LANCZOS)
            dibujar_regla_mm(img, ancho0)
            ph = ImageTk.PhotoImage(img)
            lbl = ttk.Label(izq, image=ph)
            lbl.image = ph
            lbl.pack(expand=True)
        except Exception:
            ttk.Label(izq, text="Imagen no disponible").pack(expand=True)

        der = ttk.Frame(cont)
        der.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        ttk.Label(
            der, text=f"Imagen {indice + 1}",
            style="DialogTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            der, text=item.get("imagen", ""), wraplength=360
        ).pack(anchor="w", pady=(4, 14))

        ttk.Label(
            der, text=f"Clasificador {getattr(self.model, 'nombre', 'DERMATEC')} — Predicción individual",
            style="Header.TLabel"
        ).pack(anchor="w", pady=(4, 4))
        ttk.Label(
            der, text=f"Resultado: {r.get('resultado_binario', '—')}"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            der, text=f"Clasificación: {r.get('clasificacion', '—')}"
        ).pack(anchor="w", pady=2)
        if r.get("incertidumbre_entropia_bits") is not None:
            ttk.Label(
                der,
                text=f"Incertidumbre: {float(r['incertidumbre_entropia_bits']):.2f} bits (0 = seguro, 2 = máxima duda)"
            ).pack(anchor="w", pady=2)

        if r.get("probabilidad_malignidad") is not None:
            ttk.Label(
                der,
                text=f"Malignidad: {float(r['probabilidad_malignidad']):.2f}%"
            ).pack(anchor="w", pady=2)
        for motivo in (r.get("control_calidad", {}) or {}).get("motivos", []):
            ttk.Label(der, text=f"⚠ {motivo}", wraplength=420, foreground=COLOR_AMBER).pack(anchor="w", pady=2)

        ttk.Separator(der).pack(fill="x", pady=10)

        ttk.Label(
            der, text="Rama 1 — Segmentación",
            style="Header.TLabel"
        ).pack(anchor="w", pady=(2, 4))

        ttk.Label(
            der,
            text=f"Válida: {'Sí' if r1.get('valida') else 'No'}"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            der,
            text=f"Calidad: {float(r1.get('calidad', 0.0)):.3f}"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            der,
            text=f"Área: {float(p1.get('area_px', 0.0)):.0f} px"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            der,
            text=f"Área relativa: {100.0 * float(p1.get('area_fraction', 0.0)):.2f}%"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            der,
            text=f"Resolución: {p1.get('resolucion_original_px', '—')}"
        ).pack(anchor="w", pady=2)
        ttk.Label(
            der,
            text=f"Tiempo Rama 1: {float(p1.get('tiempo_total_rama1_s', 0.0)):.3f} s"
        ).pack(anchor="w", pady=2)

        ttk.Button(
            der,
            text="Ver segmentación",
            command=self.abrir_segmentacion,
        ).pack(anchor="w", pady=(16, 0))

    # ============================================================
    # PARÁMETROS
    # ============================================================

    PARAM_GROUPS = {
        "area_relative": "Morfología",
        "circularity": "Morfología",
        "irregularity": "Morfología / borde",
        "solidity": "Morfología",
        "asymmetry_mean": "Morfología / asimetría",
        "contrast_lab": "Color / contraste",
        "lab_b_std": "Colorimetría",
        "entropy": "Textura",
        "uniformity": "Textura",
        "gradient_mean": "Bordes / gradiente",
    }

    def abrir_parametros(self):
        if not self.estado.resultados_individuales:
            return
        parent = self.analysis_window or self.root
        win = tk.Toplevel(parent)
        win.title("DERMATEC — Parámetros de la lesión")
        win.geometry(_tamano_ventana(win, 1180, 660))
        win.minsize(940, 540)
        win.transient(parent)

        cont = ttk.Frame(win, padding=16)
        cont.pack(fill="both", expand=True)
        ttk.Label(cont, text="Parámetros cuantitativos de la lesión (Rama 1)", style="DialogTitle.TLabel").pack(anchor="w")
        ttk.Label(
            cont,
            text="Medidas calculadas sobre esta imagen con la máscara de la Rama 1. Son información de apoyo para el "
                 "profesional (criterios ABCD); no son una probabilidad diagnóstica.",
            style="Subtitle.TLabel", wraplength=1100,
        ).pack(anchor="w", pady=(2, 4))
        lbl_escala = tk.Label(cont, text="", bg=COLOR_BG, fg=COLOR_MUTED, font=("Segoe UI", 9, "bold"), anchor="w")
        lbl_escala.pack(anchor="w", pady=(0, 10))

        body = ttk.Frame(cont)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)

        lista = tk.Listbox(body, font=("Segoe UI", 10), borderwidth=0, highlightthickness=1, highlightbackground=COLOR_BORDER, selectbackground=COLOR_TEAL, selectforeground="white")
        lista.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        for i, item in enumerate(self.estado.resultados_individuales, start=1):
            lista.insert("end", f"Imagen {i} — {item.get('imagen', '')[:24]}")

        tree = ttk.Treeview(body, columns=("grupo", "parametro", "valor", "descripcion"), show="headings")
        tree.heading("grupo", text="Grupo")
        tree.heading("parametro", text="Parámetro")
        tree.heading("valor", text="Valor")
        tree.heading("descripcion", text="Qué representa")
        tree.column("grupo", width=110, anchor="w", stretch=False)
        tree.column("parametro", width=230, anchor="w", stretch=False)
        tree.column("valor", width=120, anchor="center", stretch=False)
        tree.column("descripcion", width=560, anchor="w")
        tree.tag_configure("par", background="#F7FAFB")
        tree.grid(row=0, column=1, sticky="nsew")
        scroll = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        scroll.grid(row=0, column=2, sticky="ns")
        tree.configure(yscrollcommand=scroll.set)

        def cargar(idx: int):
            for x in tree.get_children():
                tree.delete(x)
            resultado = self.estado.resultados_individuales[idx].get("resultado", {}) or {}
            esc = resultado.get("escala")
            if esc:
                estimada = esc.get("estado") != "calibrada"
                lbl_escala.configure(text=escala_acople.texto_escala(esc), fg=COLOR_AMBER if estimada else COLOR_TEAL_DARK)
            else:
                lbl_escala.configure(text="Análisis de una versión anterior: vuelva a analizar para ver los parámetros en mm.", fg=COLOR_AMBER)
            params = resultado.get("parametros_lesion", {}) or {}
            for n, (clave, p) in enumerate(params.items()):
                if not isinstance(p, dict):
                    continue
                valor = parametros_lesion.formatear(p.get("valor"), p.get("unidad", ""))
                unidad = p.get("unidad", "")
                if valor != "—" and unidad and not unidad[0].isdigit() and not unidad.startswith("≥"):
                    valor = f"{valor} {unidad.split(' ')[0]}"
                tree.insert("", "end", values=(p.get("grupo", ""), p.get("nombre", clave), valor, p.get("descripcion", "")),
                            tags=("par",) if n % 2 else ())
            p1 = ((resultado.get("rama1", {}) or {}).get("parametros", {}) or {})
            if p1:
                t = p1.get("tiempo_total_rama1_s")
                tree.insert("", "end", values=("Proceso", "Rama 1", str(p1.get("rama1_version", "v1")),
                                               "Versión del algoritmo de segmentación."))
                if t is not None:
                    tree.insert("", "end", values=("Proceso", "Tiempo de la Rama 1", f"{float(t):.2f} s".replace(".", ","),
                                                   "Tiempo de eliminación de vello, segmentación y cálculo de parámetros."))
                if p1.get("resolucion_original_px"):
                    tree.insert("", "end", values=("Proceso", "Resolución de la captura", f"{p1['resolucion_original_px']} px",
                                                   "Tamaño de la imagen analizada."))

        def seleccionar(event=None):
            sel = lista.curselection()
            if sel:
                cargar(sel[0])

        lista.bind("<<ListboxSelect>>", seleccionar)
        if self.estado.resultados_individuales:
            lista.selection_set(0)
            cargar(0)

        pie = ttk.Frame(cont)
        pie.pack(fill="x", pady=(12, 0))
        ttk.Button(pie, text="Calibrar escala (mm)", command=self.calibrar_escala).pack(side="left")
        ttk.Button(pie, text="Cerrar", command=win.destroy).pack(side="right")

    # ============================================================
    # SEGMENTACIÓN
    # ============================================================

    def abrir_segmentacion(self):
        if not self.estado.resultados_individuales:
            return

        parent = self.analysis_window or self.root
        win = tk.Toplevel(parent)
        win.title("DERMATEC — Rama 1 / Segmentación")
        win.geometry(_tamano_ventana(win, 1080, 690))
        win.minsize(900, 580)
        win.transient(parent)

        cont = ttk.Frame(win, padding=16)
        cont.pack(fill="both", expand=True)

        ttk.Label(
            cont, text="Rama 1 — remoción de vello + segmentación",
            style="DialogTitle.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            cont,
            wraplength=1000,
            text=(
                ("Imagen original → eliminación de vello (DullRazor + inpainting) → exclusión de fondo/acople → corrección de "
                 "sombreado → mapa de color CIELAB → umbral de Otsu → selección de la lesión → refinamiento GrabCut → "
                 "DILATE_5 → ROI. " if RAMA1_VERSION == "v2" else "ORIGINAL → B2.23 → GMM_LAB_POST_B38 → DILATE_5 → ROI. ") +
                "La red neuronal (Rama 2) recibe la imagen original, no esta."
            ),
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 12))

        body = ttk.Frame(cont)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=5)
        body.rowconfigure(0, weight=1)

        lista = tk.Listbox(
            body,
            font=("Segoe UI", 10),
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            selectbackground=COLOR_TEAL,
            selectforeground="white",
        )
        lista.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        for i, item in enumerate(self.estado.resultados_individuales, start=1):
            lista.insert("end", f"Imagen {i}")

        grid = tk.Frame(body, bg=COLOR_BG)
        grid.grid(row=0, column=1, sticky="nsew")

        for r in range(2):
            grid.grid_rowconfigure(r, weight=1)
        for c in range(2):
            grid.grid_columnconfigure(c, weight=1)

        labels_img = []
        refs = []
        titulos = [
            ("Sin vellos" if RAMA1_VERSION == "v2" else "B2.23 sin vellos"),
            "Máscara final",
            "Contorno",
            "ROI segmentada",
        ]

        for i, titulo in enumerate(titulos):
            card = tk.Frame(
                grid,
                bg=COLOR_CARD,
                highlightbackground=COLOR_BORDER,
                highlightthickness=1,
            )
            card.grid(
                row=i // 2,
                column=i % 2,
                sticky="nsew",
                padx=6,
                pady=6,
            )
            tk.Label(
                card,
                text=titulo,
                bg=COLOR_CARD,
                fg=COLOR_TEXT,
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", padx=8, pady=(7, 3))
            lbl = tk.Label(
                card, bg="#10181B", fg="white", text="Cargando..."
            )
            lbl.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            labels_img.append(lbl)

        status = ttk.Label(cont, text="")
        status.pack(anchor="w", pady=(6, 0))

        def show(lbl, path, gray=False):
            if not path:
                lbl.configure(image="", text="No disponible")
                return

            p = self.estado.lesion.carpeta / path
            datos = np.fromfile(str(p), dtype=np.uint8)
            if datos.size == 0:
                lbl.configure(image="", text="No disponible")
                return
            arr = cv2.imdecode(
                datos,
                cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
            )
            if arr is None:
                lbl.configure(image="", text="No disponible")
                return

            if gray:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(arr)
            img.thumbnail((430, 290), Image.Resampling.LANCZOS)
            ph = ImageTk.PhotoImage(img)
            refs.append(ph)
            lbl.configure(image=ph, text="")
            lbl.image = ph

        def cargar(idx):
            refs.clear()
            item = self.estado.resultados_individuales[idx]
            r = item.get("resultado", {}) or {}
            r1 = r.get("rama1", {}) or {}
            show(labels_img[0], r1.get("imagen_limpia_b223"))
            show(labels_img[1], r1.get("mascara_final"), gray=True)
            show(labels_img[2], r1.get("contorno"))
            show(labels_img[3], r1.get("roi"))

            p = r1.get("parametros", {}) or {}
            status.configure(
                text=(
                    f"Segmentación válida: {'Sí' if r1.get('valida') else 'No'}"
                    f" · Calidad: {float(r1.get('calidad', 0.0)):.3f}"
                    f" · Tiempo Rama 1: "
                    f"{float(p.get('tiempo_total_rama1_s', 0.0)):.3f} s"
                )
            )

        def seleccionar(event=None):
            sel = lista.curselection()
            if sel:
                cargar(sel[0])

        lista.bind("<<ListboxSelect>>", seleccionar)

        if self.estado.resultados_individuales:
            lista.selection_set(0)
            cargar(0)

        ttk.Button(
            cont, text="Cerrar", command=win.destroy
        ).pack(anchor="e", pady=(10, 0))

    # ============================================================
    # OBSERVACIONES
    # ============================================================

    def abrir_observaciones(self):
        parent = self.analysis_window or self.root
        win = tk.Toplevel(parent)
        win.title("DERMATEC — Observaciones del profesional")
        win.geometry("760x570")
        win.minsize(650, 500)
        win.transient(parent)

        cont = ttk.Frame(win, padding=18)
        cont.pack(fill="both", expand=True)
        ttk.Label(cont, text="Observaciones del profesional", style="DialogTitle.TLabel").pack(anchor="w")
        ttk.Label(cont, text="Estas observaciones quedan asociadas a la lesión y se incluirán en el informe del paciente.", style="Subtitle.TLabel").pack(anchor="w", pady=(2, 14))

        ttk.Label(cont, text="Localización anatómica", style="Header.TLabel").pack(anchor="w")
        var_loc = tk.StringVar(value=self.estado.observaciones.get("localizacion_anatomica", ""))
        entry = ttk.Entry(cont, textvariable=var_loc)
        entry.pack(fill="x", pady=(4, 14))

        ttk.Label(cont, text="Observaciones", style="Header.TLabel").pack(anchor="w")
        texto = tk.Text(cont, height=14, wrap="word", font=("Segoe UI", 10), borderwidth=1, relief="solid")
        texto.pack(fill="both", expand=True, pady=(4, 12))
        texto.insert("1.0", self.estado.observaciones.get("texto", ""))

        original_loc = var_loc.get()
        original_txt = texto.get("1.0", "end-1c")
        guardado_local = {"ok": False}

        def hay_cambios_locales():
            return var_loc.get() != original_loc or texto.get("1.0", "end-1c") != original_txt

        def guardar():
            self.estado.observaciones = {
                "localizacion_anatomica": var_loc.get().strip(),
                "texto": texto.get("1.0", "end-1c").strip(),
            }
            self.estado.analisis_sucio = True
            self.estado.analisis_guardado = False
            guardado_local["ok"] = True
            messagebox.showinfo("DERMATEC", "Observaciones incorporadas al análisis actual.\n\nUse GUARDAR ANÁLISIS para persistir todos los cambios de la lesión.", parent=win)

        def cerrar():
            if hay_cambios_locales() and not guardado_local["ok"]:
                r = DialogoTresOpciones.preguntar(
                    win,
                    "Observaciones sin incorporar",
                    "Ha escrito o modificado observaciones que todavía no se han incorporado al análisis actual.",
                    "Incorporar cambios",
                    "Descartar cambios",
                )
                if r == "cancelar" or r is None:
                    return
                if r == "opcion_1":
                    self.estado.observaciones = {
                        "localizacion_anatomica": var_loc.get().strip(),
                        "texto": texto.get("1.0", "end-1c").strip(),
                    }
                    self.estado.analisis_sucio = True
                    self.estado.analisis_guardado = False
            win.destroy()

        barra = ttk.Frame(cont)
        barra.pack(fill="x")
        ttk.Button(barra, text="Cerrar", command=cerrar).pack(side="right")
        ttk.Button(barra, text="Incorporar observaciones", style="Primary.TButton", command=guardar).pack(side="right", padx=(0, 8))
        win.protocol("WM_DELETE_WINDOW", cerrar)

    # ============================================================
    # GUARDADO / EXPORTACIÓN / IMPRESIÓN
    # ============================================================

    def _construir_payload_analisis(self) -> dict[str, Any]:
        if not self.estado.paciente or not self.estado.lesion:
            raise RuntimeError("No hay paciente/lesión activos.")
        if not self.estado.resultado_general:
            raise RuntimeError("No hay un análisis generado para guardar.")

        return {
            "app": APP_NAME,
            "app_version": APP_VERSION,
            "fecha_guardado": ahora_iso(),
            "paciente": {
                "id": self.estado.paciente.id,
                "nombre": self.estado.paciente.nombre,
            },
            "lesion": {
                "id": self.estado.lesion.id,
                "nombre": self.estado.lesion.nombre,
            },
            "modelo": getattr(self.model, "nombre", "DERMATEC"),
            "firma_imagenes": list(self.estado.firma_imagenes_analizadas),
            "resultado_general": self.estado.resultado_general,
            "resultados_individuales": self.estado.resultados_individuales,
            "observaciones": self.estado.observaciones,
            "advertencia": "DERMATEC es una herramienta de apoyo para evaluación inicial y no sustituye el diagnóstico médico.",
        }

    def guardar_analisis_actual(self, silencioso: bool = False) -> bool:
        if not self.estado.lesion or not self.estado.resultado_general:
            if not silencioso:
                messagebox.showwarning("DERMATEC", "No existe un análisis generado para guardar.", parent=self.analysis_window or self.root)
            return False
        try:
            payload = self._construir_payload_analisis()
            ruta = self.gestor.guardar_analisis(self.estado.lesion, payload)
            self.estado.analisis_guardado = True
            self.estado.analisis_sucio = False
            self.estado.analisis_desactualizado = False
            if not silencioso:
                messagebox.showinfo("DERMATEC", f"Análisis guardado correctamente.\n\n{ruta}", parent=self.analysis_window or self.root)
            return True
        except Exception as exc:
            if not silencioso:
                messagebox.showerror("DERMATEC", f"No fue posible guardar el análisis.\n\n{exc}", parent=self.analysis_window or self.root)
            return False

    def exportar_informe(self):
        if not self.estado.paciente:
            messagebox.showwarning("DERMATEC", "No hay un paciente activo.", parent=self.analysis_window or self.root)
            return
        if self.estado.analisis_sucio:
            messagebox.showwarning(
                "DERMATEC",
                "El análisis actual tiene cambios sin guardar.\n\nGuarde la lesión antes de exportar el informe consolidado.",
                parent=self.analysis_window or self.root,
            )
            return
        try:
            pdf = self.gestor.exportar_pdf_paciente(self.estado.paciente)
            self.estado.ruta_ultimo_pdf = pdf
            messagebox.showinfo("DERMATEC", f"Informe exportado correctamente.\n\n{pdf}", parent=self.analysis_window or self.root)
            abrir_archivo(pdf)
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No fue posible exportar el informe.\n\n{exc}", parent=self.analysis_window or self.root)

    def imprimir_informe(self):
        if not self.estado.paciente:
            messagebox.showwarning("DERMATEC", "No hay un paciente activo.", parent=self.analysis_window or self.root)
            return
        if self.estado.analisis_sucio:
            messagebox.showwarning("DERMATEC", "Guarde el análisis actual antes de imprimir.", parent=self.analysis_window or self.root)
            return
        try:
            pdf = self.estado.ruta_ultimo_pdf
            if not pdf or not pdf.is_file():
                pdf = self.gestor.exportar_pdf_paciente(self.estado.paciente)
                self.estado.ruta_ultimo_pdf = pdf

            if ES_WINDOWS:
                os.startfile(str(pdf), "print")
                messagebox.showinfo("DERMATEC", "El informe fue enviado al sistema de impresión de Windows.", parent=self.analysis_window or self.root)
            elif shutil.which("lp"):
                subprocess.Popen(["lp", str(pdf)])
                messagebox.showinfo("DERMATEC", "El informe fue enviado a CUPS/lp.", parent=self.analysis_window or self.root)
            else:
                messagebox.showinfo("DERMATEC", "No se encontró un servicio de impresión configurado. Se abrirá el PDF para imprimirlo manualmente.", parent=self.analysis_window or self.root)
                abrir_archivo(pdf)
        except Exception as exc:
            messagebox.showerror("DERMATEC", f"No fue posible imprimir el informe.\n\n{exc}", parent=self.analysis_window or self.root)

    def abrir_carpeta_paciente(self):
        if not self.estado.paciente:
            messagebox.showwarning("DERMATEC", "No hay un paciente activo.", parent=self.root)
            return
        abrir_carpeta(self.estado.paciente.carpeta)

    # ============================================================
    # VOLVER A CAPTURA / DESCARTAR CAMBIOS
    # ============================================================

    def volver_a_captura(self):
        if self.processing:
            return
        if self.estado.analisis_sucio:
            r = DialogoTresOpciones.preguntar(
                self.analysis_window,
                "Análisis sin guardar",
                "Hay resultados u observaciones que todavía no han sido guardados.\n\nSe recomienda guardar antes de volver al entorno de captura.",
                "Guardar y volver",
                "Volver sin guardar",
            )
            if r == "cancelar" or r is None:
                return
            if r == "opcion_1":
                if not self.guardar_analisis_actual(silencioso=False):
                    return
            elif r == "opcion_2":
                self._descartar_cambios_analisis()

        self._cerrar_ventana_analisis_y_mostrar_captura()

    def _cerrar_ventana_analisis_y_mostrar_captura(self):
        if self.analysis_window:
            try:
                self.analysis_window.destroy()
            except Exception:
                pass
        self.analysis_window = None
        self.root.deiconify()
        self._maximizar_ventana(self.root)
        self.root.lift()
        self._refrescar_miniaturas()
        self._actualizar_contexto()

    def _descartar_cambios_analisis(self):
        if self.estado.lesion:
            guardado = self.gestor.cargar_analisis(self.estado.lesion)
        else:
            guardado = None
        if guardado:
            self.estado.resultados_individuales = guardado.get("resultados_individuales", []) or []
            self.estado.resultado_general = guardado.get("resultado_general")
            self.estado.observaciones = guardado.get("observaciones", {"localizacion_anatomica": "", "texto": ""}) or {"localizacion_anatomica": "", "texto": ""}
            self.estado.firma_imagenes_analizadas = guardado.get("firma_imagenes", []) or []
            self.estado.analisis_generado = bool(self.estado.resultado_general)
            self.estado.analisis_guardado = True
        else:
            self.estado.resultados_individuales = []
            self.estado.resultado_general = None
            self.estado.observaciones = {"localizacion_anatomica": "", "texto": ""}
            self.estado.firma_imagenes_analizadas = []
            self.estado.analisis_generado = False
            self.estado.analisis_guardado = False
        self.estado.analisis_sucio = False

    # ============================================================
    # PROTECCIÓN DE CIERRE
    # ============================================================

    def _hay_imagenes_sin_analisis_actualizado(self) -> bool:
        if not self.estado.lesion:
            return False
        imagenes = self.gestor.listar_imagenes(self.estado.lesion)
        if not imagenes:
            return False
        if self.estado.analisis_desactualizado:
            return True
        if not self.estado.analisis_generado and not self.gestor.cargar_analisis(self.estado.lesion):
            return True
        return False

    def intentar_cerrar_aplicacion(self):
        if self.processing:
            messagebox.showwarning(
                "DERMATEC",
                "Hay un análisis en curso. Espere a que termine antes de cerrar la aplicación.",
                parent=self.analysis_window or self.root,
            )
            return

        parent = self.analysis_window or self.root

        if self.estado.analisis_sucio:
            r = DialogoTresOpciones.preguntar(
                parent,
                "Cambios sin guardar",
                "El análisis actual contiene resultados u observaciones que todavía no han sido guardados.\n\nSi sale sin guardar, esos cambios del análisis se perderán. Las imágenes capturadas permanecen guardadas en la carpeta de la lesión.",
                "Guardar y cerrar",
                "Salir sin guardar",
            )
            if r == "cancelar" or r is None:
                return
            if r == "opcion_1":
                if not self.guardar_analisis_actual(silencioso=False):
                    return
            elif r == "opcion_2":
                pass

        elif self._hay_imagenes_sin_analisis_actualizado():
            ok = messagebox.askyesno(
                "DERMATEC — Cerrar sesión",
                "La lesión actual contiene imágenes nuevas o no tiene un análisis actualizado.\n\n"
                "Las fotografías ya están guardadas y NO se perderán, pero cerrará DERMATEC sin completar/actualizar el análisis de esta lesión.\n\n"
                "¿Desea cerrar de todas formas?",
                parent=parent,
            )
            if not ok:
                return

        else:
            ok = messagebox.askyesno(
                "Cerrar DERMATEC",
                "Todo el trabajo actual está guardado.\n\n¿Desea cerrar DERMATEC?",
                parent=parent,
            )
            if not ok:
                return

        self._cerrar_definitivamente()

    def _cerrar_definitivamente(self):
        self.app_activa = False
        try:
            self.camera.desconectar()
        except Exception:
            pass
        try:
            if self.analysis_window:
                self.analysis_window.destroy()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ============================================================
    # ESTADO UI
    # ============================================================

    def _firmas_imagenes_actuales(self) -> list[str]:
        firmas = []
        for p in self.estado.imagenes:
            try:
                firmas.append(sha256_file(p))
            except Exception:
                firmas.append("")
        return firmas

    def _actualizar_contexto(self):
        p = self.estado.paciente
        l = self.estado.lesion
        texto = f"Paciente: {p.nombre if p else 'No seleccionado'}   ·   Lesión: {l.nombre if l else 'No seleccionada'}"
        self.label_contexto.configure(text=texto)
        self._actualizar_estado_botones()

    def _actualizar_estado_botones(self):
        tiene_lesion = self.estado.lesion is not None
        tiene_imagenes = bool(self.estado.imagenes)
        tiene_frame = self.camera.obtener_ultimo_frame() is not None if hasattr(self, "camera") else False
        self.btn_capturar.configure(state="normal" if (tiene_lesion and tiene_frame and not self.processing) else "disabled")
        self.btn_nueva_lesion.configure(state="normal" if (self.estado.paciente and not self.processing) else "disabled")
        self.btn_analizar.configure(state="normal" if (tiene_lesion and tiene_imagenes and not self.processing) else "disabled")

    # ============================================================
    # EJECUCIÓN
    # ============================================================

    def run(self):
        print("=" * 90)
        print(f"DERMATEC APP V{APP_VERSION} — {'Raspberry Pi' if ES_LINUX else 'PC'}")
        print(f"Datos: {DATA_ROOT}")
        print("Windows DirectShow:", "DISPONIBLE" if DIRECTSHOW_DISPONIBLE else "NO DISPONIBLE")
        print("=" * 90)
        self.root.mainloop()
        print("DERMATEC V6 cerrado correctamente.")


# ================================================================
# 9. MAIN
# ================================================================


if __name__ == "__main__":
    app = DermatecApp()
    app.run()
