"""
DERMATEC — Escala del acople (píxeles por milímetro) y calibrador
================================================================
La escala depende del acople (distancia lente–piel) y del zoom con el que se enfoca. Se calibra UNA vez por acople
con una foto de una regla o papel milimetrado tomada con la app, y se guarda en calibracion_escala.json.

Uso dentro de la app: MENÚ → "Acople y escala".
Uso suelto:           python escala_acople.py
"""
import json
from datetime import datetime
from pathlib import Path

ARCHIVO = Path(__file__).resolve().parent / "calibracion_escala.json"

# Acoples FIJOS, calibrados con regla milimetrada el 2026-10-05 (fotos en fotos_calibracion/), imagen de 640 px:
#   8 cm    → 51,47 px/mm  (11 intervalos de 1 mm, residuo < 1 px)  · campo 12,4 × 9,3 mm · 19,4 µm/px
#   2,73 cm → 155,8 px/mm  (zoom mínimo; 4 intervalos de 1 mm en 2 fotos, ± 3 %) · campo 4,1 × 3,1 mm · 6,4 µm/px
# Cualquier otro acople se calibra en "Calibrar otro acople (libre)" y se guarda con el nombre que se le dé.
FIJOS = {
    "8 cm": {"px_por_mm": 51.47, "ancho_referencia_px": 640, "estado": "calibrada", "fecha": "2026-10-05",
             "fijo": True, "nota": "regla milimetrada, 2 fotos"},
    "2,73 cm": {"px_por_mm": 155.8, "ancho_referencia_px": 640, "estado": "calibrada", "fecha": "2026-10-05",
                "fijo": True, "nota": "regla milimetrada, 2 fotos, zoom mínimo"},
}
POR_DEFECTO = {"acople_activo": "8 cm", "acoples": json.loads(json.dumps(FIJOS))}


def leer():
    try:
        d = json.loads(ARCHIVO.read_text(encoding="utf-8"))
        if d.get("acoples"):
            for nombre, a in FIJOS.items():                 # los dos acoples fijos siempre presentes
                d["acoples"].setdefault(nombre, dict(a))
            if d.get("acople_activo") not in d["acoples"]:
                d["acople_activo"] = "8 cm"
            return d
    except Exception:
        pass
    guardar(POR_DEFECTO)
    return json.loads(json.dumps(POR_DEFECTO))


def guardar(d):
    ARCHIVO.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def escala_para(ancho_imagen_px):
    """Escala del acople activo ajustada al ancho de la imagen analizada."""
    d = leer()
    nombre = d.get("acople_activo") or next(iter(d["acoples"]))
    a = d["acoples"].get(nombre) or next(iter(d["acoples"].values()))
    ppm = float(a["px_por_mm"]) * float(ancho_imagen_px) / float(a.get("ancho_referencia_px") or ancho_imagen_px)
    return {"acople": nombre, "px_por_mm": ppm, "estado": a.get("estado", "estimada"), "fecha": a.get("fecha"),
            "ancho_px": int(ancho_imagen_px)}


def acople_activo():
    return leer().get("acople_activo", "8 cm")


def usar_acople(nombre):
    d = leer()
    if nombre not in d["acoples"]:
        raise KeyError(nombre)
    d["acople_activo"] = nombre
    guardar(d)


def texto_escala(e):
    est = "calibrada" if e.get("estado") == "calibrada" else "ESTIMADA — calibre con una regla (MENÚ → Acople y escala)"
    campo = e.get("ancho_px", 640) / e["px_por_mm"] if e.get("px_por_mm") else 0
    return f"Acople {e['acople']} · {e['px_por_mm']:.1f} px/mm · campo ≈ {campo:.1f} mm de ancho · escala {est}"


def abrir_calibrador(parent=None, al_guardar=None):
    """Ventana 'Acople y escala': elegir un acople fijo (8 cm o 2,73 cm) o uno ya calibrado, o calibrar uno nuevo
    (modo libre: foto de una regla, clic en dos marcas, distancia en mm, nombre del acople)."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk

    d = leer()
    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.title("DERMATEC — Acople y escala")
    win.geometry("1000x760")
    cont = ttk.Frame(win, padding=14)
    cont.pack(fill="both", expand=True)

    # ---------- 1) Elegir acople ----------
    ttk.Label(cont, text="1) Acople en uso", font=("Segoe UI", 13, "bold")).pack(anchor="w")
    caja = ttk.Frame(cont)
    caja.pack(fill="x", pady=(4, 6))
    var_sel = tk.StringVar(value=d.get("acople_activo", "8 cm"))

    def etiqueta(nombre, a):
        ppm = float(a["px_por_mm"]) * 640 / float(a.get("ancho_referencia_px") or 640)
        tipo = "fijo" if a.get("fijo") else ("calibrado" if a.get("estado") == "calibrada" else "estimado")
        return f"{nombre:<10}  {ppm:6.1f} px/mm   campo ≈ {640 / ppm:4.1f} × {480 / ppm:4.1f} mm   ({tipo})"

    for nombre, a in d["acoples"].items():
        ttk.Radiobutton(caja, text=etiqueta(nombre, a), value=nombre, variable=var_sel).pack(anchor="w")

    def usar():
        nombre = var_sel.get()
        d["acople_activo"] = nombre
        guardar(d)
        if al_guardar:
            al_guardar()
        win.destroy()

    ttk.Button(cont, text="USAR ESTE ACOPLE", command=usar).pack(anchor="w", pady=(0, 10))
    ttk.Separator(cont).pack(fill="x", pady=6)

    # ---------- 2) Modo libre ----------
    ttk.Label(cont, text="2) Calibrar otro acople (libre)", font=("Segoe UI", 13, "bold")).pack(anchor="w")
    ttk.Label(cont, text="Con el acople puesto y enfocado, capture una regla milimetrada. Abra la foto, haga clic en dos "
                         "marcas, escriba la distancia real entre ellas y el nombre del acople, y guarde.",
              wraplength=960).pack(anchor="w", pady=(2, 6))
    barra = ttk.Frame(cont)
    barra.pack(fill="x", pady=(0, 6))
    ttk.Label(barra, text="Nombre del acople:").pack(side="left")
    var_nombre = tk.StringVar(value="")
    ttk.Entry(barra, textvariable=var_nombre, width=12).pack(side="left", padx=(4, 12))
    ttk.Label(barra, text="Distancia entre los 2 clics (mm):").pack(side="left")
    var_mm = tk.StringVar(value="5")
    ttk.Entry(barra, textvariable=var_mm, width=6).pack(side="left", padx=(4, 12))
    lbl = ttk.Label(barra, text="Abra una imagen.")
    lbl.pack(side="left", padx=8)

    canvas = tk.Canvas(cont, bg="#152126", highlightthickness=0, height=360)
    canvas.pack(fill="both", expand=True)
    st = {"img": None, "foto": None, "f": 1.0, "pts": [], "ancho": None}

    def abrir():
        ruta = filedialog.askopenfilename(parent=win, title="Foto de la regla",
                                          filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if not ruta:
            return
        im = Image.open(ruta).convert("RGB")
        st["img"], st["ancho"], st["pts"] = im, im.width, []
        win.update_idletasks()
        cw, ch = max(canvas.winfo_width(), 400), max(canvas.winfo_height(), 300)
        st["f"] = min(cw / im.width, ch / im.height)
        st["foto"] = ImageTk.PhotoImage(im.resize((int(im.width * st["f"]), int(im.height * st["f"]))))
        canvas.delete("all")
        canvas.create_image(0, 0, image=st["foto"], anchor="nw")
        lbl.configure(text="Haga clic en la primera marca.")

    def clic(ev):
        if st["img"] is None:
            return
        if len(st["pts"]) == 2:
            st["pts"] = []
            canvas.delete("marca")
        st["pts"].append((ev.x / st["f"], ev.y / st["f"]))
        canvas.create_oval(ev.x - 5, ev.y - 5, ev.x + 5, ev.y + 5, outline="#FFD400", width=2, tags="marca")
        if len(st["pts"]) == 2:
            (x1, y1), (x2, y2) = st["pts"]
            canvas.create_line(x1 * st["f"], y1 * st["f"], x2 * st["f"], y2 * st["f"], fill="#FFD400", width=2,
                               tags="marca")
            dpx = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            try:
                ppm = dpx / float(var_mm.get().replace(",", "."))
                lbl.configure(text=f"{dpx:.1f} px → {ppm:.1f} px/mm. Pulse GUARDAR.")
            except ValueError:
                lbl.configure(text="Escriba una distancia válida en mm.")
        else:
            lbl.configure(text="Haga clic en la segunda marca.")

    def guardar_cal():
        if len(st["pts"]) != 2:
            messagebox.showwarning("DERMATEC", "Marque dos puntos sobre la regla.", parent=win)
            return
        try:
            mm = float(var_mm.get().replace(",", "."))
            assert mm > 0
        except Exception:
            messagebox.showwarning("DERMATEC", "Distancia en mm no válida.", parent=win)
            return
        nombre = var_nombre.get().strip()
        if not nombre:
            messagebox.showwarning("DERMATEC", "Escriba el nombre del acople (por ejemplo: 5 cm).", parent=win)
            return
        if nombre in FIJOS and not messagebox.askyesno(
                "DERMATEC", f"El acople {nombre} ya está calibrado (fijo). ¿Reemplazar su escala?", parent=win):
            return
        (x1, y1), (x2, y2) = st["pts"]
        ppm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / mm
        d["acoples"][nombre] = {"px_por_mm": round(ppm, 2), "ancho_referencia_px": st["ancho"], "estado": "calibrada",
                                "fecha": datetime.now().isoformat(timespec="seconds")}
        d["acople_activo"] = nombre
        guardar(d)
        messagebox.showinfo("DERMATEC", f"Escala guardada: acople {nombre} = {ppm:.2f} px/mm.\n"
                                        "Los próximos análisis usarán esta escala.", parent=win)
        if al_guardar:
            al_guardar()
        win.destroy()

    abajo = ttk.Frame(cont)
    abajo.pack(fill="x", pady=(10, 0))
    ttk.Button(abajo, text="Abrir foto de la regla…", command=abrir).pack(side="left")
    ttk.Button(abajo, text="GUARDAR ACOPLE NUEVO", command=guardar_cal).pack(side="right")
    canvas.bind("<Button-1>", clic)
    if parent is None:
        win.mainloop()


if __name__ == "__main__":
    abrir_calibrador()
