"""
DERMATEC — Escala del acople (píxeles por milímetro) y calibrador
================================================================
La escala depende del acople (distancia lente–piel) y del zoom con el que se enfoca. Se calibra UNA vez por acople
con una foto de una regla o papel milimetrado tomada con la app, y se guarda en calibracion_escala.json.

Uso dentro de la app: MENÚ → "Calibrar escala (mm)".
Uso suelto:           python escala_acople.py
"""
import json
from datetime import datetime
from pathlib import Path

ARCHIVO = Path(__file__).resolve().parent / "calibracion_escala.json"

# Estimaciones iniciales a partir de la calibración del campo de visión del capilógrafo con regla (tubos de 27,3;
# 58,6; 85,6 y 107,5 mm): campo horizontal ≈ 0,155 × L (R² 0,998) → f·k_u ≈ 4 140 px con imagen de 640 px
# → escala ≈ 4 140 / L(mm) px/mm (acople de 8 cm: campo ≈ 12,4 mm, ≈ 51,7 px/mm).
# Son SOLO de partida: el zoom con que se enfoca cambia la escala → calibrar con regla (MENÚ → Calibrar escala).
POR_DEFECTO = {
    "acople_activo": "8 cm",
    "acoples": {
        nombre: {"px_por_mm": round(4140 / (10 * cm), 1), "ancho_referencia_px": 640, "estado": "estimada", "fecha": None}
        for nombre, cm in (("8 cm", 8), ("2 cm", 2), ("6 cm", 6), ("9 cm", 9))
    },
}


def leer():
    try:
        d = json.loads(ARCHIVO.read_text(encoding="utf-8"))
        if d.get("acoples"):
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
    return {"acople": nombre, "px_por_mm": ppm, "estado": a.get("estado", "estimada"), "fecha": a.get("fecha")}


def texto_escala(e):
    est = "calibrada" if e.get("estado") == "calibrada" else "ESTIMADA — calibre con una regla (MENÚ → Calibrar escala)"
    return f"Acople {e['acople']} · {e['px_por_mm']:.1f} px/mm · escala {est}"


def abrir_calibrador(parent=None, al_guardar=None):
    """Ventana: abrir foto de una regla, hacer clic en dos marcas, escribir la distancia en mm y guardar."""
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk

    d = leer()
    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.title("DERMATEC — Calibrar escala del acople")
    win.geometry("980x700")
    cont = ttk.Frame(win, padding=14)
    cont.pack(fill="both", expand=True)
    ttk.Label(cont, text="Calibrar escala (mm)", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    ttk.Label(cont, text="1) Con el acople puesto y enfocado, capture una regla o papel milimetrado.  "
                         "2) Abra esa foto.  3) Haga clic en dos marcas de la regla.  4) Escriba la distancia real y guarde.",
              wraplength=940).pack(anchor="w", pady=(2, 8))

    barra = ttk.Frame(cont)
    barra.pack(fill="x", pady=(0, 8))
    ttk.Label(barra, text="Acople:").pack(side="left")
    var_acople = tk.StringVar(value=d.get("acople_activo", "8 cm"))
    combo = ttk.Combobox(barra, textvariable=var_acople, values=list(d["acoples"]), width=10)
    combo.pack(side="left", padx=(4, 12))
    ttk.Label(barra, text="Distancia entre los 2 clics (mm):").pack(side="left")
    var_mm = tk.StringVar(value="10")
    ttk.Entry(barra, textvariable=var_mm, width=6).pack(side="left", padx=(4, 12))
    lbl = ttk.Label(barra, text="Abra una imagen.")
    lbl.pack(side="left", padx=8)

    canvas = tk.Canvas(cont, bg="#152126", highlightthickness=0)
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
        vista = im.resize((int(im.width * st["f"]), int(im.height * st["f"])))
        st["foto"] = ImageTk.PhotoImage(vista)
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
            canvas.create_line(x1 * st["f"], y1 * st["f"], x2 * st["f"], y2 * st["f"], fill="#FFD400", width=2, tags="marca")
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
        (x1, y1), (x2, y2) = st["pts"]
        ppm = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / mm
        nombre = var_acople.get().strip() or "acople"
        d["acoples"][nombre] = {"px_por_mm": round(ppm, 2), "ancho_referencia_px": st["ancho"], "estado": "calibrada",
                                "fecha": datetime.now().isoformat(timespec="seconds")}
        d["acople_activo"] = nombre
        guardar(d)
        messagebox.showinfo("DERMATEC", f"Escala guardada: acople {nombre} = {ppm:.2f} px/mm.\n"
                                        "Los próximos análisis usarán esta escala.", parent=win)
        if al_guardar:
            al_guardar()
        win.destroy()

    def solo_activar():
        nombre = var_acople.get().strip()
        if nombre not in d["acoples"]:
            messagebox.showwarning("DERMATEC", "Ese acople no tiene escala; calíbrelo primero.", parent=win)
            return
        d["acople_activo"] = nombre
        guardar(d)
        if al_guardar:
            al_guardar()
        win.destroy()

    abajo = ttk.Frame(cont)
    abajo.pack(fill="x", pady=(10, 0))
    ttk.Button(abajo, text="Abrir foto de la regla…", command=abrir).pack(side="left")
    ttk.Button(abajo, text="Solo usar este acople", command=solo_activar).pack(side="left", padx=8)
    ttk.Button(abajo, text="GUARDAR CALIBRACIÓN", command=guardar_cal).pack(side="right")
    canvas.bind("<Button-1>", clic)
    if parent is None:
        win.mainloop()


if __name__ == "__main__":
    abrir_calibrador()
