# ============================================================
# DERMATEC — ENSAMBLE FINAL Z.18 + Z.19 (sin entrenar, 1–2 min)
# Usa las probabilidades que guardó el reporte de cada corrida (probabilidades_train_val_test.npz)
# y los PAQUETE_APP de cada una. Calibra en VALIDATION (sesgo + umbral sens ≥ 0,95), reporta Train/Val/Test
# con la regla de la app, grafica matrices y arma PAQUETE_FINAL_ENSAMBLE para la App V6.
# REGLA (fijada antes de mirar Test): se usa el ensamble si su Macro-F1 de Validation supera al mejor
# modelo individual en ≥ 0,005; si no, el individual con mejor Validation (empate → Z.19, más liviano).
# ============================================================
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

CLASES = ["Nevo", "Melanoma", "BCC", "SCC"]
RAICES = [Path("/kaggle/working"), Path("/kaggle/input")]
def buscar_run(exp):
    c = sorted(p.parent for r in RAICES if r.exists() for p in r.rglob("probabilidades_train_val_test.npz") if f"/{exp}_" in str(p))
    assert c, f"❌ No encontré el reporte de {exp}. Si fue otra sesión, agrega su output como input del notebook."
    return c[-1]
RUNS = {"Z.18": buscar_run("Z.18"), "Z.19": buscar_run("Z.19")}
D = {k: np.load(v / "probabilidades_train_val_test.npz") for k, v in RUNS.items()}
for k, v in RUNS.items(): print(f"✅ {k}: {v}")
Y = {s: D["Z.18"][f"y_{s}"] for s in ["train", "val", "test"]}
assert all((D["Z.19"][f"y_{s}"] == Y[s]).all() for s in Y), "❌ Los conjuntos no están en el mismo orden"

def umbral_para_sens(p, y, smin=0.95):
    pm, yb = 1 - p[:, 0], y > 0
    for t in np.arange(0.95, 0.04, -0.005):
        if ((pm >= t) & yb).sum() / yb.sum() >= smin: return float(t)
    return 0.05
def calibrar_sesgo(p, y):
    b, lp = np.zeros(4), np.log(p + 1e-9)
    for _ in range(3):
        for c in [1, 2, 3]:
            b[c] = max((f1_score(y, (lp + np.where(np.arange(4) == c, v, b)).argmax(1), average="macro"), v)
                       for v in np.arange(-1.5, 1.51, 0.05))[1]
    return b
def evaluar(p, y, s, u):                     # regla de la app (jerárquica)
    lp = np.log(p + 1e-9) + s; pm = 1 - p[:, 0]; yb = (y > 0).astype(int); pb = (pm >= u).astype(int)
    pred = np.where(pb == 1, 1 + lp[:, 1:].argmax(1), 0)
    P_, R_, F_, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2, 3], zero_division=0)
    return {"n": int(len(y)), "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "balanced_acc": balanced_accuracy_score(y, pred), "bin_accuracy": accuracy_score(yb, pb), "bin_f1": f1_score(yb, pb),
            "bin_sens": ((pb == 1) & (yb == 1)).sum() / yb.sum(), "bin_espec": ((pb == 0) & (yb == 0)).sum() / (1 - yb).sum(),
            "bin_auc": roc_auc_score(yb, pm), **{f"F1_{c}": F_[i] for i, c in enumerate(CLASES)},
            **{f"Sens_{c}": R_[i] for i, c in enumerate(CLASES)},
            "cm": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(), "cm_bin": confusion_matrix(yb, pb, labels=[0, 1]).tolist()}

CANDIDATOS = {"Z.18": ["Z.18"], "Z.19": ["Z.19"], "Z.18 + Z.19": ["Z.18", "Z.19"]}
RES, CAL = {}, {}
for nombre, comp in CANDIDATOS.items():
    p = {s: np.mean([D[m][s] for m in comp], 0) for s in ["train", "val", "test"]}
    s_, u_ = calibrar_sesgo(p["val"], Y["val"]), umbral_para_sens(p["val"], Y["val"])
    CAL[nombre] = (comp, s_, u_, p)
    RES[nombre] = {conj: evaluar(p[k], Y[k], s_, u_) for conj, k in [("Train", "train"), ("Validation", "val"), ("Test", "test")]}

filas = [{"modelo": n, "conjunto": c, **{m: v for m, v in r.items() if not m.startswith("cm")}}
         for n, rr in RES.items() for c, r in rr.items()]
tabla = pd.DataFrame(filas)
print(tabla[["modelo", "conjunto", "accuracy", "macro_f1", "bin_sens", "bin_espec", "bin_f1", "bin_auc",
             "F1_Nevo", "F1_Melanoma", "F1_BCC", "F1_SCC"]].round(4).to_string(index=False))

# ---- decisión con Validation (regla fijada arriba) ----
val = {n: RES[n]["Validation"]["macro_f1"] for n in CANDIDATOS}
mejor_ind = max(["Z.18", "Z.19"], key=lambda n: (round(val[n], 3), n == "Z.19"))
ELEGIDO = "Z.18 + Z.19" if val["Z.18 + Z.19"] >= val[mejor_ind] + 0.005 else mejor_ind
print(f"\n🔒 Elegido por Validation: {ELEGIDO}  (Val Macro-F1: " + ", ".join(f"{n} {v:.4f}" for n, v in val.items()) + ")")

# ---- matrices del elegido ----
comp, s_, u_, p = CAL[ELEGIDO]; R = RES[ELEGIDO]
fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
for j, (k, r) in enumerate(R.items()):
    for fi, (M, et, tit) in enumerate([(np.array(r["cm"]), CLASES, f"{k} — 4 clases\nMacro-F1 {r['macro_f1']:.3f} | Acc {r['accuracy']:.3f}"),
                                       (np.array(r["cm_bin"]), ["Benigno", "Maligno"], f"{k} — binaria\nSens {r['bin_sens']:.3f} | Espec {r['bin_espec']:.3f}")]):
        a = ax[fi, j]; Mn = M / M.sum(1, keepdims=True); a.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i_ in range(len(M)):
            for j_ in range(len(M)):
                a.text(j_, i_, f"{M[i_, j_]}\n{100 * Mn[i_, j_]:.0f}%", ha="center", va="center", fontsize=9,
                       color="white" if Mn[i_, j_] > 0.5 else "black")
        a.set_xticks(range(len(M)), et, rotation=20); a.set_yticks(range(len(M)), et)
        a.set_xlabel("Predicción"); a.set_ylabel("Real"); a.set_title(tit, fontsize=10)
plt.suptitle(f"DERMATEC final ({ELEGIDO}) — umbral {u_:.3f}, sesgo de Validation"); plt.tight_layout()
SAL = Path("/kaggle/working/PAQUETE_FINAL_DERMATEC"); shutil.rmtree(SAL, ignore_errors=True); SAL.mkdir()
plt.savefig(SAL / "MATRICES_confusion_final.png", dpi=150); plt.show()

# ---- paquete para la App V6 ----
modelos = []
for m in comp:
    cfg_m = json.loads((RUNS[m] / "PAQUETE_APP" / "config_dermatec.json").read_text())
    tfl = RUNS[m] / "PAQUETE_APP" / cfg_m["modelos"][0]["archivo"]; shutil.copy(tfl, SAL / tfl.name)
    modelos.append({**cfg_m["modelos"][0], "tta": bool(cfg_m["tta"])})   # TTA por modelo, igual que en su calibración
H95 = float(np.percentile(-(p["val"] * np.log2(p["val"] + 1e-12)).sum(1), 95))
cfg = {"nombre": "DERMATEC_" + ELEGIDO.replace(" ", "").replace(".", ""), "modelos": modelos, "tta": True,
       "clases": ["Nevo", "Melanoma", "Carcinoma basocelular", "Carcinoma escamocelular"],
       "sesgo": [float(v) for v in s_], "umbral_p_maligno": float(u_), "entropia_p95_bits": H95,
       "calidad": {"modo": "advertencia", "entropia_max_bits": H95, "area_lesion_max": 0.60, "bordes_tocados_max": 1},
       "regla_salida": "BENIGNA → Nevo; MALIGNA → argmax(log p + sesgo) entre Melanoma, BCC y SCC",
       "seleccion": {"regla": "Validation; ensamble solo si supera al mejor individual en ≥ 0,005", "val_macro_f1": val},
       "metricas": {c: {m: float(v) for m, v in r.items() if not m.startswith("cm")} for c, r in R.items()}}
(SAL / "config_dermatec.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
tabla.to_csv(SAL / "METRICAS_Z18_Z19_ensamble_train_val_test.csv", index=False)
zip_ = shutil.make_archive("/kaggle/working/PAQUETE_FINAL_DERMATEC", "zip", SAL)
print(f"📦 {zip_} | modelos {[m['archivo'] for m in modelos]} | umbral {u_:.3f} | p95 entropía {H95:.2f} bits")
