"""
DERMATEC — Verificar el conjunto DERMATEC_KAGGLE_V2 (integridad, conteos y fugas entre particiones)
===================================================================================================
Uso:
    python verificar_dataset.py --manifest manifest_DERMATEC_KAGGLE_V2.csv
    python verificar_dataset.py --manifest manifest_DERMATEC_KAGGLE_V2.csv --tar DERMATEC_KAGGLE_V2.tar --sha256
Comprueba:
  1) conteos por partición y clase (esperado 12 994 / 2 156 / 2 146; 7 000 / 4 500 / 4 500 / 1 296)
  2) cero fugas entre particiones por imagen, sha256, grupo, paciente, lesión e ID ISIC (las columnas que existan)
  3) opcional: tamaño del TAR (6 748 149 760 bytes) y su SHA-256 (d09835a1…f5e248f; tarda varios minutos)
"""
import argparse, hashlib, sys
from pathlib import Path
import pandas as pd

ESPERADO_SPLIT = {"train": 12994, "validation": 2156, "test": 2146}
ESPERADO_CLASE = {0: 7000, 1: 4500, 2: 4500, 3: 1296}
TAR_BYTES = 6_748_149_760
TAR_SHA = "d09835a1a51fb7f7bf0159fccc874f13872abbced8d5f9934de0986f6f5e248f"

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tar")
ap.add_argument("--sha256", action="store_true")
a = ap.parse_args()

df = pd.read_csv(a.manifest)
low = {c.lower(): c for c in df.columns}
col_split = next((low[k] for k in ["split", "partition", "particion", "split_final"] if k in low), None)
col_y = next((low[k] for k in ["indice_clase_4", "label", "label_4", "clase_idx", "class_idx", "target"] if k in low), None)
if col_split is None or col_y is None:
    sys.exit(f"No encuentro las columnas de partición/clase. Columnas: {list(df.columns)}")
df[col_split] = df[col_split].astype(str).str.lower().str.strip().replace({"val": "validation", "valid": "validation"})

ok = True
print("Conteos por partición y clase:")
tabla = pd.crosstab(df[col_split], df[col_y], margins=True)
print(tabla.to_string())
for s, n in ESPERADO_SPLIT.items():
    real = int((df[col_split] == s).sum())
    if real != n:
        ok = False; print(f"  ❌ {s}: {real} (esperado {n})")
for c, n in ESPERADO_CLASE.items():
    real = int((df[col_y].astype(int) == c).sum())
    if real != n:
        ok = False; print(f"  ❌ clase {c}: {real} (esperado {n})")

print("\nFugas entre particiones:")
cols = [c for c in ["image_relpath", "image_id", "sha256", "group_id_split", "patient_scoped", "lesion_scoped",
                    "patient_id", "lesion_id", "isic_id"] if c in df.columns]
for c in cols:
    conj = {s: set(df.loc[(df[col_split] == s) & df[c].notna(), c].astype(str)) for s in ESPERADO_SPLIT}
    for s1, s2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        n = len(conj[s1] & conj[s2])
        if n:
            ok = False
        print(f"  {c:16s} {s1:10s} ∩ {s2:10s}: {n}{'  ❌' if n else ''}")

if a.tar:
    t = Path(a.tar); tam = t.stat().st_size
    print(f"\nTAR: {tam:,} bytes {'✅' if tam == TAR_BYTES else '❌ (esperado ' + format(TAR_BYTES, ',') + ')'}")
    ok &= tam == TAR_BYTES
    if a.sha256:
        h = hashlib.sha256()
        with open(t, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 24), b""):
                h.update(bloque)
        print("SHA-256:", h.hexdigest(), "✅" if h.hexdigest() == TAR_SHA else "❌")
        ok &= h.hexdigest() == TAR_SHA
print("\nRESULTADO:", "✅ conjunto íntegro y sin fugas" if ok else "❌ revisar lo marcado")
