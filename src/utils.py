import pandas as pd
import numpy as np
from collections import defaultdict

# -----------------------------
# Clinical significance weights
# -----------------------------
CLNSIG_WEIGHTS = {
    "Pathogenic": 1.0,
    "Likely_pathogenic": 0.8,
    "Uncertain_significance": 0.3,
    "Likely_benign": 0.1,
    "Benign": 0.0
}

def get_clnsig_weight(clnsig: str) -> float:
    """
    Convert ClinVar clinical significance to numeric weight.
    Accepts raw strings and uses CLNSIG_WEIGHTS keys as canonical forms.
    """
    if clnsig is None:
        return 0.0
    raw = str(clnsig).strip()
    if raw == "":
        return 0.0
    # Try direct first
    if raw in CLNSIG_WEIGHTS:
        return CLNSIG_WEIGHTS[raw]
    # Normalize various spellings
    key = raw.replace(" ", "_").replace("-", "_").lower()
    for k in CLNSIG_WEIGHTS:
        if k.lower() == key:
            return CLNSIG_WEIGHTS[k]
    # fallback
    return 0.0

# -----------------------------
# Load phenotype.hpoa: return both HPO map and canonical disease name map
# -----------------------------
def load_hpoa_info(hpoa_path):
    """
    Load phenotype.hpoa and return:
      - disease_hpo_map: { disease_id -> set(hpo_id) }
      - disease_name_map: { disease_id -> disease_name }  (canonical name from file)
    The hpoa file typically has columns like:
    database_id    disease_name    qualifier    hpo_id    reference ...
    """
    # tolerant loading: if header exists use it, otherwise attempt defaults
    try:
        df = pd.read_csv(hpoa_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    except Exception as e:
        # return empty maps if file missing or unreadable
        print(f"Warning: could not read hpoa file at {hpoa_path}: {e}")
        return defaultdict(set), {}

    disease_hpo_map = defaultdict(set)
    disease_name_map = {}

    # Normalize column names if they differ slightly
    cols = {c.lower(): c for c in df.columns}
    # Expect 'database_id' and 'disease_name' and 'hpo_id' columns; fallback to first three if not present
    db_col = cols.get("database_id", None) or (df.columns[0] if len(df.columns) > 0 else None)
    name_col = cols.get("disease_name", None) or (df.columns[1] if len(df.columns) > 1 else None)
    hpo_col = cols.get("hpo_id", None) or (df.columns[3] if len(df.columns) > 3 else None)

    if not db_col or not hpo_col:
        # file unexpected format
        print("Warning: phenotype.hpoa does not contain expected columns 'database_id' and 'hpo_id'.")
        return defaultdict(set), {}

    for _, row in df.iterrows():
        disease_id = str(row.get(db_col, "")).strip()
        if disease_id == "":
            continue
        hpo_id = str(row.get(hpo_col, "")).strip()
        if hpo_id != "":
            disease_hpo_map[disease_id].add(hpo_id)

        if name_col:
            disease_name = str(row.get(name_col, "")).strip()
            if disease_name:
                disease_name_map[disease_id] = disease_name

    return disease_hpo_map, disease_name_map
