import os
import csv
from parser import parse_clinvar_vcf
from utils import load_hpoa_info

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use whichever filename you keep for your VCF; adjust if you use cClinvar.vcf
CLINVAR_VCF = os.path.join(BASE_DIR, "data", "raw", "clinvar.vcf")
HPOA_FILE = os.path.join(BASE_DIR, "data", "raw", "phenotype.hpoa")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "clinvar_hpoa_merged.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# canonical normalized uncertain tag
UNCERTAIN_TAGS = {"uncertain_significance", "uncertain_significance", "uncertain-significance", "uncertain"}

def normalize_clnsig(raw):
    if raw is None:
        return ""
    s = str(raw).strip()
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()

def extract_first_name_from_clndn(clndn_raw):
    """
    clndn_raw can be string or list; split on |,;,,
    return first non-empty token or ''.
    """
    if not clndn_raw:
        return ""
    if isinstance(clndn_raw, list):
        # join then split
        s = "|".join(str(x) for x in clndn_raw if x)
    else:
        s = str(clndn_raw)
    for sep in ["|", ";", ","]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            if parts:
                return parts[0]
    return s.strip()

def build_merged_csv():
    print("🔹 Parsing ClinVar VCF...")
    clinvar_records = parse_clinvar_vcf(CLINVAR_VCF)

    print("🔹 Loading phenotype.hpoa info...")
    disease_hpo_map, disease_name_map = load_hpoa_info(HPOA_FILE)

    # Step 1: detect diseases that have ANY uncertain_significance evidence in ClinVar
    diseases_with_uncertain = set()
    for rec in clinvar_records:
        clnsig_raw = rec.get("clnsig", "") or rec.get("clnsig_raw", "")
        clnsig_norm = normalize_clnsig(clnsig_raw)
        if clnsig_norm in UNCERTAIN_TAGS or "uncertain" in clnsig_norm:
            diseases = rec.get("omim_ids", []) + rec.get("orphanet_ids", [])
            for d in diseases:
                diseases_with_uncertain.add(d)

    print(f"🔹 Detected {len(diseases_with_uncertain)} diseases with uncertain_significance evidence (they will be excluded).")

    rows = []
    for rec in clinvar_records:
        chrom = rec.get("chrom", "")
        pos = rec.get("pos", "")
        ref = rec.get("ref", "")
        alt = rec.get("alt", "")
        gene = rec.get("gene", "")
        clnsig_raw = rec.get("clnsig", "") or rec.get("clnsig_raw", "")
        clnsig_norm = normalize_clnsig(clnsig_raw)
        weight = rec.get("weight", 0.0)

        diseases = rec.get("omim_ids", []) + rec.get("orphanet_ids", [])
        # CLNDN fallback (string possibly containing multiple names)
        clndn_raw = rec.get("disease_names", "") or rec.get("clndn_list", "") or rec.get("CLNDN", "")

        for disease in diseases:
            # skip whole disease if any uncertain evidence exists for it
            if disease in diseases_with_uncertain:
                continue

            # resolve disease_name: prefer HPOA canonical name, else CLNDN fallback, else disease id
            disease_name = disease_name_map.get(disease, "")
            if not disease_name:
                # attempt to use CLNDN from record
                dn = extract_first_name_from_clndn(clndn_raw)
                disease_name = dn if dn else disease

            # gather HPOs: clinvar-record HPOs and those in HPOA
            clinvar_hpos = set(rec.get("hpo_ids", []))
            hpoa_hpos = disease_hpo_map.get(disease, set())

            # write ClinVar HPOs (source = ClinVar)
            for hpo in clinvar_hpos:
                if not hpo:
                    continue
                rows.append({
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene": gene,
                    "clnsig": clnsig_raw,
                    "variant_type": rec.get("variant_type", ""),
                    "disease_id": disease,
                    "disease_name": disease_name,
                    "disease_source": "ClinVar",
                    "hpo_id": hpo,
                    "hpo_source": "ClinVar",
                    "clinvar_weight": weight
                })

            # write HPOA HPOs (source = HPOA)
            for hpo in hpoa_hpos:
                if not hpo:
                    continue
                rows.append({
                    "chrom": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene": gene,
                    "clnsig": clnsig_raw,
                    "variant_type": rec.get("variant_type", ""),
                    "disease_id": disease,
                    "disease_name": disease_name,
                    "disease_source": "HPOA",
                    "hpo_id": hpo,
                    "hpo_source": "HPOA",
                    "clinvar_weight": weight
                })

    print(f"🔹 Writing {len(rows)} rows to CSV...")

    fieldnames = [
        "chrom", "pos", "ref", "alt", "gene",
        "clnsig", "variant_type",
        "disease_id", "disease_name", "disease_source",
        "hpo_id", "hpo_source", "clinvar_weight"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("✅ CSV created at:")
    print(OUTPUT_CSV)
    if diseases_with_uncertain:
        sample = list(diseases_with_uncertain)[:20]
        print("⚠️ Excluded diseases sample (have uncertain evidence):", sample)

if __name__ == "__main__":
    build_merged_csv()
