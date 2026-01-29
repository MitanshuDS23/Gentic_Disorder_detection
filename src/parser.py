import vcfpy
from collections import defaultdict
from utils import get_clnsig_weight

# -----------------------------
# Parse ClinVar VCF (PURE PYTHON)
# -----------------------------
def parse_clinvar_vcf(vcf_path):
    reader = vcfpy.Reader.from_path(vcf_path)
    records = []

    for rec in reader:
        info = rec.INFO

        clnsig = info.get("CLNSIG", ["Uncertain_significance"])
        if isinstance(clnsig, list):
            clnsig = clnsig[0]

        weight = get_clnsig_weight(clnsig)

        clndisdb = info.get("CLNDISDB", [])
        orphanet_ids, omim_ids, hpo_ids = [], [], []

        if isinstance(clndisdb, str):
            clndisdb = clndisdb.split(",")

        for entry in clndisdb:
            if entry.startswith("Orphanet:"):
                orphanet_ids.append(entry)
            elif entry.startswith("OMIM:"):
                omim_ids.append(entry)
            elif entry.startswith("Human_Phenotype_Ontology:"):
                hpo_ids.append(
                    entry.replace("Human_Phenotype_Ontology:", "")
                )

        records.append({
            "chrom": rec.CHROM,
            "pos": rec.POS,
            "ref": rec.REF,
            "alt": ",".join([alt.value for alt in rec.ALT]),
            "gene": info.get("GENEINFO", ""),
            "clnsig": clnsig,
            "weight": weight,
            "orphanet_ids": orphanet_ids,
            "omim_ids": omim_ids,
            "hpo_ids": hpo_ids
        })

    return records

# -----------------------------
# Build disease evidence
# -----------------------------
def build_disease_evidence(clinvar_records):
    disease_map = defaultdict(lambda: {"hpos": set(), "score": 0.0})

    for rec in clinvar_records:
        diseases = rec["orphanet_ids"] + rec["omim_ids"]
        for disease in diseases:
            disease_map[disease]["hpos"].update(rec["hpo_ids"])
            disease_map[disease]["score"] += rec["weight"]

    return disease_map
