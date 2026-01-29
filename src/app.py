from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import os
import pandas as pd
import json
import re

# ================================
# CONFIG
# ================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "variant_predictor.pt")
GENE_ENCODER_PATH = os.path.join(MODEL_DIR, "gene_encoder.pt")
DISEASE_ENCODER_PATH = os.path.join(MODEL_DIR, "disease_encoder.pt")
DISEASE_GENE_MAP_PATH = os.path.join(MODEL_DIR, "disease_id_to_genes.json")

CURATED_CSV = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clinvar_hpoa_merged.csv"
)

CONFIDENCE_THRESHOLD = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

app = Flask(__name__)
CORS(app)

# ================================
# LOAD CSV
# ================================

curated_df = pd.read_csv(CURATED_CSV, low_memory=False)

# ================================
# LOAD GENE MAP
# ================================

with open(DISEASE_GENE_MAP_PATH) as f:
    disease_gene_map = json.load(f)

# ================================
# LOAD ENCODERS
# ================================

print("Loading encoders...")

gene_encoder = torch.load(GENE_ENCODER_PATH, map_location="cpu")
disease_encoder = torch.load(DISEASE_ENCODER_PATH, map_location="cpu")

NUM_CLASSES = len(disease_encoder.classes_)

# ================================
# MODEL DEFINITION (MATCH TRAIN)
# ================================

INPUT_DIM = 8   # <<< UPDATED


class AdvancedDiseaseNet(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# LOAD MODEL
# ================================

print("Loading trained model...")

model = AdvancedDiseaseNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)

model.eval()

print("Model ready.")

# ================================
# FEATURE HELPERS
# ================================

BASE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_base(b):
    return BASE_MAP.get(str(b).upper(), 4)


def encode_gt(gt):
    return {"0/0": 0, "0/1": 1, "1/1": 2}.get(gt, 1)


def encode_chrom(ch):

    ch = str(ch).replace("chr", "")

    if ch.isdigit():
        return int(ch)

    if ch == "X":
        return 23
    if ch == "Y":
        return 24

    return 0


def extract_gene_from_info(info):

    # Try ANN=...|GENE| format (VEP/SnpEff)
    match = re.search(r"ANN=[^|]*\|[^|]*\|[^|]*\|([^|]+)", info)
    if match:
        return match.group(1)

    # Try GENE=BRCA1
    match = re.search(r"GENE=([^;]+)", info)
    if match:
        return match.group(1)

    return None


# ================================
# VCF PARSER
# ================================

def parse_vcf_simple(path):

    records = []

    with open(path) as f:

        for line in f:

            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")

            if len(parts) < 10:
                continue

            chrom, pos, _, ref, alt, qual, filt, info, fmt, sample = parts[:10]

            keys = fmt.split(":")
            vals = sample.split(":")

            fmt_dict = dict(zip(keys, vals))

            gene = extract_gene_from_info(info)

            records.append({
                "chrom": chrom,
                "pos": int(pos),
                "ref": ref,
                "alt": alt,
                "gt": fmt_dict.get("GT", "0/1"),
                "dp": int(fmt_dict.get("DP", 30)),
                "gq": int(fmt_dict.get("GQ", 60)),
                "gene": gene,
            })

    return records


# ================================
# FEATURE VECTOR
# ================================

def build_feature_vector(rec):

    if rec["gene"] and rec["gene"] in gene_encoder.classes_:
        gene_enc = int(gene_encoder.transform([rec["gene"]])[0])
    else:
        gene_enc = 0

    return torch.tensor([
        encode_chrom(rec["chrom"]),
        float(rec["pos"]),
        encode_base(rec["ref"]),
        encode_base(rec["alt"]),
        encode_gt(rec["gt"]),
        float(rec["dp"]),
        float(rec["gq"]),
        float(gene_enc),
    ], dtype=torch.float32)


# ================================
# API ROUTE
# ================================

@app.route("/predict_vcf", methods=["POST"])
def predict_vcf():

    if "file" not in request.files:
        return jsonify({"error": "No VCF uploaded"}), 400

    file = request.files["file"]

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    records = parse_vcf_simple(save_path)

    if not records:
        return jsonify({"error": "No variants found"}), 400

    batch = torch.stack(
        [build_feature_vector(r) for r in records]
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    mean_probs = probs.mean(dim=0)

    best_idx = int(torch.argmax(mean_probs))
    best_score = float(mean_probs[best_idx])

    print("Best probability:", best_score)

    # ================================
    # HEALTHY DECISION
    # ================================

    if best_score <= CONFIDENCE_THRESHOLD:

        return jsonify({
            "status": "HEALTHY",
            "probability of disease": f"{round(best_score * 1000, 6)}%",
            "variants_processed": len(records),
            "affected_genes": []
        })

    # ================================
    # TOP DISEASES + GENES
    # ================================

    ranked = sorted(
        enumerate(mean_probs.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    summary = []

    for idx, score in ranked[:5]:

        disease_id = disease_encoder.inverse_transform([idx])[0]

        disease_name = (
            curated_df[curated_df["disease_id"] == disease_id]["disease_name"]
            .iloc[0]
        )

        genes = disease_gene_map.get(disease_id, [])

        summary.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "affected_genes": genes,
            "probability": f"{round(score * 1000, 6)}%"
        })

    return jsonify({
        "status": "DISEASE_DETECTED",
        "variants_processed": len(records),
        "top_predictions": summary
    })


# ================================
# MAIN
# ================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
