import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clinvar_hpoa_merged.csv",
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "variant_predictor.pt")
GENE_ENCODER_PATH = os.path.join(MODEL_DIR, "gene_encoder.pt")
DISEASE_ENCODER_PATH = os.path.join(MODEL_DIR, "disease_encoder.pt")

FINAL_DIR = os.path.join(
    BASE_DIR,
    "Statistics",
    "diagrams",
    "final_evaluation",
)

os.makedirs(FINAL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH, low_memory=False)

required_cols = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "disease_id",
    "disease_name",
]

df = df.dropna(subset=required_cols)

counts = df["disease_id"].value_counts()
valid_diseases = counts[counts >= 2].index
df = df[df["disease_id"].isin(valid_diseases)]

# ============================================================
# LOAD ENCODERS
# ============================================================

gene_encoder = torch.load(GENE_ENCODER_PATH)
disease_encoder = torch.load(DISEASE_ENCODER_PATH)

df["gene_enc"] = gene_encoder.transform(df["gene"].astype(str))
df["disease_enc"] = disease_encoder.transform(df["disease_id"].astype(str))

NUM_CLASSES = len(disease_encoder.classes_)

# ============================================================
# FEATURE ENGINEERING
# ============================================================

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


if "gt" not in df.columns:
    df["gt"] = "0/1"

if "dp" not in df.columns:
    df["dp"] = 30

if "gq" not in df.columns:
    df["gq"] = 60


X = np.column_stack([
    df["chrom"].apply(encode_chrom),
    df["pos"].values,
    df["ref"].apply(encode_base),
    df["alt"].apply(encode_base),
    df["gt"].apply(encode_gt),
    df["dp"].values,
    df["gq"].values,
    df["gene_enc"].values,
])

y = df["disease_enc"].values

# ============================================================
# SAME SPLIT
# ============================================================

_, X_val, _, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42,
)

# ============================================================
# MODEL
# ============================================================

import torch.nn as nn


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

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model = AdvancedDiseaseNet(X.shape[1], NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================================================
# INFERENCE
# ============================================================

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    logits = model(X_val_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_val, preds)

macro_precision = precision_score(
    y_val, preds, average="macro", zero_division=0
)

macro_recall = recall_score(
    y_val, preds, average="macro", zero_division=0
)

macro_f1 = f1_score(
    y_val, preds, average="macro", zero_division=0
)

weighted_f1 = f1_score(
    y_val, preds, average="weighted", zero_division=0
)

metrics = {
    "accuracy": accuracy,
    "macro_precision": macro_precision,
    "macro_recall": macro_recall,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
}

with open(os.path.join(FINAL_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# ============================================================
# HEATMAP 1 — RAW CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_val, preds)

plt.figure(figsize=(14, 12))
plt.imshow(cm, cmap="hot")
plt.colorbar(label="Sample Count")
plt.title("Confusion Matrix Heatmap (Raw Counts)")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "confusion_matrix_heatmap_raw.png"))
plt.close()

# ============================================================
# HEATMAP 2 — NORMALIZED
# ============================================================

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(14, 12))
plt.imshow(cm_norm, cmap="hot")
plt.colorbar(label="Row Normalized Value")
plt.title("Normalized Confusion Matrix Heatmap")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                 ha="center", va="center", fontsize=6)

plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "confusion_matrix_heatmap_normalized.png"))
plt.close()

# ============================================================
# HISTOGRAM — CONFIDENCE
# ============================================================

confidence = probs.max(axis=1)

plt.figure(figsize=(10, 6))
plt.hist(confidence, bins=40)
plt.title("Prediction Confidence Histogram")
plt.xlabel("Maximum Softmax Probability")
plt.ylabel("Sample Count")
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "prediction_confidence_histogram.png"))
plt.close()

# ============================================================
# HISTOGRAM — PER CLASS F1
# ============================================================

f1_vals = f1_score(y_val, preds, average=None)

plt.figure(figsize=(12, 6))
plt.hist(f1_vals, bins=30)
plt.title("Per-Class F1 Score Histogram")
plt.xlabel("F1 Score")
plt.ylabel("Number of Disease Classes")
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "per_class_f1_histogram.png"))
plt.close()

# ============================================================
# HISTOGRAM — CLASS DISTRIBUTION
# ============================================================

dist = pd.Series(y_val).value_counts().sort_index()

plt.figure(figsize=(14, 6))
plt.bar(range(len(dist)), dist.values)
plt.title("Validation Class Distribution")
plt.xlabel("Disease Class Index")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "validation_class_distribution.png"))
plt.close()

# ============================================================
# FINAL COMPOSITE DASHBOARD
# ============================================================

fig, axs = plt.subplots(2, 3, figsize=(22, 14))

# Confusion Matrix
axs[0, 0].imshow(cm, cmap="hot")
axs[0, 0].set_title("Confusion Matrix (Raw)")
axs[0, 0].set_xlabel("Predicted")
axs[0, 0].set_ylabel("True")

# Normalized CM
axs[0, 1].imshow(cm_norm, cmap="hot")
axs[0, 1].set_title("Confusion Matrix (Normalized)")
axs[0, 1].set_xlabel("Predicted")
axs[0, 1].set_ylabel("True")

# Confidence Hist
axs[0, 2].hist(confidence, bins=40)
axs[0, 2].set_title("Prediction Confidence")

# Class Dist
axs[1, 0].bar(range(len(dist)), dist.values)
axs[1, 0].set_title("Class Distribution")

# F1 Hist
axs[1, 1].hist(f1_vals, bins=30)
axs[1, 1].set_title("Per-Class F1")

# Metric Text Panel
metric_text = "\n".join([
    f"Accuracy: {accuracy:.4f}",
    f"Macro Precision: {macro_precision:.4f}",
    f"Macro Recall: {macro_recall:.4f}",
    f"Macro F1: {macro_f1:.4f}",
    f"Weighted F1: {weighted_f1:.4f}",
])

axs[1, 2].axis("off")
axs[1, 2].text(0.05, 0.95, metric_text, fontsize=14, va="top")

plt.suptitle("Final Evaluation Dashboard — Disease Prediction Model", fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "final_evaluation_dashboard.png"))
plt.close()
