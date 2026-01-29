import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import multiprocessing

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "clinvar_hpoa_merged.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "variant_predictor.pt")
GENE_ENCODER_PATH = os.path.join(MODEL_DIR, "gene_encoder.pt")
DISEASE_ENCODER_PATH = os.path.join(MODEL_DIR, "disease_encoder.pt")
DISEASE_NAME_MAP_PATH = os.path.join(MODEL_DIR, "disease_id_to_name.json")
DISEASE_GENE_MAP_PATH = os.path.join(MODEL_DIR, "disease_id_to_genes.json")

STAT_DIR = os.path.join(BASE_DIR, "Statistics", "diagrams", "training")
os.makedirs(STAT_DIR, exist_ok=True)

RF_STAT_DIR = os.path.join(BASE_DIR, "Statistics", "diagrams", "rf")
os.makedirs(RF_STAT_DIR, exist_ok=True)

RF_MODEL_PATH = os.path.join(MODEL_DIR, "variant_rf.pkl")

BATCH_SIZE = 1024
EPOCHS = 35
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("Initial rows:", len(df))

required_cols = [
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "disease_id",
    "disease_name"
]

df = df.dropna(subset=required_cols)

print("Rows after cleaning:", len(df))

counts = df["disease_id"].value_counts()
valid_diseases = counts[counts >= 2].index
df = df[df["disease_id"].isin(valid_diseases)]

print("Rows after rare-disease filtering:", len(df))
print("Remaining diseases:", df["disease_id"].nunique())

print("Building encoders...")

gene_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

df["gene_enc"] = gene_encoder.fit_transform(df["gene"].astype(str))
df["disease_enc"] = disease_encoder.fit_transform(df["disease_id"].astype(str))

torch.save(gene_encoder, GENE_ENCODER_PATH)
torch.save(disease_encoder, DISEASE_ENCODER_PATH)

NUM_CLASSES = len(disease_encoder.classes_)

print("Diseases:", NUM_CLASSES)

disease_map = (
    df[["disease_id", "disease_name"]]
    .drop_duplicates()
    .set_index("disease_id")["disease_name"]
    .to_dict()
)

with open(DISEASE_NAME_MAP_PATH, "w") as f:
    json.dump(disease_map, f, indent=2)

disease_gene_map = (
    df.groupby("disease_id")["gene"]
    .apply(lambda x: sorted(set(x.astype(str))))
    .to_dict()
)

with open(DISEASE_GENE_MAP_PATH, "w") as f:
    json.dump(disease_gene_map, f, indent=2)

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

INPUT_DIM = X.shape[1]
print("Input dim:", INPUT_DIM)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.15,
    stratify=y,
    random_state=42
)

class VariantDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    VariantDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    VariantDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    num_workers=0
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)

weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

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

model = AdvancedDiseaseNet(INPUT_DIM, NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

history = {
    "epoch": [],
    "train_loss": [],
    "val_accuracy": [],
    "val_precision_macro": [],
    "val_recall_macro": [],
    "val_f1_macro": [],
}

def main():

    print("Starting training...")

    best_val = 0

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()

        correct = 0
        total = 0

        all_preds = []
        all_true = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                preds = model(xb).argmax(dim=1)

                correct += (preds == yb).sum().item()
                total += len(yb)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(yb.cpu().numpy())

        acc = correct / total

        precision = precision_score(all_true, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_true, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss {total_loss:.4f} | "
            f"ValAcc {acc:.4f}"
        )

        if acc > best_val:
            best_val = acc
            torch.save(model.state_dict(), MODEL_PATH)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(acc)
        history["val_precision_macro"].append(precision)
        history["val_recall_macro"].append(recall)
        history["val_f1_macro"].append(f1)

    print("Training complete.")
    print("Saved model to:", MODEL_PATH)

    print("Starting Random Forest baseline training")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    rf.fit(X_train, y_train)

    rf_preds = rf.predict(X_val)

    rf_acc = accuracy_score(y_val, rf_preds)

    report = classification_report(y_val, rf_preds)

    print("RF Accuracy:", rf_acc)
    print(report)

    with open(os.path.join(RF_STAT_DIR, "rf_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_val, rf_preds)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm)
    plt.title("Random Forest Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(RF_STAT_DIR, "rf_confusion_matrix.png"))
    plt.close()

    with open(RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)

    print("Saved RF model to:", RF_MODEL_PATH)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
