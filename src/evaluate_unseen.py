"""
evaluate_unseen.py
==================
Train on the full main dataset and evaluate on UNSEEN_DATA.
Supports both Random Forest (MFCC) and CNN (Mel spectrogram).

Usage:
  # RF only
  python evaluate_unseen.py \
      --train_meta    data/sample_meta_stratifie5.csv \
      --train_aircraft data/aircraft_meta_new.csv \
      --train_root    data \
      --unseen_meta   UNSEEN_DATA/sample_meta.csv \
      --unseen_aircraft UNSEEN_DATA/aircraft_meta.csv \
      --unseen_root   UNSEEN_DATA \
      --output_dir    results_unseen \
      --rf_only

  # Both RF + CNN
  python evaluate_unseen.py \
      --train_meta    data/sample_meta_stratifie5.csv \
      --train_aircraft data/aircraft_meta_new.csv \
      --train_root    data \
      --unseen_meta   UNSEEN_DATA/sample_meta.csv \
      --unseen_aircraft UNSEEN_DATA/aircraft_meta.csv \
      --unseen_root   UNSEEN_DATA \
      --output_dir    results_unseen
"""

import argparse
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Constants (must match train_and_evaluate.py) ──────────────────────────────
SR         = 16000
DURATION   = 10
N_MELS     = 64
N_FFT      = 1024
HOP_LENGTH = 512
N_MFCC     = 40
SUBCLASSES = ["Piston", "Turboprop", "Turbofan", "Turboshaft"]
ALL_LABELS = ["background"] + SUBCLASSES


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def build_meta(sample_path: str, aircraft_path: str, audio_root: str,
               is_unseen: bool = False) -> pd.DataFrame:
    sample   = pd.read_csv(sample_path)
    aircraft = pd.read_csv(aircraft_path)[["hex_id", "Engtype"]]
    merged   = sample.merge(aircraft, on="hex_id", how="left")

    # Fill class from Engtype if missing (unseen data)
    if merged["class"].isna().any():
        merged["class"] = merged["class"].fillna(
            merged["Engtype"].apply(lambda x: 1 if pd.notna(x) else np.nan)
        )

    merged.loc[merged["class"] == 0, "Engtype"] = "background"
    merged["Engtype"] = merged["Engtype"].replace("Diesel Engine", "Piston")

    # For unseen data: if Engtype still missing for class==1, fill from aircraft meta
    if is_unseen:
        # All unseen are aircraft (class=1); fill any missing Engtype with Turbofan as fallback
        merged.loc[(merged["class"] == 1) & merged["Engtype"].isna(), "Engtype"] = "Turbofan"

    before  = len(merged)
    merged  = merged[~((merged["class"] == 1) & merged["Engtype"].isna())]
    dropped = before - len(merged)
    if dropped:
        print(f"  Dropped {dropped} sample(s) with missing Engtype")

    merged["flat_label"] = merged["Engtype"]
    merged["audio_root"] = audio_root
    print(f"  {len(merged)} samples | classes: {merged['flat_label'].value_counts().to_dict()}")
    return merged


def get_audio_path(row, is_unseen: bool = False) -> Path:
    root     = Path(row["audio_root"])
    filename = row["filename"]
    fold     = int(row["fold"])

    if filename.startswith("fold_"):
        return root / filename

    # Unseen: try with class subfolder, then without
    if is_unseen:
        with_class    = root / f"fold_{fold}" / str(int(row["class"])) / filename
        without_class = root / f"fold_{fold}" / filename
        return with_class if with_class.exists() else without_class

    # Main dataset: fold_X/class/filename
    return root / f"fold_{int(row['original_fold'])}" / str(int(row["class"])) / filename


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    target_len = SR * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio


def extract_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def precompute_features(meta: pd.DataFrame, feature_type: str,
                        desc: str = "", is_unseen: bool = False):
    features, valid_indices = [], []
    for idx, row in tqdm(meta.iterrows(), total=len(meta),
                         desc=f"  Extracting {feature_type} [{desc}]"):
        path = get_audio_path(row, is_unseen=is_unseen)
        try:
            audio = load_audio(path)
            feat  = extract_mel(audio) if feature_type == "mel" else extract_mfcc(audio)
            features.append(feat)
            valid_indices.append(idx)
        except Exception as e:
            print(f"    [SKIP] {path}: {e}")
    return features, meta.loc[valid_indices].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CNN (identical architecture to train_and_evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.drop3 = nn.Dropout(0.5)
        self.fc1   = nn.Linear(128, 128)

    def forward(self, x):
        x = self.drop1(self.pool1(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop2(self.pool2(self.bn2(F.relu(self.conv2(x)))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.gap(x).flatten(1)
        x = self.drop3(x)
        return F.relu(self.fc1(x))


class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _CNNBackbone()
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class SubclassCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = _CNNBackbone()
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def train_model(model, X_train, y_train, *, binary, epochs=30,
                batch_size=32, val_split=0.1, patience=4, lr=1e-3,
                device=None, seed=42):
    device = device or get_device()
    model  = model.to(device)
    if seed is not None:
        torch.manual_seed(seed)

    n     = len(X_train)
    n_val = max(1, int(round(n * val_split)))
    X_tr  = torch.from_numpy(X_train[:-n_val]).float()
    X_val = torch.from_numpy(X_train[-n_val:]).float()

    if binary:
        y_tr    = torch.from_numpy(y_train[:-n_val]).float()
        y_val   = torch.from_numpy(y_train[-n_val:]).float()
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        y_tr    = torch.from_numpy(y_train[:-n_val]).long()
        y_val   = torch.from_numpy(y_train[-n_val:]).long()
        loss_fn = nn.CrossEntropyLoss()

    loader   = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    optim    = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state, bad_epochs = None, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()

        model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                xb = X_val[i:i+batch_size].to(device)
                yb = y_val[i:i+batch_size].to(device)
                total_loss += loss_fn(model(xb), yb).item() * len(xb)
                total_n    += len(xb)
        val_loss = total_loss / max(total_n, 1)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, X, batch_size=32, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    X_t, chunks = torch.from_numpy(X).float(), []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            chunks.append(model(X_t[i:i+batch_size].to(device)).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0,))


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION — RF
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rf(train_meta, train_feats, test_meta, test_feats):
    fold_results = []

    for fold in range(3):
        # Train on all folds except this one
        train_mask = (train_meta["fold"] != fold).values
        fold_train_meta = train_meta[train_mask].reset_index(drop=True)
        fold_train_feats = [f for f, m in zip(train_feats, train_mask) if m]

        X_train = np.array(fold_train_feats)
        y_train = fold_train_meta["flat_label"].values

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            min_samples_split=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_res, y_res)

        X_test = np.array(test_feats)
        y_test = test_meta["flat_label"].values
        y_pred = rf.predict(X_test)

        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro",
                          labels=ALL_LABELS, zero_division=0)
        report = classification_report(y_test, y_pred,
                                       labels=ALL_LABELS, zero_division=0)
        cm     = confusion_matrix(y_test, y_pred, labels=ALL_LABELS).tolist()

        print(f"    [Fold {fold}] acc={acc:.4f}  f1_macro={f1:.4f}")
        # Count predictions by class
        pred_counts = {label: int(np.sum(y_pred == label)) for label in ALL_LABELS}
        print(f"      Pred distribution: {pred_counts}")

        fold_results.append({
            "fold": fold,
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "report": report,
            "confusion_matrix": [[int(x) for x in row] for row in cm],
            "pred_distribution": pred_counts,
        })

    # Aggregate across folds
    mean_acc = float(np.mean([r["accuracy"] for r in fold_results]))
    std_acc = float(np.std([r["accuracy"] for r in fold_results]))
    mean_f1 = float(np.mean([r["f1_macro"] for r in fold_results]))
    std_f1 = float(np.std([r["f1_macro"] for r in fold_results]))

    print(f"  → RF  acc={mean_acc:.4f}±{std_acc:.4f}  f1_macro={mean_f1:.4f}±{std_f1:.4f}")

    return {
        "model":            "RandomForest",
        "mean_accuracy":    mean_acc,
        "std_accuracy":     std_acc,
        "mean_f1_macro":    mean_f1,
        "std_f1_macro":     std_f1,
        "labels":           ALL_LABELS,
        "folds":            fold_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION — CNN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cnn(train_meta, train_mels, test_meta, test_mels):
    device = get_device()
    le     = LabelEncoder()
    le.fit(SUBCLASSES)

    fold_results = []

    for fold in range(3):
        # Train on all folds except this one
        train_mask = (train_meta["fold"] != fold).values
        fold_train_meta = train_meta[train_mask].reset_index(drop=True)
        fold_train_mels = [f for f, m in zip(train_mels, train_mask) if m]

        X_tr_raw = np.array(fold_train_mels)
        mu, sigma = X_tr_raw.mean(), X_tr_raw.std() + 1e-8
        X_train = ((X_tr_raw - mu) / sigma)[:, np.newaxis].astype(np.float32)
        X_test  = ((np.array(test_mels) - mu) / sigma)[:, np.newaxis].astype(np.float32)

        # Stage 1 — binary
        y_tr_bin = fold_train_meta["class"].values.astype(np.float32)
        y_te_bin = test_meta["class"].values.astype(np.float32)

        binary = BinaryCNN()
        train_model(binary, X_train, y_tr_bin, binary=True, device=device)

        s1_logits = predict(binary, X_test, device=device)
        s1_preds  = (torch.sigmoid(torch.from_numpy(s1_logits)).numpy() >= 0.5).astype(int)
        s1_acc    = accuracy_score(y_te_bin, s1_preds)
        s1_f1     = f1_score(y_te_bin, s1_preds, average="macro")
        s1_report = classification_report(y_te_bin, s1_preds,
                                          target_names=["background","aircraft"],
                                          labels=[0, 1],
                                          zero_division=0)
        print(f"    [Fold {fold}] acc={s1_acc:.4f}  f1_macro={s1_f1:.4f}")
        print(f"      Pred distribution: bg={np.sum(s1_preds==0)} aircraft={np.sum(s1_preds==1)}")

        s1_cm = confusion_matrix(y_te_bin, s1_preds, labels=[0,1]).tolist()
        fold_result = {
            "fold": fold,
            "stage1": {
                "accuracy": float(s1_acc),
                "f1_macro": float(s1_f1),
                "report":   s1_report,
                "confusion_matrix": [[int(x) for x in row] for row in s1_cm],
                "pred_distribution": [int(np.sum(s1_preds==0)), int(np.sum(s1_preds==1))],
            },
            "stage2": None,
        }

        # Stage 2 — subclass (aircraft only)
        tr_ac = fold_train_meta["class"] == 1
        te_ac = test_meta["class"]  == 1

        if tr_ac.sum() > 0 and te_ac.sum() > 0:
            y_tr_sub = le.transform(fold_train_meta.loc[tr_ac, "Engtype"]).astype(np.int64)
            y_te_sub = test_meta.loc[te_ac, "Engtype"].values

            subclass = SubclassCNN(n_classes=len(SUBCLASSES))
            train_model(subclass, X_train[tr_ac.values], y_tr_sub,
                        binary=False, device=device)

            s2_logits = predict(subclass, X_test[te_ac.values], device=device)
            sub_preds = le.inverse_transform(np.argmax(s2_logits, axis=1))

            s2_acc    = accuracy_score(y_te_sub, sub_preds)
            s2_f1     = f1_score(y_te_sub, sub_preds, average="macro",
                                 labels=SUBCLASSES, zero_division=0)
            s2_report = classification_report(y_te_sub, sub_preds,
                                              labels=SUBCLASSES, zero_division=0)
            s2_cm     = confusion_matrix(y_te_sub, sub_preds, labels=SUBCLASSES).tolist()
            print(f"      Acc={s2_acc:.4f}  f1_macro={s2_f1:.4f}")
            s2_pred_counts = {label: int(np.sum(sub_preds == label)) for label in SUBCLASSES}
            print(f"      Pred distribution: {s2_pred_counts}")

            fold_result["stage2"] = {
                "accuracy":         float(s2_acc),
                "f1_macro":         float(s2_f1),
                "report":           s2_report,
                "confusion_matrix": [[int(x) for x in row] for row in s2_cm],
                "labels":           SUBCLASSES,
                "pred_distribution": s2_pred_counts,
            }

        fold_results.append(fold_result)

    # Aggregate across folds
    s1_accs = [r["stage1"]["accuracy"] for r in fold_results]
    s1_f1s = [r["stage1"]["f1_macro"] for r in fold_results]
    print(f"  → CNN (Binary)  acc={np.mean(s1_accs):.4f}±{np.std(s1_accs):.4f}  "
          f"f1_macro={np.mean(s1_f1s):.4f}±{np.std(s1_f1s):.4f}")

    s2_accs = [r["stage2"]["accuracy"] for r in fold_results if r["stage2"]]
    s2_f1s = [r["stage2"]["f1_macro"] for r in fold_results if r["stage2"]]
    if s2_accs:
        print(f"  → CNN (Subclass)  acc={np.mean(s2_accs):.4f}±{np.std(s2_accs):.4f}  "
              f"f1_macro={np.mean(s2_f1s):.4f}±{np.std(s2_f1s):.4f}")

    return {
        "model": "HierarchicalCNN",
        "stage1": {
            "mean_accuracy": float(np.mean(s1_accs)),
            "std_accuracy":  float(np.std(s1_accs)),
            "mean_f1_macro": float(np.mean(s1_f1s)),
            "std_f1_macro":  float(np.std(s1_f1s)),
        },
        "stage2": {
            "mean_accuracy": float(np.mean(s2_accs)) if s2_accs else None,
            "std_accuracy":  float(np.std(s2_accs)) if s2_accs else None,
            "mean_f1_macro": float(np.mean(s2_f1s)) if s2_f1s else None,
            "std_f1_macro":  float(np.std(s2_f1s)) if s2_f1s else None,
        } if s2_accs else None,
        "folds": fold_results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta",      required=True, help="Main dataset sample_meta CSV (clean)")
    parser.add_argument("--train_aircraft",  required=True, help="Main dataset aircraft_meta CSV")
    parser.add_argument("--train_root",      required=True, help="Main dataset audio root")
    parser.add_argument("--b25_meta",        help="B25 augmented metadata CSV")
    parser.add_argument("--b25_root",        help="B25 augmented audio root")
    parser.add_argument("--b50_meta",        help="B50 augmented metadata CSV")
    parser.add_argument("--b50_root",        help="B50 augmented audio root")
    parser.add_argument("--b75_meta",        help="B75 augmented metadata CSV")
    parser.add_argument("--b75_root",        help="B75 augmented audio root")
    parser.add_argument("--unseen_meta",     required=True, help="UNSEEN_DATA sample_meta CSV")
    parser.add_argument("--unseen_aircraft", required=True, help="UNSEEN_DATA aircraft_meta CSV")
    parser.add_argument("--unseen_root",     required=True, help="UNSEEN_DATA audio root")
    parser.add_argument("--output_dir",      default="results_unseen")
    parser.add_argument("--clean_only",      action="store_true", help="Train on clean data only (skip augmented)")
    parser.add_argument("--augmented_only",  action="store_true", help="Train on augmented data only (skip clean)")
    parser.add_argument("--rf_only",  action="store_true", help="Skip CNN")
    parser.add_argument("--cnn_only", action="store_true", help="Skip RF")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Metadata ───────────────────────────────────────────────────────────
    print("\n[1/4] Building metadata ...")

    # Load clean data (if not augmented_only)
    train_datasets = []
    if not args.augmented_only:
        print("  Train (Clean):")
        train_meta  = build_meta(args.train_meta, args.train_aircraft, args.train_root)
        train_datasets.append(train_meta)

    # Load augmented datasets if provided (and not clean_only)
    if not args.clean_only:
        if args.b25_meta and args.b25_root:
            print("  Train (B25):")
            b25_meta = build_meta(args.b25_meta, args.train_aircraft, args.b25_root)
            train_datasets.append(b25_meta)

        if args.b50_meta and args.b50_root:
            print("  Train (B50):")
            b50_meta = build_meta(args.b50_meta, args.train_aircraft, args.b50_root)
            train_datasets.append(b50_meta)

        if args.b75_meta and args.b75_root:
            print("  Train (B75):")
            b75_meta = build_meta(args.b75_meta, args.train_aircraft, args.b75_root)
            train_datasets.append(b75_meta)

    # Combine all training datasets
    combined_train_meta = pd.concat(train_datasets, ignore_index=True)
    print(f"  Combined Train: {len(combined_train_meta)} samples")

    print("  Unseen:")
    unseen_meta = build_meta(args.unseen_meta, args.unseen_aircraft,
                             args.unseen_root, is_unseen=True)

    all_results = {}

    # ── 2. RF ─────────────────────────────────────────────────────────────────
    if not args.cnn_only:
        print("\n[2/4] Extracting MFCC features ...")
        train_mfcc, train_mfcc_meta = precompute_features(
            combined_train_meta, "mfcc", desc="Train")
        unseen_mfcc, unseen_mfcc_meta = precompute_features(
            unseen_meta, "mfcc", desc="Unseen", is_unseen=True)

        print("\n[3/4] Training RF and evaluating on UNSEEN_DATA ...")
        rf_results = evaluate_rf(train_mfcc_meta, train_mfcc,
                                 unseen_mfcc_meta, unseen_mfcc)
        all_results["rf"] = rf_results
    else:
        print("\n[2/4] Skipping MFCC (--cnn_only)")
        print("\n[3/4] Skipping RF (--cnn_only)")

    # ── 3. CNN ────────────────────────────────────────────────────────────────
    if not args.rf_only:
        print("\n[2/4] Extracting Mel spectrogram features ...")
        train_mel, train_mel_meta = precompute_features(
            combined_train_meta, "mel", desc="Train")
        unseen_mel, unseen_mel_meta = precompute_features(
            unseen_meta, "mel", desc="Unseen", is_unseen=True)

        print("\n[4/4] Training CNN and evaluating on UNSEEN_DATA ...")
        cnn_results = evaluate_cnn(train_mel_meta, train_mel,
                                   unseen_mel_meta, unseen_mel)
        all_results["cnn"] = cnn_results
    else:
        print("\n[4/4] Skipping CNN (--rf_only)")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    out_path = output_dir / "unseen_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done! Results saved to: {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
