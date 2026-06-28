"""
train_and_evaluate.py
=====================
Cross-dataset evaluation pipeline for aircraft sound classification.
Memory-efficient version: features are extracted fold-by-fold and discarded
after each fold to avoid running out of RAM on machines with limited memory.

Models:
  1. CNN — Hierarchical
       Stage 1: Binary classifier (aircraft vs background)
       Stage 2: Subclass classifier (Piston / Turboprop / Turbofan / Turboshaft)

  2. Random Forest — Flat
       Single classifier: background / Piston / Turboprop / Turbofan / Turboshaft

Experimental matrix (8 experiments per model):
  ┌─────────────────────────┬──────────┬─────────────────────────────────────┐
  │ Train                   │ Test     │ Purpose                             │
  ├─────────────────────────┼──────────┼─────────────────────────────────────┤
  │ A (clean)               │ A        │ Baseline                            │
  │ A (clean)               │ B25      │ Robustness to unseen noise (low)    │
  │ A (clean)               │ B50      │ Robustness to unseen noise (med)    │
  │ A (clean)               │ B75      │ Robustness to unseen noise (high)   │
  │ A + B25 + B50 + B75     │ A        │ Does augmentation hurt clean perf?  │
  │ A + B25 + B50 + B75     │ B25      │ Does augmentation help? (low)       │
  │ A + B25 + B50 + B75     │ B50      │ Does augmentation help? (med)       │
  │ A + B25 + B50 + B75     │ B75      │ Does augmentation help? (high)      │
  └─────────────────────────┴──────────┴─────────────────────────────────────┘

Usage:
  python train_and_evaluate.py \
      --sample_meta   data/sample_meta_new.csv \
      --aircraft_meta data/aircraft_meta_new.csv \
      --audio_root    data \
      --b25_meta      data/output/B25_meta.csv \
      --b50_meta      data/output/B50_meta.csv \
      --b75_meta      data/output/B75_meta.csv \
      --b25_root      data/output/B25 \
      --b50_root      data/output/B50 \
      --b75_root      data/output/B75 \
      --output_dir    results \
      [--rf_only] [--cnn_only]
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
SR         = 16000
DURATION   = 10
N_MELS     = 32
N_FFT      = 1024
HOP_LENGTH = 256
N_MFCC     = 40
SUBCLASSES = ["Piston", "Turboprop", "Turbofan", "Turboshaft"]
ALL_LABELS = ["background"] + SUBCLASSES
FOLDS      = [0, 1, 2, 3, 4]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def build_master_meta(sample_meta_path: str, aircraft_meta_path: str) -> pd.DataFrame:
    sample   = pd.read_csv(sample_meta_path)
    aircraft = pd.read_csv(aircraft_meta_path)[["hex_id", "Engtype"]]
    merged   = sample.merge(aircraft, on="hex_id", how="left")

    merged.loc[merged["class"] == 0, "Engtype"] = "background"
    merged["Engtype"] = merged["Engtype"].replace("Diesel Engine", "Piston")

    before = len(merged)
    merged = merged[~((merged["class"] == 1) & (merged["Engtype"].isna()))]
    dropped = before - len(merged)
    if dropped:
        print(f"  Dropped {dropped} aircraft sample(s) with missing Engtype")

    merged["flat_label"] = merged["Engtype"]
    merged["audio_root"] = ""
    print(f"  Master metadata: {len(merged)} samples")
    print(merged["flat_label"].value_counts().to_string())
    return merged


def make_noisy_meta(noisy_meta_path: str,
                    aircraft_meta_path: str,
                    noisy_audio_root: str,
                    clean_background: pd.DataFrame) -> pd.DataFrame:
    noisy    = pd.read_csv(noisy_meta_path)
    aircraft = pd.read_csv(aircraft_meta_path)[["hex_id", "Engtype"]]
    noisy    = noisy.merge(aircraft, on="hex_id", how="left")
    noisy["Engtype"] = noisy["Engtype"].replace("Diesel Engine", "Piston")
    noisy    = noisy[~((noisy["class"] == 1) & (noisy["Engtype"].isna()))]
    noisy["flat_label"] = noisy["Engtype"]
    noisy["audio_root"] = noisy_audio_root
    return pd.concat([noisy, clean_background], ignore_index=True)


def get_audio_path(row) -> Path:
    root     = Path(row["audio_root"])
    filename = row["filename"]
    if filename.startswith("fold_"):
        return root / filename
    return root / f"fold_{int(row['fold'])}" / str(int(row['class'])) / filename


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION  (per-fold, memory efficient)
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
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def extract_features_for_rows(meta_subset: pd.DataFrame,
                               feature_type: str) -> tuple:
    """
    Extract features only for the given subset of rows.
    Returns (features_list, valid_meta) — failed rows are dropped.
    """
    features      = []
    valid_indices = []
    for idx, row in meta_subset.iterrows():
        path = get_audio_path(row)
        try:
            audio = load_audio(path)
            feat  = extract_mel(audio) if feature_type == "mel" else extract_mfcc(audio)
            features.append(feat)
            valid_indices.append(idx)
        except Exception as e:
            print(f"    [SKIP] {path}: {e}")
    valid_meta = meta_subset.loc[valid_indices].reset_index(drop=True)
    return features, valid_meta


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CNN
# ══════════════════════════════════════════════════════════════════════════════

def build_binary_cnn(input_shape):
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="binary_cnn")
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def build_subclass_cnn(input_shape, n_classes):
    from tensorflow.keras import layers, models
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="subclass_cnn")
    m.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def run_cnn_fold(train_meta, train_mels, test_meta, test_mels, fold: int) -> dict:
    import tensorflow as tf
    from tensorflow.keras import backend as K

    le = LabelEncoder()
    le.fit(SUBCLASSES)

    X_tr_raw = np.array(train_mels)
    mu, sigma = X_tr_raw.mean(), X_tr_raw.std() + 1e-8
    X_train = ((X_tr_raw - mu) / sigma)[..., np.newaxis]
    del X_tr_raw
    gc.collect()

    X_test  = ((np.array(test_mels) - mu) / sigma)[..., np.newaxis]
    input_shape = X_train.shape[1:]

    # Stage 1 — binary
    y_tr_bin = train_meta["class"].values.astype(np.float32)
    y_te_bin = test_meta["class"].values.astype(np.float32)

    binary = build_binary_cnn(input_shape)
    binary.fit(X_train, y_tr_bin, epochs=30, batch_size=16,
               validation_split=0.1, verbose=0,
               callbacks=[tf.keras.callbacks.EarlyStopping(
                   patience=4, restore_best_weights=True)])

    s1_preds = (binary.predict(X_test, verbose=0).flatten() >= 0.5).astype(int)
    s1_acc   = accuracy_score(y_te_bin, s1_preds)
    s1_f1    = f1_score(y_te_bin, s1_preds, average="macro")

    # Free binary model memory before stage 2
    del binary
    K.clear_session()
    gc.collect()

    # Stage 2 — subclass (aircraft only)
    tr_ac = train_meta["class"] == 1
    te_ac = test_meta["class"]  == 1

    y_tr_sub = le.transform(train_meta.loc[tr_ac, "Engtype"])
    y_te_sub = test_meta.loc[te_ac, "Engtype"].values

    subclass = build_subclass_cnn(input_shape, n_classes=len(SUBCLASSES))
    subclass.fit(X_train[tr_ac.values], y_tr_sub, epochs=30, batch_size=16,
                 validation_split=0.1, verbose=0,
                 callbacks=[tf.keras.callbacks.EarlyStopping(
                     patience=4, restore_best_weights=True)])

    sub_preds = le.inverse_transform(
        np.argmax(subclass.predict(X_test[te_ac.values], verbose=0), axis=1)
    )

    s2_acc    = accuracy_score(y_te_sub, sub_preds)
    s2_f1     = f1_score(y_te_sub, sub_preds, average="macro",
                         labels=SUBCLASSES, zero_division=0)
    s2_report = classification_report(y_te_sub, sub_preds,
                                      labels=SUBCLASSES, zero_division=0)
    s2_cm     = confusion_matrix(y_te_sub, sub_preds, labels=SUBCLASSES).tolist()

    print(f"    [Fold {fold}] Stage1 acc={s1_acc:.3f} f1={s1_f1:.3f} | "
          f"Stage2 acc={s2_acc:.3f} f1={s2_f1:.3f}")

    # Free memory before next fold
    del subclass, X_train, X_test
    K.clear_session()
    gc.collect()

    return {
        "fold": fold,
        "stage1": {"accuracy": s1_acc, "f1_macro": s1_f1},
        "stage2": {"accuracy": s2_acc, "f1_macro": s2_f1,
                   "report": s2_report, "confusion_matrix": s2_cm,
                   "labels": SUBCLASSES},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

#def run_rf_fold(train_meta, train_mfcc, test_meta, test_mfcc, fold: int) -> dict:
#    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
#                                random_state=42, n_jobs=-1)
#    rf.fit(np.array(train_mfcc), train_meta["flat_label"].values)
#    y_pred = rf.predict(np.array(test_mfcc))
#    y_test = test_meta["flat_label"].values

#    acc    = accuracy_score(y_test, y_pred)
#    f1     = f1_score(y_test, y_pred, average="macro",
#                      labels=ALL_LABELS, zero_division=0)
#    report = classification_report(y_test, y_pred,
#                                   labels=ALL_LABELS, zero_division=0)
#    cm     = confusion_matrix(y_test, y_pred, labels=ALL_LABELS).tolist()

#    print(f"    [Fold {fold}] acc={acc:.3f}  f1={f1:.3f}")
#    return {"fold": fold, "accuracy": acc, "f1_macro": f1,
#            "report": report, "confusion_matrix": cm, "labels": ALL_LABELS}

def run_rf_fold(train_meta, train_mfcc, test_meta, test_mfcc, fold: int) -> dict:
    
    X_train = np.array(train_mfcc)
    y_train = train_meta["flat_label"].values

    # Oversample minority classes before training
    # k_neighbors=3 because Piston only has ~90 training samples per fold
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=300,        # more trees = more stable
        class_weight="balanced",
        min_samples_leaf=2,      # prevents overfitting on resampled data
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res)
    y_pred = rf.predict(np.array(test_mfcc))
    y_test = test_meta["flat_label"].values

    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro",
                      labels=ALL_LABELS, zero_division=0)
    report = classification_report(y_test, y_pred,
                                   labels=ALL_LABELS, zero_division=0)
    cm     = confusion_matrix(y_test, y_pred, labels=ALL_LABELS).tolist()

    print(f"    [Fold {fold}] acc={acc:.3f}  f1={f1:.3f}")
    return {"fold": fold, "accuracy": acc, "f1_macro": f1,
            "report": report, "confusion_matrix": cm, "labels": ALL_LABELS}
    
# ══════════════════════════════════════════════════════════════════════════════
# 5.  EXPERIMENT RUNNER  (memory-efficient: extract per fold)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp_name: str,
                   train_metas: list,       # list of DataFrames (one per train dataset)
                   test_meta: pd.DataFrame,
                   feature_type: str,
                   output_dir: Path) -> dict:
    """
    Run 5-fold cross-dataset experiment.
    Features are extracted fresh for each fold and discarded after — this keeps
    memory usage to one fold's worth of data at a time instead of the whole dataset.

    train_metas : list of DataFrames for training datasets
    test_meta   : DataFrame for the test dataset
    feature_type: 'mfcc' → RF  |  'mel' → CNN
    """
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold in FOLDS:
        print(f"  [Fold {fold}] Extracting features ...", flush=True)

        # ── Training data: extract train folds only ───────────────────────────
        combined_train_meta  = []
        combined_train_feats = []

        for meta in train_metas:
            train_subset = meta[meta["fold"] != fold]
            feats, valid = extract_features_for_rows(train_subset, feature_type)
            combined_train_meta.append(valid)
            combined_train_feats.extend(feats)
            del feats
            gc.collect()

        combined_train_meta = pd.concat(combined_train_meta, ignore_index=True)

        # ── Test data: extract test fold only ─────────────────────────────────
        test_subset = test_meta[test_meta["fold"] == fold]
        test_feats, fold_test_meta = extract_features_for_rows(test_subset, feature_type)

        # ── Train and evaluate ────────────────────────────────────────────────
        if feature_type == "mfcc":
            res = run_rf_fold(combined_train_meta, combined_train_feats,
                              fold_test_meta,      test_feats, fold)
        else:
            res = run_cnn_fold(combined_train_meta, combined_train_feats,
                               fold_test_meta,      test_feats, fold)

        fold_results.append(res)

        # Free everything before next fold
        del combined_train_meta, combined_train_feats, test_feats, fold_test_meta
        gc.collect()

    # ── Aggregate across folds ────────────────────────────────────────────────
    if feature_type == "mfcc":
        summary = {
            "experiment":    exp_name,
            "model":         "RandomForest",
            "mean_accuracy": float(np.mean([r["accuracy"] for r in fold_results])),
            "std_accuracy":  float(np.std( [r["accuracy"] for r in fold_results])),
            "mean_f1_macro": float(np.mean([r["f1_macro"] for r in fold_results])),
            "std_f1_macro":  float(np.std( [r["f1_macro"] for r in fold_results])),
            "folds": fold_results,
        }
        print(f"  → RF  acc={summary['mean_accuracy']:.3f}±{summary['std_accuracy']:.3f}  "
              f"f1={summary['mean_f1_macro']:.3f}±{summary['std_f1_macro']:.3f}")
    else:
        summary = {
            "experiment": exp_name,
            "model":      "HierarchicalCNN",
            "stage1": {
                "mean_accuracy": float(np.mean([r["stage1"]["accuracy"] for r in fold_results])),
                "std_accuracy":  float(np.std( [r["stage1"]["accuracy"] for r in fold_results])),
                "mean_f1_macro": float(np.mean([r["stage1"]["f1_macro"] for r in fold_results])),
                "std_f1_macro":  float(np.std( [r["stage1"]["f1_macro"] for r in fold_results])),
            },
            "stage2": {
                "mean_accuracy": float(np.mean([r["stage2"]["accuracy"] for r in fold_results])),
                "std_accuracy":  float(np.std( [r["stage2"]["accuracy"] for r in fold_results])),
                "mean_f1_macro": float(np.mean([r["stage2"]["f1_macro"] for r in fold_results])),
                "std_f1_macro":  float(np.std( [r["stage2"]["f1_macro"] for r in fold_results])),
            },
            "folds": fold_results,
        }
        print(f"  → CNN Stage1 acc={summary['stage1']['mean_accuracy']:.3f}±{summary['stage1']['std_accuracy']:.3f}  "
              f"f1={summary['stage1']['mean_f1_macro']:.3f}")
        print(f"  → CNN Stage2 acc={summary['stage2']['mean_accuracy']:.3f}±{summary['stage2']['std_accuracy']:.3f}  "
              f"f1={summary['stage2']['mean_f1_macro']:.3f}")

    label = "rf" if feature_type == "mfcc" else "cnn"
    with open(exp_dir / f"{label}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_meta",   required=True)
    parser.add_argument("--aircraft_meta", required=True)
    parser.add_argument("--audio_root",    required=True)
    parser.add_argument("--b25_meta",      required=True)
    parser.add_argument("--b50_meta",      required=True)
    parser.add_argument("--b75_meta",      required=True)
    parser.add_argument("--b25_root",      required=True)
    parser.add_argument("--b50_root",      required=True)
    parser.add_argument("--b75_root",      required=True)
    parser.add_argument("--output_dir",    default="results")
    parser.add_argument("--rf_only",       action="store_true",
                        help="Skip CNN — useful for quick testing")
    parser.add_argument("--cnn_only",      action="store_true",
                        help="Skip RF — run CNN only")
    parser.add_argument("--experiment", type=int, default=None,
                    help="Run only this experiment index (0-7). If omitted, runs all.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Build metadata ─────────────────────────────────────────────────────
    print("\n[1/4] Building metadata ...")
    master = build_master_meta(args.sample_meta, args.aircraft_meta)
    master["audio_root"] = args.audio_root
    clean_bg = master[master["class"] == 0].copy()

    b25 = make_noisy_meta(args.b25_meta, args.aircraft_meta, args.b25_root, clean_bg)
    b50 = make_noisy_meta(args.b50_meta, args.aircraft_meta, args.b50_root, clean_bg)
    b75 = make_noisy_meta(args.b75_meta, args.aircraft_meta, args.b75_root, clean_bg)

    all_metas = {"A": master, "B25": b25, "B50": b50, "B75": b75}

    # ── 2. Define experiments ─────────────────────────────────────────────────
    experiments = [
        # (name,                        train_keys,              test_key)
        ("train_clean__test_A",         ["A"],                   "A"),
        ("train_clean__test_B25",       ["A"],                   "B25"),
        ("train_clean__test_B50",       ["A"],                   "B50"),
        ("train_clean__test_B75",       ["A"],                   "B75"),
        ("train_augmented__test_A",     ["A","B25","B50","B75"], "A"),
        ("train_augmented__test_B25",   ["A","B25","B50","B75"], "B25"),
        ("train_augmented__test_B50",   ["A","B25","B50","B75"], "B50"),
        ("train_augmented__test_B75",   ["A","B25","B50","B75"], "B75"),
    ]
    if args.experiment is not None:
        experiments = [experiments[args.experiment]]

    all_results = {"rf": [], "cnn": []}

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    if not args.cnn_only:
        print("\n[2/4] Running Random Forest experiments ...")
        for exp_name, train_keys, test_key in experiments:
            print(f"\n  {exp_name}")
            summary = run_experiment(
                exp_name,
                train_metas=[all_metas[k] for k in train_keys],
                test_meta=all_metas[test_key],
                feature_type="mfcc",
                output_dir=output_dir,
            )
            all_results["rf"].append(summary)
    else:
        print("\n[2/4] Skipping Random Forest (--cnn_only)")

    # ── 4. CNN ────────────────────────────────────────────────────────────────
    if not args.rf_only:
        print("\n[3/4] Running CNN experiments ...")
        for exp_name, train_keys, test_key in experiments:
            print(f"\n  {exp_name}")
            summary = run_experiment(
                exp_name,
                train_metas=[all_metas[k] for k in train_keys],
                test_meta=all_metas[test_key],
                feature_type="mel",
                output_dir=output_dir,
            )
            all_results["cnn"].append(summary)
    else:
        print("\n[3/4] Skipping CNN (--rf_only)")

    # ── 5. Save master summary ────────────────────────────────────────────────
    summary_path = output_dir / "all_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All done! Results saved to: {output_dir}/")
    print(f"  Master summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
