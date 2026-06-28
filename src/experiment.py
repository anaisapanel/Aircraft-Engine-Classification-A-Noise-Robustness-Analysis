"""
Per-fold training + evaluation runners, and the cross-dataset
cross-validation orchestrator.

  - run_cnn_fold: train both hierarchical CNN stages on one fold,
    predict on the test fold, return per-stage metrics.
  - run_rf_fold: SMOTE + Random Forest on one fold.
  - run_experiment: loop over folds, aggregate mean/std, write JSON.

The "experiment" abstraction takes (train_datasets, test_dataset) so
the same runner covers the 8-cell (clean vs augmented train) ×
(A/B25/B50/B75 test) matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder

from data import ALL_LABELS, FOLDS, SUBCLASSES
from models import (BinaryCNN, SubclassCNN, _predict_torch,
                    _train_torch_model, get_device)


def run_cnn_fold(train_meta, train_mels, test_meta, test_mels, fold: int) -> dict:
    device = get_device()

    le = LabelEncoder()
    le.fit(SUBCLASSES)

    # Normalise with train statistics; reshape to (N, 1, H, W) for PyTorch
    X_tr_raw = np.array(train_mels)
    mu, sigma = X_tr_raw.mean(), X_tr_raw.std() + 1e-8
    X_train = ((X_tr_raw - mu) / sigma)[:, np.newaxis, :, :].astype(np.float32)
    X_test  = ((np.array(test_mels) - mu) / sigma)[:, np.newaxis, :, :].astype(np.float32)

    # Stage 1 - binary
    y_tr_bin = train_meta["class"].values.astype(np.float32)
    y_te_bin = test_meta["class"].values.astype(np.float32)

    binary = BinaryCNN()
    _train_torch_model(binary, X_train, y_tr_bin,
                       binary=True, device=device)

    s1_logits = _predict_torch(binary, X_test, device=device)
    s1_probs  = torch.sigmoid(torch.from_numpy(s1_logits)).numpy()
    s1_preds  = (s1_probs >= 0.5).astype(int)
    s1_acc    = accuracy_score(y_te_bin, s1_preds)
    s1_f1     = f1_score(y_te_bin, s1_preds, average="macro")

    # Stage 2 - subclass (aircraft only)
    tr_ac = train_meta["class"] == 1
    te_ac = test_meta["class"]  == 1

    y_tr_sub = le.transform(train_meta.loc[tr_ac, "Engtype"]).astype(np.int64)
    y_te_sub = test_meta.loc[te_ac, "Engtype"].values

    subclass = SubclassCNN(n_classes=len(SUBCLASSES))
    _train_torch_model(subclass, X_train[tr_ac.values], y_tr_sub,
                       binary=False, device=device)

    s2_logits = _predict_torch(subclass, X_test[te_ac.values], device=device)
    sub_preds = le.inverse_transform(np.argmax(s2_logits, axis=1))

    s2_acc    = accuracy_score(y_te_sub, sub_preds)
    s2_f1     = f1_score(y_te_sub, sub_preds, average="macro",
                         labels=SUBCLASSES, zero_division=0)
    s2_report = classification_report(y_te_sub, sub_preds,
                                      labels=SUBCLASSES, zero_division=0)
    s2_cm     = confusion_matrix(y_te_sub, sub_preds, labels=SUBCLASSES).tolist()

    print(f"    [Fold {fold}] Stage1 acc={s1_acc:.3f} f1={s1_f1:.3f} | "
          f"Stage2 acc={s2_acc:.3f} f1={s2_f1:.3f}")

    return {
        "fold": fold,
        "stage1": {"accuracy": s1_acc, "f1_macro": s1_f1},
        "stage2": {"accuracy": s2_acc, "f1_macro": s2_f1,
                   "report": s2_report, "confusion_matrix": s2_cm,
                   "labels": SUBCLASSES},
    }


def run_rf_fold(train_meta, train_mfcc, test_meta, test_mfcc, fold: int) -> dict:
    from imblearn.over_sampling import SMOTE

    X_train = np.array(train_mfcc)
    y_train = train_meta["flat_label"].values

    # Oversample minority classes before training
    # k_neighbors=3 because Piston only has ~90 training samples per fold
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
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


def run_experiment(exp_name: str,
                   train_datasets: list,
                   test_dataset: tuple,
                   feature_type: str,
                   output_dir: Path) -> dict:
    """
    Run 3-fold cross-dataset experiment.

    train_datasets : list of (meta_df, features_list)
    test_dataset   : (meta_df, features_list)
    feature_type   : 'mfcc' → RF   |   'mel' → CNN
    """
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    test_meta, test_feats = test_dataset
    fold_results = []

    for fold in FOLDS:
        # Build combined training split (exclude test fold across ALL datasets)
        combined_train_meta  = []
        combined_train_feats = []
        for (meta, feats) in train_datasets:
            mask = (meta["fold"] != fold).values
            combined_train_meta.append(meta[mask].reset_index(drop=True))
            combined_train_feats.extend([f for f, m in zip(feats, mask) if m])
        combined_train_meta = pd.concat(combined_train_meta, ignore_index=True)

        # Build test split (only test fold from test dataset)
        test_mask       = (test_meta["fold"] == fold).values
        fold_test_meta  = test_meta[test_mask].reset_index(drop=True)
        fold_test_feats = [f for f, m in zip(test_feats, test_mask) if m]

        if feature_type == "mfcc":
            res = run_rf_fold(combined_train_meta, combined_train_feats,
                              fold_test_meta,      fold_test_feats, fold)
        else:
            res = run_cnn_fold(combined_train_meta, combined_train_feats,
                               fold_test_meta,      fold_test_feats, fold)
        fold_results.append(res)

    # Aggregate
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
