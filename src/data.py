"""
Metadata layer for the AeroSonicDB pipeline.

Owns the label vocabulary and fold convention, and builds the
DataFrame views the experiment runner consumes:

  - build_master_meta: merge sample_meta + aircraft_meta, clean Engtype,
    add flat_label and audio_root columns.
  - make_noisy_meta: combine an SNR-augmented B-meta CSV with clean
    background rows; aircraft rows point at the B root, background
    rows keep their clean audio_root.
  - get_audio_path: resolve a row to its WAV on disk - clean rows via
    original_fold (AeroSonicDB's on-disk layout), noisy rows via the
    fold_X/1/ prefix embedded in the filename.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Label vocabulary used across models.
SUBCLASSES = ["Piston", "Turboprop", "Turbofan", "Turboshaft"]
ALL_LABELS = ["background"] + SUBCLASSES

# Stratified 3-fold IDs (see src/restratify_folds.py).
FOLDS = [0, 1, 2]


def build_master_meta(sample_meta_path: str,
                      aircraft_meta_path: str) -> pd.DataFrame:
    """
    Merge sample_meta with aircraft_meta on hex_id.
    Cleans labels:
      - background (class=0)  → Engtype = 'background'
      - Diesel Engine         → Piston
      - Drop rows with missing Engtype for class=1
    Adds 'flat_label' column used by Random Forest.
    """
    sample   = pd.read_csv(sample_meta_path)
    aircraft = pd.read_csv(aircraft_meta_path)[["hex_id", "Engtype"]]
    merged   = sample.merge(aircraft, on="hex_id", how="left")

    merged.loc[merged["class"] == 0, "Engtype"] = "background"
    merged["Engtype"] = merged["Engtype"].replace("Diesel Engine", "Piston")

    before  = len(merged)
    merged  = merged[~((merged["class"] == 1) & (merged["Engtype"].isna()))]
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
    """
    Combine noisy aircraft rows with clean background rows.
    Aircraft rows point to noisy_audio_root; background rows keep their
    original audio_root from clean_background.
    """
    noisy    = pd.read_csv(noisy_meta_path)
    aircraft = pd.read_csv(aircraft_meta_path)[["hex_id", "Engtype"]]
    noisy    = noisy.merge(aircraft, on="hex_id", how="left")
    noisy["Engtype"] = noisy["Engtype"].replace("Diesel Engine", "Piston")
    noisy    = noisy[~((noisy["class"] == 1) & (noisy["Engtype"].isna()))]
    noisy["flat_label"] = noisy["Engtype"]
    noisy["audio_root"] = noisy_audio_root
    return pd.concat([noisy, clean_background], ignore_index=True)


def get_audio_path(row) -> Path:
    root = Path(row["audio_root"])
    filename = row["filename"]
    # Noisy datasets (B25/B50/B75) already have fold_X/1/ embedded in filename
    if filename.startswith("fold_"):
        return root / filename
    # Clean dataset A has bare filenames only
    return root / f"fold_{int(row['original_fold'])}" / str(int(row['class'])) / filename
