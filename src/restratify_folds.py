"""
restratify_folds.py
===================
Takes the original AeroSonicDB metadata and reassigns fold IDs using
Stratified 3-Fold cross-validation on aircraft engine types.

Why restratify?
  Original fold distribution (aircraft only):
    Engtype     Piston  Turbofan  Turboprop  Turboshaft
    fold
    0               37       470        114           4
    1               24       469         85          11
    2               20       504         78           3
    3               20       565        100          17
    4               12       450         72           1   ← only 1 Turboshaft!

  With 36 Turboshaft total and 5 folds → some folds get 1 sample.
  One misclassification = 100% recall swing on that fold.

  Stratified 3-fold target (~12 Turboshaft per fold):
    One misclassification = ~8% recall swing — much more stable.

What this script produces:
  - sample_meta_stratified3.csv  ← drop-in replacement for sample_meta_new.csv
    Same columns as original, only 'fold' column is changed (values 0,1,2).

Usage:
  python restratify_folds.py \
      --sample_meta   data/sample_meta_new.csv \
      --aircraft_meta data/aircraft_meta_new.csv \
      --output        data/sample_meta_stratified3.csv

  Then use sample_meta_stratified3.csv everywhere instead of sample_meta_new.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 3
RANDOM_STATE = 42


def restratify(sample_meta_path: str,
               aircraft_meta_path: str,
               output_path: str) -> pd.DataFrame:

    # ── Load and merge ────────────────────────────────────────────────────────
    sample   = pd.read_csv(sample_meta_path)
    aircraft = pd.read_csv(aircraft_meta_path)[["hex_id", "Engtype"]]
    merged   = sample.merge(aircraft, on="hex_id", how="left")

    merged.loc[merged["class"] == 0, "Engtype"] = "background"
    merged["Engtype"] = merged["Engtype"].replace("Diesel Engine", "Piston")
    merged = merged[~((merged["class"] == 1) & (merged["Engtype"].isna()))]
    merged["original_fold"] = merged["fold"] 

    print(f"Total samples after cleaning: {len(merged)}")
    print("\nOriginal fold distribution (aircraft only):")
    orig_dist = (merged[merged["class"] == 1]
                 .groupby("fold")["Engtype"]
                 .value_counts()
                 .unstack(fill_value=0))
    print(orig_dist.to_string())

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_STATE)

    # ── Restratify aircraft ───────────────────────────────────────────────────
    aircraft_rows = merged[merged["class"] == 1].copy()
    new_folds     = np.zeros(len(aircraft_rows), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(
            skf.split(aircraft_rows, aircraft_rows["Engtype"])):
        new_folds[test_idx] = fold_idx
    aircraft_rows["fold"] = new_folds

    # ── Restratify background ─────────────────────────────────────────────────
    # Background is large and balanced — just split evenly
    background_rows = merged[merged["class"] == 0].copy()
    dummy           = np.zeros(len(background_rows), dtype=int)
    new_folds_bg    = np.zeros(len(background_rows), dtype=int)
    for fold_idx, (_, test_idx) in enumerate(
            skf.split(background_rows, dummy)):
        new_folds_bg[test_idx] = fold_idx
    background_rows["fold"] = new_folds_bg

    # ── Recombine ─────────────────────────────────────────────────────────────
    result = pd.concat([aircraft_rows, background_rows], ignore_index=True)
    result["fold"] = result["fold"].astype(int)

    print(f"\nNew stratified {N_FOLDS}-fold distribution (aircraft only):")
    new_dist = (result[result["class"] == 1]
                .groupby("fold")["Engtype"]
                .value_counts()
                .unstack(fill_value=0))
    print(new_dist.to_string())

    print(f"\nNew stratified {N_FOLDS}-fold distribution (background):")
    bg_dist = result[result["class"] == 0]["fold"].value_counts().sort_index()
    print(bg_dist.to_string())

    # ── Save — only keep original sample_meta columns + updated fold ──────────
    # This makes it a drop-in replacement for sample_meta_new.csv
    original_cols = pd.read_csv(sample_meta_path).columns.tolist()

    # Drop rows that were removed during cleaning (missing Engtype)
    # by inner-joining back to the original sample_meta on filename
    #output = result[original_cols].copy()
    output = result[original_cols + ["original_fold"]].copy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Columns: {output.columns.tolist()}")
    print(f"Fold values: {sorted(output['fold'].unique())}")

    return output

       
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_meta",   required=True,
                        help="Path to original sample_meta_new.csv")
    parser.add_argument("--aircraft_meta", required=True,
                        help="Path to aircraft_meta_new.csv")
    parser.add_argument("--output",        required=True,
                        help="Output path for restratified metadata CSV")
    args = parser.parse_args()

    restratify(args.sample_meta, args.aircraft_meta, args.output)


if __name__ == "__main__":
    main()
