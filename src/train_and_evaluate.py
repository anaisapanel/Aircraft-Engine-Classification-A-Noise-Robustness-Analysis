"""
train_and_evaluate.py
=====================
Cross-dataset evaluation CLI for aircraft sound classification.

Composition root - wires together the four library modules:
  data.py       - metadata (build_master_meta, make_noisy_meta)
  features.py   - audio features (precompute_features)
  models.py     - CNN + torch training helpers (configure_gpu)
  experiment.py - per-fold runners + cross-dataset orchestration

Models:
  1. CNN - Hierarchical
       Stage 1: Binary classifier (aircraft vs background)
       Stage 2: Subclass classifier (Piston / Turboprop / Turbofan / Turboshaft)
  2. Random Forest - Flat
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

Key design decisions:
  - 3-fold cross-validation throughout
  - When combining datasets, all versions of the same recording share the same
    fold → no leakage between train and test
  - Test fold is always held out across ALL datasets simultaneously
  - Background (class=0) always comes from clean Dataset A
  - Validation split (10%) taken from training folds only

Usage:
  python src/train_and_evaluate.py \
      --sample_meta   data/sample_meta_stratified_snr.csv \
      --aircraft_meta data/aircraft_meta_new.csv \
      --audio_root    data/audio \
      --b25_meta      data/output_snr/B25_meta.csv \
      --b50_meta      data/output_snr/B50_meta.csv \
      --b75_meta      data/output_snr/B75_meta.csv \
      --b25_root      data/output_snr/B25 \
      --b50_root      data/output_snr/B50 \
      --b75_root      data/output_snr/B75 \
      --output_dir    results \
      [--rf_only] [--cnn_only]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from data import build_master_meta, make_noisy_meta
from experiment import run_experiment
from features import precompute_features
from models import configure_gpu

warnings.filterwarnings("ignore")


# Name, training datasets, test dataset - shared by the RF and CNN loops.
EXPERIMENTS = [
    ("train_clean__test_A",       ["A"],                   "A"),
    ("train_clean__test_B25",     ["A"],                   "B25"),
    ("train_clean__test_B50",     ["A"],                   "B50"),
    ("train_clean__test_B75",     ["A"],                   "B75"),
    ("train_augmented__test_A",   ["A", "B25", "B50", "B75"], "A"),
    ("train_augmented__test_B25", ["A", "B25", "B50", "B75"], "B25"),
    ("train_augmented__test_B50", ["A", "B25", "B50", "B75"], "B50"),
    ("train_augmented__test_B75", ["A", "B25", "B50", "B75"], "B75"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_meta",   required=True)
    p.add_argument("--aircraft_meta", required=True)
    p.add_argument("--audio_root",    required=True)
    p.add_argument("--b25_meta",      required=True)
    p.add_argument("--b50_meta",      required=True)
    p.add_argument("--b75_meta",      required=True)
    p.add_argument("--b25_root",      required=True)
    p.add_argument("--b50_root",      required=True)
    p.add_argument("--b75_root",      required=True)
    p.add_argument("--output_dir",    default="results")
    p.add_argument("--rf_only",  action="store_true",
                   help="Skip CNN - useful for quick testing")
    p.add_argument("--cnn_only", action="store_true",
                   help="Skip RF - run CNN only")
    return p.parse_args()


def _build_all_metas(args: argparse.Namespace) -> dict:
    master = build_master_meta(args.sample_meta, args.aircraft_meta)
    master["audio_root"] = args.audio_root
    clean_bg = master[master["class"] == 0].copy()

    b25 = make_noisy_meta(args.b25_meta, args.aircraft_meta, args.b25_root, clean_bg)
    b50 = make_noisy_meta(args.b50_meta, args.aircraft_meta, args.b50_root, clean_bg)
    b75 = make_noisy_meta(args.b75_meta, args.aircraft_meta, args.b75_root, clean_bg)

    return {"A": master, "B25": b25, "B50": b50, "B75": b75}


def _precompute(all_metas: dict, feature_type: str) -> dict:
    cache = {}
    for name, meta in all_metas.items():
        feats, valid_meta = precompute_features(meta, feature_type, desc=name)
        cache[name] = (valid_meta, feats)
    return cache


def _run_all(cache: dict, feature_type: str, output_dir: Path) -> list:
    summaries = []
    for exp_name, train_keys, test_key in EXPERIMENTS:
        print(f"\n  {exp_name}")
        summaries.append(run_experiment(
            exp_name,
            train_datasets=[cache[k] for k in train_keys],
            test_dataset=cache[test_key],
            feature_type=feature_type,
            output_dir=output_dir,
        ))
    return summaries


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.rf_only:
        print("\n[0/5] Checking for CUDA GPU ...")
        configure_gpu()

    print("\n[1/5] Building metadata ...")
    all_metas = _build_all_metas(args)

    if not args.cnn_only:
        print("\n[2/5] Extracting MFCC features (Random Forest) ...")
        mfcc_cache = _precompute(all_metas, "mfcc")
    else:
        print("\n[2/5] Skipping MFCC extraction (--cnn_only)")
        mfcc_cache = {}

    if not args.rf_only:
        print("\n[3/5] Extracting Mel spectrogram features (CNN) ...")
        mel_cache = _precompute(all_metas, "mel")
    else:
        print("\n[3/5] Skipping Mel extraction (--rf_only)")
        mel_cache = {}

    all_results = {"rf": [], "cnn": []}

    if not args.cnn_only:
        print("\n[4/5] Running Random Forest experiments ...")
        all_results["rf"] = _run_all(mfcc_cache, "mfcc", output_dir)
    else:
        print("\n[4/5] Skipping Random Forest (--cnn_only)")

    if not args.rf_only:
        print("\n[5/5] Running CNN experiments ...")
        all_results["cnn"] = _run_all(mel_cache, "mel", output_dir)
    else:
        print("\n[5/5] Skipping CNN (--rf_only)")

    summary_path = output_dir / "all_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All done! Results saved to: {output_dir}/")
    print(f"  Master summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
