"""
build_dataset_b.py
==================
Generates noise-augmented datasets (B25, B50, B75) from AeroSonicDB.

Strategy:
  - For each aircraft sample (class=1), randomly select a background
    sample (class=0) from the *same fold* to use as noise.
  - Mix at a target SNR (dB):
      SNR = 10 * log10(signal_power / noise_power)
      → noise is scaled so the signal-to-noise ratio equals the target.
  - Clips to [-1, 1] to prevent digital distortion.
  - Saves output WAVs preserving fold structure.
  - Writes a new metadata CSV for each noise level.

SNR naming convention:
  B25 → SNR = 25 dB  (low noise — aircraft clearly audible)
  B50 → SNR = 50 dB  (very low noise — barely any background)  (kept for naming compat.)  
  B75 → SNR =  5 dB  (high noise — aircraft hard to hear)

  NOTE: In SNR terms, LOWER dB = MORE noise. So B25 (25 dB) has more noise
  than B50 (50 dB). If you want to rename datasets to reflect noise level
  rather than SNR value, see the NOISE_LEVELS dict below.

IMPORTANT: Pass the stratified metadata (sample_meta_stratified3.csv) so
that output B datasets inherit the correct 3-fold IDs.

Output layout:
  output_root/
    B25/
      fold_0/1/filename.wav
      fold_1/1/filename.wav
      fold_2/1/filename.wav
    B50/  (same structure)
    B75/  (same structure)
    B25_meta.csv
    B50_meta.csv
    B75_meta.csv

Usage:
  python build_dataset_b_snr.py \
      --meta        data/sample_meta_stratified_snr.csv \
      --audio_root  data \
      --output_root data/output_snr \
      [--seed 42]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ── SNR levels (dB) ───────────────────────────────────────────────────────────
# Keys are dataset names, values are target SNR in dB.
# Lower SNR = more noise mixed in.
#   25 dB → moderate noise  (aircraft + noticeable background)
#   15 dB → heavy noise     (background competes with aircraft)
#    5 dB → very heavy noise (background dominates)
# Adjust these values to match your experimental design.
NOISE_LEVELS = {
    "B25": 25.0,   # moderate noise
    "B50": 15.0,   # heavy noise
    "B75":  5.0,   # very heavy noise
}


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV file as float32. Resamples if needed."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError(
                f"Sample rate mismatch ({sr} vs {target_sr}) and librosa "
                "not installed. Install librosa or ensure all files are 16 kHz."
            )
    return audio


def match_length(aircraft: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Tile or trim noise array to match aircraft length."""
    n_a, n_n = len(aircraft), len(noise)
    if n_n == 0:
        return np.zeros_like(aircraft)
    if n_n < n_a:
        repeats = int(np.ceil(n_a / n_n))
        noise   = np.tile(noise, repeats)
    return noise[:n_a]


def mix_at_snr(aircraft: np.ndarray,
               noise: np.ndarray,
               snr_db: float) -> np.ndarray:
    """
    Mix aircraft signal and noise at a target SNR (dB).

    SNR formula:
        SNR_dB = 10 * log10(P_signal / P_noise)

    Rearranging to find the required noise gain:
        P_noise_target = P_signal / 10^(SNR_dB / 10)
        gain = sqrt(P_noise_target / P_noise_actual)
               = RMS_signal / (RMS_noise * 10^(SNR_dB / 20))

    The noise is then scaled by this gain before adding to the signal.
    Result is clipped to [-1, 1] to prevent digital distortion.

    Args:
        aircraft : clean aircraft signal
        noise    : background noise signal (will be length-matched)
        snr_db   : target signal-to-noise ratio in dB
                   lower = more noise (e.g. 5 dB is very noisy)
                   higher = less noise (e.g. 25 dB is moderate)

    Returns:
        Mixed signal as float32, clipped to [-1, 1].
    """
    noise = match_length(aircraft, noise)

    signal_rms = np.sqrt(np.mean(aircraft ** 2)) + 1e-9
    noise_rms  = np.sqrt(np.mean(noise    ** 2)) + 1e-9

    # Scale noise to achieve the target SNR
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))
    noise_scaled     = noise * (target_noise_rms / noise_rms)

    mixed = aircraft + noise_scaled
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(meta: pd.DataFrame,
                  audio_root: Path,
                  output_root: Path,
                  dataset_name: str,
                  snr_db: float,
                  rng: random.Random) -> pd.DataFrame:
    """
    Build one noise-augmented dataset (e.g. B25).

    For each aircraft recording, picks a random background recording from
    the SAME fold as the noise source. This preserves fold integrity —
    noise used in training folds never comes from the test fold.

    Returns a metadata DataFrame (same schema as input meta) with:
      - 'filename' updated to the relative path  (fold_X/1/name.wav)
      - 'snr_db' column showing the target SNR used
      - 'source_noise_file' column for reproducibility

    Background rows are NOT included in the output — the training pipeline
    always uses clean background from Dataset A directly.
    """
    out_dir    = output_root / dataset_name
    aircraft   = meta[meta["class"] == 1].copy()
    background = meta[meta["class"] == 0].copy()

    # Pre-index background filenames by fold for O(1) lookup
    bg_by_fold: dict = (
        background.groupby("fold")["filename"].apply(list).to_dict()
    )

    rows    = []
    skipped = 0

    for _, row in tqdm(aircraft.iterrows(),
                       total=len(aircraft),
                       desc=f"  Building {dataset_name} (SNR={snr_db} dB)"):

        fold     = int(row["fold"])
        original_fold = int(row["original_fold"])
        src_path = (audio_root
                    / f"fold_{original_fold}"
                    / str(int(row["class"]))
                    / row["filename"])

        # Pick a random background sample from the same fold
        candidates = bg_by_fold.get(fold, [])
        if not candidates:
            candidates = background["filename"].tolist()
            print(f"  WARNING: no background samples in fold {fold}, "
                  f"drawing from all folds as fallback")

        noise_filename = rng.choice(candidates)
        noise_row      = background[
            background["filename"] == noise_filename
        ].iloc[0]
        noise_path = (audio_root
                      / f"fold_{int(noise_row['original_fold'])}"
                      / str(int(noise_row["class"]))
                      / noise_filename)

        # Load
        try:
            a_audio = load_audio(src_path)
            n_audio = load_audio(noise_path)
        except Exception as e:
            print(f"    [SKIP] {src_path.name}: {e}")
            skipped += 1
            continue

        # Mix at target SNR
        mixed_audio = mix_at_snr(a_audio, n_audio, snr_db)

        # Save — mirror Dataset A fold structure
        out_fold_dir = out_dir / f"fold_{fold}" / "1"
        out_fold_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"fold_{fold}/1/{src_path.name}"
        out_path     = out_dir / out_filename
        sf.write(str(out_path), mixed_audio, 16000)

        # Record metadata — keep all original columns
        new_row                      = row.to_dict()
        new_row["filename"]          = out_filename
        new_row["snr_db"]            = snr_db
        new_row["source_noise_file"] = noise_filename
        rows.append(new_row)

    if skipped:
        print(f"  Skipped {skipped} files")

    result = pd.DataFrame(rows)

    # Print fold distribution for verification
    print(f"  {dataset_name}: {len(result)} aircraft files written")
    if "Engtype" in result.columns:
        dist = (result.groupby("fold")["Engtype"]
                .value_counts()
                .unstack(fill_value=0))
        print(dist.to_string())
    else:
        print(result["fold"].value_counts().sort_index().to_string())

    return result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build AeroSonicDB noise-augmented datasets B25/B50/B75 "
                    "using SNR-based mixing."
    )
    parser.add_argument("--meta",        required=True,
                        help="Stratified sample meta (sample_meta_stratified3.csv)")
    parser.add_argument("--audio_root",  required=True,
                        help="Root folder of clean Dataset A audio files")
    parser.add_argument("--output_root", required=True,
                        help="Output directory for B25/B50/B75 folders + CSVs")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    meta        = pd.read_csv(args.meta)
    audio_root  = Path(args.audio_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded metadata: {len(meta)} samples")
    print(f"  Aircraft   (class=1): {(meta['class'] == 1).sum()}")
    print(f"  Background (class=0): {(meta['class'] == 0).sum()}")
    print(f"  Folds present: {sorted(meta['fold'].unique())}")

    n_folds = meta["fold"].nunique()
    if n_folds != 3:
        print(f"\n  WARNING: expected 3 folds but found {n_folds}. "
              f"Did you pass sample_meta_stratified3.csv?")

    print(f"\nNoise levels to build:")
    for name, snr in NOISE_LEVELS.items():
        print(f"  {name}: SNR = {snr} dB")
    print()

    for dataset_name, snr_db in NOISE_LEVELS.items():
        print(f"Building {dataset_name} (SNR = {snr_db} dB) ...")
        level_rng = random.Random(args.seed + int(snr_db))
        new_meta  = build_dataset(
            meta, audio_root, output_root,
            dataset_name, snr_db, level_rng
        )
        meta_path = output_root / f"{dataset_name}_meta.csv"
        new_meta.to_csv(meta_path, index=False)
        print(f"  Metadata → {meta_path}\n")

    print("=" * 50)
    print("Done. Output structure:")
    for name in NOISE_LEVELS:
        print(f"  {output_root}/{name}/")
        print(f"  {output_root}/{name}_meta.csv")
    print()
    print("Next step — run training:")
    print("  python train_and_evaluate.py \\")
    print(f"      --sample_meta {args.meta} \\")
    print(f"      --b25_meta    {output_root}/B25_meta.csv \\")
    print(f"      --b50_meta    {output_root}/B50_meta.csv \\")
    print(f"      --b75_meta    {output_root}/B75_meta.csv \\")
    print(f"      --b25_root    {output_root}/B25 \\")
    print(f"      --b50_root    {output_root}/B50 \\")
    print(f"      --b75_root    {output_root}/B75 \\")
    print(f"      --audio_root  {audio_root} \\")
    print(f"      --output_dir  results")


if __name__ == "__main__":
    main()
