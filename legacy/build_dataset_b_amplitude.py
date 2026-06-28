"""
build_dataset_b.py
==================
Generates noise-augmented datasets (B25, B50, B75) from AeroSonicDB.

Strategy:
  - For each aircraft sample (class=1) in a fold, randomly select a background
    sample (class=0) from the *same fold* to use as noise.
  - Mix: mixed = aircraft + alpha * noise
  - Clips to [-1, 1] to prevent digital distortion.
  - Saves output WAVs preserving fold structure.
  - Writes a new metadata CSV for each noise level.

Output layout:
  output_root/
    B25/
      fold1/  ...noisy WAVs...
      fold2/  ...
      ...
    B50/  (same)
    B75/  (same)
    B25_meta.csv
    B50_meta.csv
    B75_meta.csv

Usage:
  python build_dataset_b.py \
      --meta path/to/sample_meta.csv \
      --audio_root path/to/audio_files \
      --output_root path/to/output \
      --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ── Noise levels to generate ──────────────────────────────────────────────────
NOISE_LEVELS = {
    "B25": 0.25,
    "B50": 0.50,
    "B75": 0.75,
}


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV file as a float32 array. Resamples if needed (requires librosa)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)        
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError(
                f"Sample rate mismatch ({sr} vs {target_sr}) and librosa not installed. "
                "Install librosa or ensure all files are already 16 kHz."
            )
    return audio


def match_length(aircraft: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Tile or trim noise to match aircraft length."""
    n_a, n_n = len(aircraft), len(noise)
    if n_n == 0:
        return np.zeros_like(aircraft)
    if n_n < n_a:
        repeats = int(np.ceil(n_a / n_n))
        noise = np.tile(noise, repeats)
    return noise[:n_a]


def mix(aircraft: np.ndarray, noise: np.ndarray, alpha: float) -> np.ndarray:
    """Amplitude-based mixing: mixed = aircraft + alpha * noise, clipped to [-1,1]."""
    noise = match_length(aircraft, noise)
    mixed = aircraft + alpha * noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def build_dataset(
    meta: pd.DataFrame,
    audio_root: Path,
    output_root: Path,
    dataset_name: str,
    alpha: float,
    rng: random.Random,
) -> pd.DataFrame:
    """
    Build one noise-augmented dataset.

    Returns a metadata DataFrame for the new dataset (same schema as input,
    with an extra `source_noise_file` column for reproducibility).
    """
    out_dir = output_root / dataset_name
    rows = []

    aircraft = meta[meta["class"] == 1].copy()
    background = meta[meta["class"] == 0].copy()

    # Pre-index background samples by fold for fast lookup
    bg_by_fold: dict[int, list[str]] = (
        background.groupby("fold")["filename"].apply(list).to_dict()
    )

    for _, row in tqdm(aircraft.iterrows(), total=len(aircraft), desc=dataset_name):
        fold = row["fold"]
        src_path = audio_root / f"fold_{row['fold']}" / str(row['class']) / row["filename"]

        # Pick a random background sample from the same fold
        candidates = bg_by_fold.get(fold, [])
        if not candidates:
            # Fallback: use any fold (should not happen in a balanced dataset)
            candidates = background["filename"].tolist()
        noise_filename = rng.choice(candidates)
        noise_row = background[background["filename"] == noise_filename].iloc[0]
        noise_path = audio_root / f"fold_{noise_row['fold']}" / str(noise_row['class']) / noise_filename

        # Load and mix
        try:
            a_audio = load_audio(src_path)
            n_audio = load_audio(noise_path)
        except Exception as e:
            print(f"  [SKIP] {src_path.name}: {e}")
            continue

        mixed_audio = mix(a_audio, n_audio, alpha)

        # Save to output directory mirroring fold structure
        out_fold_dir = out_dir / f"fold_{fold}" / "1"
        out_fold_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"fold_{fold}/1/{src_path.name}"
        out_path = out_dir / out_filename
        sf.write(out_path, mixed_audio, 16000)

        # Record metadata row
        new_row = row.to_dict()
        new_row["filename"] = out_filename
        new_row["source_noise_file"] = noise_filename
        rows.append(new_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build AeroSonicDB noise-augmented datasets.")
    parser.add_argument("--meta",        required=True, help="Path to sample_meta.csv")
    parser.add_argument("--audio_root",  required=True, help="Root folder containing audio files")
    parser.add_argument("--output_root", required=True, help="Where to write B25/B50/B75 folders")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    meta = pd.read_csv(args.meta)
    audio_root = Path(args.audio_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print(f"Loaded metadata: {len(meta)} samples")
    print(f"  Aircraft (class=1): {(meta['class']==1).sum()}")
    print(f"  Background (class=0): {(meta['class']==0).sum()}")
    print(f"  Folds: {sorted(meta['fold'].unique())}")
    print()

    for dataset_name, alpha in NOISE_LEVELS.items():
        print(f"Building {dataset_name} (alpha={alpha}) ...")
        # Use a fresh but deterministic RNG per level so order doesn't matter
        level_rng = random.Random(args.seed + int(alpha * 100))
        new_meta = build_dataset(
            meta, audio_root, output_root, dataset_name, alpha, level_rng
        )
        meta_path = output_root / f"{dataset_name}_meta.csv"
        new_meta.to_csv(meta_path, index=False)
        print(f"  Saved {len(new_meta)} samples → {meta_path}\n")

    print("Done. Output structure:")
    for name in NOISE_LEVELS:
        print(f"  {output_root}/{name}/  +  {output_root}/{name}_meta.csv")


if __name__ == "__main__":
    main()
