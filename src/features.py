"""
Audio feature extraction for the pipeline.

Two feature flavours:
  - mel: 64-bin log-mel spectrogram (CNN input).
  - mfcc: 40 MFCCs, mean+std aggregated into an 80-d flat vector (RF input).

Audio loading pads / truncates to a fixed 10-second duration at 16 kHz
so every clip produces features of identical shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import get_audio_path


# Signal / feature parameters. Changing any of these invalidates cached
# features and any previously-trained CNN (shape will differ).
SR         = 16000
DURATION   = 10
N_MELS     = 64
N_FFT      = 1024
HOP_LENGTH = 512
N_MFCC     = 40


def load_audio(path: Path) -> np.ndarray:
    import librosa
    import soundfile as sf
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
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    import librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def precompute_features(meta: pd.DataFrame,
                        feature_type: str,
                        desc: str = "") -> tuple:
    """
    Extract features for every row in meta.
    Returns (features_list, filtered_meta) dropping any rows that fail.
    """
    features      = []
    valid_indices = []
    for idx, row in tqdm(meta.iterrows(), total=len(meta),
                         desc=f"  Extracting {feature_type} [{desc}]"):
        path = get_audio_path(row)
        try:
            audio = load_audio(path)
            feat  = extract_mel(audio) if feature_type == "mel" else extract_mfcc(audio)
            features.append(feat)
            valid_indices.append(idx)
        except Exception as e:
            print(f"    [SKIP] {path}: {e}")
    return features, meta.loc[valid_indices].reset_index(drop=True)
