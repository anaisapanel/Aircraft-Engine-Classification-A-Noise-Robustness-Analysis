# AeroSonicDB - Aircraft Sound Classification under Background Noise

Cross-dataset evaluation of aircraft sound classifiers (CNN + Random Forest)
trained and tested across four noise conditions: clean (A), and three
SNR-mixed variants (B25 / B50 / B75).

## Repository layout

```
sound_final/
├── src/                                # Active pipeline (canonical)
│   ├── restratify_folds.py             # 1. Stratified 3-fold split
│   ├── build_dataset_b_snr.py          # 2. SNR-based noise augmentation
│   └── train_and_evaluate.py           # 3. Train + cross-eval (PyTorch)
├── tests/                              # 61 tests total
│   ├── test_train_and_evaluate.py      #   33 tests: CNN architecture / training
│   └── test_metadata.py                #   28 tests: 20 metadata + 8 audio-on-disk
├── data/
│   ├── sample_meta_stratified_snr.csv  # 9433 rows, 3 stratified folds (tracked)
│   ├── aircraft_meta_new.csv           # 583 aircraft records (Engtype, etc.) (tracked)
│   ├── *.zip                           # Raw archives (gitignored, see below)
│   ├── audio/                          # Extracted clean corpus (gitignored)
│   │   └── fold_{0..4}/{0,1}/*.wav     #   ~9434 WAVs after unzipping fold_N zips
│   └── output_snr/                     # Extracted SNR-augmented corpus (gitignored)
│       ├── B{25,50,75}/fold_{0..2}/1/*.wav   # 3 × 3057 noisy aircraft WAVs
│       └── B{25,50,75}_meta.csv              # per-variant metadata
├── UNSEEN_DATA/                        # Real-world aircraft recordings (independent collection)
│   ├── sample_meta.csv                 # Metadata for unseen recordings
│   ├── aircraft_meta.csv               # Aircraft type information
│   └── *.wav                           # 31 real-world aircraft recordings from Amsterdam Airport
├── results/                            # CNN and RF model results on AeroSonicDB
│   ├── train_clean__test_A/            # Clean training, tested on clean audio
│   ├── train_clean__test_B25/          # Clean training, tested on 25 dB SNR
│   ├── train_clean__test_B50/          # Clean training, tested on 15 dB SNR
│   ├── train_clean__test_B75/          # Clean training, tested on 5 dB SNR
│   ├── train_augmented__test_A/        # Augmented training, tested on clean audio
│   ├── train_augmented__test_B25/      # Augmented training, tested on 25 dB SNR
│   ├── train_augmented__test_B50/      # Augmented training, tested on 15 dB SNR
│   ├── train_augmented__test_B75/      # Augmented training, tested on 5 dB SNR
│   └── all_results_summary.json        # Aggregated results across all experiments
├── results_unseen_1/                   # Model generalization results on real-world unseen data
│   ├── train_clean__test_A/            # Models trained on clean AeroSonicDB, tested on unseen
│   ├── train_augmented__test_A/        # Models trained on augmented AeroSonicDB, tested on unseen
│   └── unseen_results.json             # Summary of synthetic-to-real generalization gap
├── legacy/                             # Pre-stratified amplitude-mixing pipeline
│   ├── build_dataset_b_amplitude.py    # was build_dataset_b.py (root)
│   └── train_and_evaluate_tf_keras.py  # was train_and_evaluate_new.py
├── requirements.txt                    # Lower-bound deps for the active pipeline
├── .gitignore                          # Caches, env dirs
└── README.md                           # This file
```

### Why each folder exists

**`src/`** - every script that participates in the canonical SNR + 3-fold
PyTorch pipeline lives here, in execution order. Anything outside `src/`
either supports it (`tests/`, `data/`) or is historical implementation
(`legacy/`). If a new researcher asks "what does this repo run
when I follow the README?", the answer is exactly the three files in
`src/` and nothing else.

**`tests/`** - pytest suite, runnable on a laptop with no audio data.
The CNN tests use synthetic spectrograms; the metadata tests use the
shipped CSVs. Together they guard against silent regressions in
architecture, training loop, and the column contracts between the three
`src/` scripts. Audio-dependent integration tests are deliberately
out-of-scope here (the audio corpus is not in the repo).

**`data/`** - mixed tracked + local-only payload:
- **Tracked** (small, version-controlled): the stratified sample metadata
  (output of `restratify_folds.py`, kept so nobody has to recompute it)
  and the aircraft metadata (`Engtype` per `hex_id`, sourced from
  AeroSonicDB).
- **Gitignored** (bulk audio, local only): `data/*.zip` hold the raw
  AeroSonicDB clean corpus (one zip per original fold) and the
  pre-built SNR-augmented corpus. After unzipping they materialise as
  `data/audio/fold_{0..4}/{0,1}/*.wav` (clean) and
  `data/output_snr/B{25,50,75}/fold_{0..2}/1/*.wav` (noisy) plus the
  three `B*_meta.csv` files. The zips, the extracted trees, and
  `results/` together reach ~26 GB - far larger than GitHub allows -
  which is why they are excluded from version control.

**`legacy/`** - earlier versions kept verbatim so reviewers can trace
how the pipeline evolved, but quarantined so nobody runs them by
accident. They target the original 5-fold non-stratified split with
amplitude-based mixing and were superseded by the SNR pipeline (commits
`0cfa5f1`, `9cc5af4`, `8fcc778`). They are **not compatible** with the
current stratified metadata - running them against
`data/sample_meta_stratified_snr.csv` would dereference the wrong fold
directories. They have no imports from `src/` and `src/` has no imports
from them; deleting `legacy/` would not affect the active pipeline.

**`UNSEEN_DATA/`** - Real-world aircraft recordings collected independently 
from Amsterdam Airport Schiphol. Contains 31 commercial aircraft (Boeing, 
Airbus, Embraer) recorded during landing operations in natural acoustic 
environments. Includes metadata files (`sample_meta.csv`, `aircraft_meta.csv`) 
describing aircraft types and recording characteristics. Used for evaluating 
synthetic-to-real generalization, testing whether models trained on AeroSonicDB 
transfer to real deployment conditions.

**`results/`** - Aggregated results from training CNN and Random Forest models 
on AeroSonicDB across eight train-test conditions (2 training regimes × 4 test 
noise levels). Each subdirectory contains model outputs (`rf_results.json`, 
`cnn_results.json`) for both architectures, with macro-F1 scores, confusion 
matrices, and per-fold metrics. `all_results_summary.json` aggregates across 
all experiments for analysis.

**`results_unseen_1/`** - Generalization results from models trained on AeroSonicDB 
(both clean and augmented regimes) and evaluated on the independent UNSEEN_DATA 
dataset. Documents the synthetic-to-real generalization gap, showing whether lab 
robustness translates to deployment readiness.

**Top-level files** - `requirements.txt` lists only what the active
scripts in `src/` actually import; `.gitignore` excludes Python /
test / type / lint / coverage caches, editor cruft, and environments;
`README.md` is what you are reading.

## Pipeline (canonical)

```
                     ┌─────────────────────────────┐
data/aircraft_meta_new.csv ──┐                     │
                             ▼                     │
data/sample_meta_new.csv → restratify_folds.py     │ Stratified 3-fold by Engtype
   (5-fold original)         │                     │ (Turboshaft only has 36 samples
                             ▼                     │  → 5-fold gave 1 per fold)
              data/sample_meta_stratified_snr.csv  │
                             │                     │
                             ▼                     │
        ┌── build_dataset_b_snr.py ── audio mix at SNR ──┐
        │   (B25 = 25 dB, B50 = 15 dB, B75 = 5 dB)       │
        ▼                                                 ▼
   B25/, B50/, B75/ (WAV)              B25_meta.csv, B50_meta.csv, B75_meta.csv
                             │
                             ▼
                  train_and_evaluate.py
              ┌─────────────────────────────────┐
              │  CNN (mel) - Hierarchical:      │
              │   Stage 1: aircraft vs bg       │
              │   Stage 2: 4-class subtype      │
              │  RF  (MFCC) - Flat 5-class      │
              │  3-fold CV × 8 train/test pairs │
              └─────────────────────────────────┘
                             │
                             ▼
                  results/<exp>/{rf,cnn}_results.json
```

## Reproduce

### 1. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 2. Drop the audio zips into `data/`

The audio corpus is gitignored (too large) and must be placed
locally. You need **six files** from the AeroSonicDB distribution:

```
data/
├── aircraft_meta_new.csv              ← tracked (in the repo)
├── sample_meta_stratified_snr.csv     ← tracked (in the repo)
├── fold_0*.zip                        ← you provide (clean, original fold 0)
├── fold_1*.zip                        ← you provide (clean, original fold 1)
├── fold_2*.zip                        ← you provide (clean, original fold 2)
├── fold_3*.zip                        ← you provide (clean, original fold 3)
├── fold_4*.zip                        ← you provide (clean, original fold 4)
└── output_snr.zip                     ← you provide (pre-built SNR-augmented)
```

Filename rules (the extraction glob is tolerant, so timestamp suffixes
are fine):

| File                  | Required name pattern                  | What's inside                                   |
|-----------------------|----------------------------------------|-------------------------------------------------|
| Clean fold N          | `fold_N*.zip` (one per `N∈{0..4}`)     | A `fold_N/` root with `0/*.wav` and `1/*.wav`   |
| SNR-augmented corpus  | `output_snr.zip` (exact name)          | `output_snr/B{25,50,75}/fold_{0..2}/1/*.wav` + three `B*_meta.csv` |

(Google Drive exports with names like
`fold_0-20260423T073852Z-3-001.zip` match the `fold_N*.zip` glob and
work unchanged. `output_snr.zip` must keep that exact name because
it's unzipped by explicit path.)

### 3. Unpack

```bash
# Clean corpus (one zip per original fold) → data/audio/fold_{0..4}/{0,1}/*.wav
for z in data/fold_*.zip; do unzip -q -o "$z" -d data/audio/; done

# Pre-built SNR-augmented corpus → data/output_snr/B{25,50,75}/...
unzip -q -o data/output_snr.zip -d data/

# Strip macOS zip cruft
find data -name '__MACOSX' -type d -prune -exec rm -rf {} +
find data -name '.DS_Store' -delete
```

After unpacking, `data/` should look exactly like this (sizes are
approximate):

```
data/
├── aircraft_meta_new.csv              (tracked, 98 KB)
├── sample_meta_stratified_snr.csv     (tracked, 556 KB)
├── fold_0*.zip … fold_4*.zip          (gitignored, ~4 GB total)
├── output_snr.zip                     (gitignored, ~8.5 GB)
├── audio/                             (gitignored, extracted)
│   ├── fold_0/0/*.wav   (1236 WAVs)
│   ├── fold_0/1/*.wav   ( 625 WAVs)
│   ├── fold_1/0/*.wav   (1054 WAVs)
│   ├── fold_1/1/*.wav   ( 590 WAVs)
│   ├── fold_2/0/*.wav   (1387 WAVs)
│   ├── fold_2/1/*.wav   ( 605 WAVs)
│   ├── fold_3/0/*.wav   (1528 WAVs)
│   ├── fold_3/1/*.wav   ( 702 WAVs)
│   ├── fold_4/0/*.wav   (1171 WAVs)
│   └── fold_4/1/*.wav   ( 536 WAVs)     ── total 9434 WAVs
└── output_snr/                        (gitignored, extracted)
    ├── B25/fold_{0,1,2}/1/*.wav       (3057 WAVs)
    ├── B50/fold_{0,1,2}/1/*.wav       (3057 WAVs)
    ├── B75/fold_{0,1,2}/1/*.wav       (3057 WAVs)
    ├── B25_meta.csv                   (3057 rows)
    ├── B50_meta.csv                   (3057 rows)
    └── B75_meta.csv                   (3057 rows)
```

Sanity-check the placement before spending time on a training run:

```bash
python3 -m pytest tests/test_metadata.py::TestExtractedCleanAudio \
                  tests/test_metadata.py::TestExtractedNoisyAudio -v
```

The 8 tests pass iff every row in `sample_meta_stratified_snr.csv`
and in each `B*_meta.csv` resolves to an existing WAV. A failure
points at exactly which filename is missing.

Folder-layout notes:

- Clean audio is indexed by the **original** fold (0..4 from
  AeroSonicDB), not the new stratified fold. That mapping is kept in
  the `original_fold` column of the stratified metadata, and
  `get_audio_path()` in `src/train_and_evaluate.py` uses it.
- SNR audio is indexed by the **new stratified** fold (0..2). The
  B*_meta.csv files already encode the relative path
  `fold_X/1/<filename>.wav` in their `filename` column, so the
  B25/B50/B75 subtrees only need to live under `data/output_snr/`.
- The one file `data/audio/fold_1/1/ABA975_2024-02-29_09-49-03_0_1.wav`
  is on disk but not referenced by any metadata row - it's a Hawker 850
  with `Engtype` missing in `aircraft_meta_new.csv`, dropped by
  `restratify_folds.py`. Harmless; leave it in place.

Expected totals after extraction:

| Path                                | Contents                                      |
|-------------------------------------|-----------------------------------------------|
| `data/audio/fold_0..4/{0,1}/*.wav`  | 9434 WAVs (6376 background + 3058 aircraft)   |
| `data/output_snr/B25/fold_0..2/1/`  | 3057 noisy aircraft WAVs                      |
| `data/output_snr/B50/fold_0..2/1/`  | 3057 noisy aircraft WAVs                      |
| `data/output_snr/B75/fold_0..2/1/`  | 3057 noisy aircraft WAVs                      |
| `data/output_snr/B{25,50,75}_meta.csv` | 3057 rows each                             |

(The clean corpus has 1 more aircraft WAV than the pipeline references -
hex `ABA975`, a Hawker 850 whose `Engtype` is missing in
`aircraft_meta_new.csv`. `restratify_folds.py` drops that row; the file
is a harmless orphan on disk.)

### 4. (Optional) Restratify

Already done - the committed `data/sample_meta_stratified_snr.csv` is
the output. Only rerun if you receive a new raw `sample_meta_new.csv`
from AeroSonicDB:

```bash
python3 src/restratify_folds.py \
    --sample_meta   data/sample_meta_new.csv \
    --aircraft_meta data/aircraft_meta_new.csv \
    --output        data/sample_meta_stratified_snr.csv
```

### 5. (Optional) Rebuild the SNR-augmented corpus

Already provided by `data/output_snr.zip`. Only rerun if you change the
SNR levels or want to verify reproducibility from clean audio:

```bash
python3 src/build_dataset_b_snr.py \
    --meta        data/sample_meta_stratified_snr.csv \
    --audio_root  data/audio \
    --output_root data/output_snr \
    --seed        42
```

### 6. Train and evaluate

8 experiments × 2 models × 3 folds:

```bash
python3 src/train_and_evaluate.py \
    --sample_meta   data/sample_meta_stratified_snr.csv \
    --aircraft_meta data/aircraft_meta_new.csv \
    --audio_root    data/audio \
    --b25_meta      data/output_snr/B25_meta.csv \
    --b50_meta      data/output_snr/B50_meta.csv \
    --b75_meta      data/output_snr/B75_meta.csv \
    --b25_root      data/output_snr/B25 \
    --b50_root      data/output_snr/B50 \
    --b75_root      data/output_snr/B75 \
    --output_dir    results
```

`--rf_only` skips the CNN; `--cnn_only` skips the RF.

## Tests

```bash
python3 -m pytest tests/ -v
```

Expected: **61 passed** when the audio corpus is extracted;
**53 passed + 8 skipped** when it is not (the 8 audio-on-disk tests
auto-skip if `data/audio/` or `data/output_snr/` is absent, so the
suite runs cleanly on a fresh clone without the zips).

| File                          | Tests | Scope                                      |
|-------------------------------|-------|--------------------------------------------|
| `test_train_and_evaluate.py`  | 33    | CNN architecture, training, early stopping |
| `test_metadata.py`            | 28    | 20 metadata-pipeline + 8 audio-on-disk     |

### How to run individual tests

All commands assume you are in the repo root and have `pytest` installed
(it ships with `requirements.txt`). `-v` adds per-test output; drop it
for a one-line summary.

**By file** - pick one of the two test files:

```bash
python3 -m pytest tests/test_metadata.py -v             # 20 metadata tests
python3 -m pytest tests/test_train_and_evaluate.py -v   # 33 CNN tests
```

**By class** - each `TestXxx` class groups related tests:

```bash
# Metadata layer (tests/test_metadata.py)
python3 -m pytest tests/test_metadata.py::TestShippedMetadataSchema -v   # 6  shipped CSV schema
python3 -m pytest tests/test_metadata.py::TestBuildMasterMeta -v         # 8  build_master_meta round-trip
python3 -m pytest tests/test_metadata.py::TestMakeNoisyMeta -v           # 3  make_noisy_meta concat
python3 -m pytest tests/test_metadata.py::TestGetAudioPath -v            # 3  clean vs noisy path routing
python3 -m pytest tests/test_metadata.py::TestExtractedCleanAudio -v     # 2  every sample_meta row resolves under data/audio/ (skipped if not extracted)
python3 -m pytest tests/test_metadata.py::TestExtractedNoisyAudio -v     # 6  every B-meta row resolves under data/output_snr/ (skipped if not extracted)

# CNN model (tests/test_train_and_evaluate.py)
python3 -m pytest tests/test_train_and_evaluate.py::TestArchitecture -v    # 9  layer inventory, channel counts, param count
python3 -m pytest tests/test_train_and_evaluate.py::TestForwardShapes -v   # 8  output shapes (parametrised over device + input shape)
python3 -m pytest tests/test_train_and_evaluate.py::TestDevice -v          # 3  CPU/CUDA placement
python3 -m pytest tests/test_train_and_evaluate.py::TestTraining -v        # 7  loss decreases, overfits tiny set, seed reproducibility
python3 -m pytest tests/test_train_and_evaluate.py::TestEarlyStopping -v   # 3  validation split semantics, patience, best-weight restore
python3 -m pytest tests/test_train_and_evaluate.py::TestConfigureGpu -v    # 1  configure_gpu() returns a torch.device
python3 -m pytest tests/test_train_and_evaluate.py::TestRunCnnFold -v      # 2  end-to-end run_cnn_fold on synthetic data
```

**By single test** - use `::TestClass::test_name`:

```bash
python3 -m pytest tests/test_train_and_evaluate.py::TestArchitecture::test_parameter_count_is_stable -v
python3 -m pytest tests/test_metadata.py::TestGetAudioPath::test_clean_filename_uses_original_fold_for_path -v
```

**By keyword** - `-k` matches any substring of the test ID:

```bash
python3 -m pytest tests/ -k "early_stopping" -v   # only TestEarlyStopping tests
python3 -m pytest tests/ -k "fold or path" -v     # both fold-related and get_audio_path tests
python3 -m pytest tests/ -k "not Training" -v     # everything except TestTraining
```

**Speed control** - the slowest tests are `TestTraining` and
`TestRunCnnFold` (they actually train tiny networks). For a 1-second
sanity check, skip them:

```bash
python3 -m pytest tests/ -k "not Training and not RunCnnFold"
```

**Useful flags** - `-x` stops at first failure; `--tb=short` truncates
tracebacks; `-q` quiet mode for CI; `--co` lists tests without running
(use to confirm collection equals 53):

```bash
python3 -m pytest tests/ --co -q                  # list all 53 tests
python3 -m pytest tests/ -x --tb=short            # fail fast with short traceback
python3 -m pytest tests/ -q                       # quiet (one char per test)
```

## Experimental matrix

Eight (train, test) pairs, each evaluated with both the CNN and the RF:

| Train                  | Test  | Purpose                              |
|------------------------|-------|--------------------------------------|
| A (clean)              | A     | Baseline                             |
| A (clean)              | B25   | Robustness to unseen noise (low)     |
| A (clean)              | B50   | Robustness to unseen noise (med)     |
| A (clean)              | B75   | Robustness to unseen noise (high)    |
| A + B25 + B50 + B75    | A     | Does augmentation hurt clean perf?   |
| A + B25 + B50 + B75    | B25   | Does augmentation help? (low)        |
| A + B25 + B50 + B75    | B50   | Does augmentation help? (med)        |
| A + B25 + B50 + B75    | B75   | Does augmentation help? (high)       |

When training datasets are combined, the same recording's clean and noisy
versions share a fold ID - preventing leakage between train and test.

## SNR convention

| Dataset | Target SNR | Description           |
|---------|------------|-----------------------|
| B25     | 25 dB      | moderate noise        |
| B50     | 15 dB      | heavy noise           |
| B75     |  5 dB      | very heavy noise      |

(Lower dB = more noise. Names are kept for backward compatibility with
earlier amplitude-based naming where 25/50/75 referred to mixing weights.)

## Unseen Data Evaluation

Beyond synthetic robustness testing, this study includes evaluation on real-world 
unseen aircraft recordings collected near Amsterdam Airport Schiphol. The unseen 
dataset (31 commercial aircraft) enables assessment of the synthetic-to-real 
generalization gap—whether models achieving high robustness on AeroSonicDB + 
synthetic noise actually perform well on real-world operational data.

**Key finding**: Synthetic robustness does not guarantee real-world generalization. 
Models trained on clean AeroSonicDB achieve 0.527 F1 (RF) / 0.331 F1 (CNN Stage 2) 
on synthetic test sets but degrade to 0.158 F1 (RF) / 0.250 F1 (CNN Stage 2) on 
unseen real recordings, revealing that both architectures learn recording context 
rather than generalizable engine acoustics. Results are stored in `results_unseen_1/`.

## Models

| Component | Features            | Architecture                                                                                                           |
|-----------|---------------------|------------------------------------------------------------------------------------------------------------------------|
| CNN       | 64-mel spectrogram  | Conv(32)→BN→Pool→Drop(.25), Conv(64)→BN→Pool→Drop(.25), Conv(128)→BN→GAP→Drop(.5), FC(128). Two heads (binary, 4-class). |
| RF        | 40-MFCC mean+std    | 500 trees, balanced class weight, SMOTE oversampling (k_neighbors=3 for the 90-sample Piston minority).                 |

## Reproducibility

- Seeds are fixed: `restratify_folds.py` uses `random_state=42`;
  `build_dataset_b_snr.py` uses `--seed 42` and an SNR-derived offset per
  level; `train_and_evaluate.py:_train_torch_model` defaults to `seed=42`.
- 3-fold cross-validation; the test fold is held out across **all four**
  datasets simultaneously.
- Validation split (10%) is taken from the training folds only and is the
  trailing slice (matches Keras `validation_split` semantics - see
  `tests/test_train_and_evaluate.py::TestEarlyStopping`).
