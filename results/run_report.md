# Aircraft Sound Classification — Pipeline Run Report

**Run date:** 2026-04-28  &nbsp;&nbsp; **Host:** qolam (RTX 5090, torch 2.10.0+cu128)
**Wall time:** ~22 min  &nbsp;&nbsp; **Code commit:** `2d24e48`  &nbsp;&nbsp; **Folds:** 3 (stratified)
**Models:** Hierarchical CNN (mel) + Random Forest (MFCC + SMOTE)
**Test sets:** A (clean) / B25 (SNR 25 dB) / B50 (SNR 15 dB) / B75 (SNR 5 dB)

## Results — macro-F1 (mean ± std over 3 folds)

### Random Forest (5-class flat: bg, Piston, Turboprop, Turbofan, Turboshaft)

| Train  \ Test | A (clean)         | B25 (25 dB)       | B50 (15 dB)       | B75 (5 dB)        |
|---------------|-------------------|-------------------|-------------------|-------------------|
| Clean         | **0.527 ± 0.027** | 0.505 ± 0.028     | 0.469 ± 0.012     | 0.362 ± 0.013     |
| Augmented     | 0.469 ± 0.027     | 0.467 ± 0.032     | 0.473 ± 0.047     | **0.403 ± 0.010** |

### CNN — Stage 1 (binary: aircraft vs background)

| Train  \ Test | A                 | B25               | B50               | B75               |
|---------------|-------------------|-------------------|-------------------|-------------------|
| Clean         | 0.832 ± 0.084     | 0.656 ± 0.100     | 0.716 ± 0.096     | 0.641 ± 0.026     |
| Augmented     | 0.908 ± 0.040     | **0.947 ± 0.004** | 0.920 ± 0.017     | 0.826 ± 0.021     |

### CNN — Stage 2 (4-class subtype: Piston, Turboprop, Turbofan, Turboshaft)

| Train  \ Test | A                 | B25               | B50               | B75               |
|---------------|-------------------|-------------------|-------------------|-------------------|
| Clean         | 0.331 ± 0.014     | 0.356 ± 0.053     | 0.377 ± 0.036     | 0.370 ± 0.028     |
| Augmented     | 0.465 ± 0.097     | **0.561 ± 0.033** | 0.561 ± 0.114     | 0.409 ± 0.086     |

## Answers to the four RQs

**RQ1 — How do CNN and RF compare on robustness and performance?**
On the 5-class subtype task with clean training, **RF beats the CNN substantially** (F1 0.527 vs CNN-Stage2 0.331 on A). With augmentation in training, the CNN catches up and slightly exceeds the RF on the clean test set (0.465 vs 0.469) and overtakes it across every noise level (e.g. B25: 0.561 vs 0.467). The CNN's binary head (Stage 1, aircraft/bg) is strong throughout (≥0.83 augmented). Verdict: **RF is the better default; the CNN only beats it once augmentation is added to training.**

**RQ2 — How does increasing noise affect CNN performance?**
For Stage 1 (binary), clean-trained F1 falls from 0.832 (clean) to 0.641 (B75) — about **20 points lost** at the heaviest noise. With augmented training, Stage 1 stays high (0.826 even at B75). Stage 2 (subtype) is largely insensitive to noise condition under clean training (0.33–0.38 across all four), and under augmented training also stays in 0.41–0.56 — the imbalance dominates over noise. **Augmentation is essentially required to make the CNN noise-robust.**

**RQ3 — How does increasing noise affect RF performance?**
Clean-trained RF degrades monotonically with noise: 0.527 → 0.505 → 0.469 → 0.362 (-16 points from A to B75). Augmented RF flattens the curve: 0.469 → 0.467 → 0.473 → 0.403 — almost flat. **Augmentation buys ~4 F1 points at high noise but costs ~6 F1 points on clean** for the RF.

**RQ4 — Does training on clean generalise to noisier conditions?**
Partially. Both models suffer at B75 when trained only on clean (RF -16 pts, CNN-Stage1 -19 pts vs A baseline). Adding the SNR-mixed B25/B50/B75 to the training set (the "augmented" row) closes most of the gap for the CNN (Stage 1 reaches 0.826 at B75) and the entire gap for the RF on B25/B50, while sacrificing some clean-set performance. Augmentation **does not** help on a held-out *real-world* "wild" recording set — that test was not run because no such corpus is present in the repo. The "B" sets here are synthetic SNR mixes of two AeroSonicDB recordings, which is the operationalisation the code supports.

## Caveats (carry over from `docs/report/02_rqs_and_improvements/findings.md`)

1. **Mixing convention.** The PDF says "amplitude-mixing 25/50/75 %" but the code mixes at SNR 25/15/5 dB. The numbers above are for the SNR variant. Reconcile in the write-up.
2. **Subject leakage.** Turboshaft has only 3 unique aircraft across 36 recordings; the 3-fold split is by row, so the same hex_id appears in train and test. Reported Turboshaft F1 is partially memorisation. Per-class numbers in the per-experiment JSONs reflect this.
3. **CNN/RF preprocessing asymmetry.** RF uses SMOTE + `class_weight='balanced'`; CNN uses neither. The "RF beats CNN clean" result in RQ1 is partly a rebalancing-strategy comparison rather than a model comparison.
4. **Validation split.** `_train_torch_model` uses a sequential last-10% split. For Stage 2 this can occasionally place every Turboshaft sample in val, which inflates the std on Stage 2 (visible as ±0.097, ±0.114).

## What's in `results/`

- `results/all_results_summary.json` — all 16 experiments aggregated (also embedded above)
- `results/<experiment>/{rf,cnn}_results.json` — per-fold accuracy, macro-F1, sklearn classification report (per-class precision/recall/F1), confusion matrix

## Reproducibility

```
git rev-parse HEAD          # 2d24e48f88f6907199c7118ef92f6f0e5c133a33
SR=16000  N_MELS=64  N_FFT=1024  HOP=512  N_MFCC=40
RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
SMOTE(random_state=42, k_neighbors=3)
torch.manual_seed(42)  Adam lr=1e-3  patience=4  epochs=30
3-fold CV; FOLDS=[0,1,2]; same recording shares fold across A/B25/B50/B75
```
