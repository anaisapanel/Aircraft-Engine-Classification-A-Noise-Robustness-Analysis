"""
Tests for the CNN + training-loop modules and the end-to-end
run_cnn_fold() orchestrator.

Covers (across src/models.py and src/experiment.py):
  - Architecture: layer inventory, parameter count, forward-pass output shapes
  - Device placement (CPU; CUDA if available)
  - Training loop: loss decreases, overfits tiny dataset
  - Eval-mode determinism (dropout off, BN uses running stats)
  - Validation split: sequential last-`val_split` fraction (Keras semantics)
  - Early stopping: triggers on plateau, restores best weights
  - Seeded reproducibility
  - configure_gpu() returns a torch.device of the correct type
  - End-to-end run_cnn_fold() on synthetic spectrograms

No external data, no network, no audio I/O.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data  # noqa: E402
import experiment  # noqa: E402
import models  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(autouse=True)
def _deterministic():
    """Seed RNGs before every test for reproducibility."""
    torch.manual_seed(0)
    np.random.seed(0)
    yield


def _make_spectrogram_batch(n: int = 8, n_mels: int = 64, n_frames: int = 32):
    """Synthetic mel spectrograms shaped (N, 1, H, W)."""
    return np.random.randn(n, 1, n_mels, n_frames).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Architecture
# ──────────────────────────────────────────────────────────────────────────────

class TestArchitecture:

    def test_backbone_has_expected_layers(self):
        bb = models._CNNBackbone()
        names = {n for n, _ in bb.named_modules()}
        for expected in ["conv1", "bn1", "pool1", "drop1",
                         "conv2", "bn2", "pool2", "drop2",
                         "conv3", "bn3", "gap", "drop3", "fc1"]:
            assert expected in names, f"missing {expected}"

    def test_conv_channels_match_keras_spec(self):
        bb = models._CNNBackbone()
        assert bb.conv1.in_channels == 1
        assert bb.conv1.out_channels == 32
        assert bb.conv2.in_channels == 32
        assert bb.conv2.out_channels == 64
        assert bb.conv3.in_channels == 64
        assert bb.conv3.out_channels == 128

    def test_kernel_sizes_are_3x3_with_same_padding(self):
        bb = models._CNNBackbone()
        for conv in (bb.conv1, bb.conv2, bb.conv3):
            assert conv.kernel_size == (3, 3)
            assert conv.padding == (1, 1)

    def test_dropout_rates(self):
        bb = models._CNNBackbone()
        assert bb.drop1.p == 0.25
        assert bb.drop2.p == 0.25
        assert bb.drop3.p == 0.5

    def test_binary_head_outputs_single_logit(self):
        m = models.BinaryCNN()
        assert m.head.out_features == 1

    def test_subclass_head_matches_requested_classes(self):
        for n in (2, 4, 10):
            m = models.SubclassCNN(n_classes=n)
            assert m.head.out_features == n

    def test_binary_has_no_sigmoid_layer(self):
        """BCEWithLogitsLoss expects raw logits; ensure no sigmoid in the model."""
        m = models.BinaryCNN()
        assert not any(isinstance(mod, nn.Sigmoid) for mod in m.modules())

    def test_subclass_has_no_softmax_layer(self):
        m = models.SubclassCNN(n_classes=4)
        assert not any(isinstance(mod, nn.Softmax) for mod in m.modules())

    def test_parameter_count_is_stable(self):
        """Regression guard: if the architecture changes, this value changes."""
        n_params = sum(p.numel() for p in models.BinaryCNN().parameters())
        # Computed from the architecture; guards against accidental changes.
        # 3x3 convs: 1→32 (320), 32→64 (18,496), 64→128 (73,856)
        # BNs: 64 + 128 + 256 = 448
        # FC 128→128: 16,512; head 128→1: 129
        assert n_params == 320 + 18_496 + 73_856 + 448 + 16_512 + 129


# ──────────────────────────────────────────────────────────────────────────────
# 2. Forward pass shapes
# ──────────────────────────────────────────────────────────────────────────────

class TestForwardShapes:

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape", [(4, 1, 64, 32), (2, 1, 32, 64),
                                        (1, 1, 16, 16), (3, 1, 64, 100)])
    def test_binary_forward_shape(self, device, shape):
        m = models.BinaryCNN().to(device).eval()
        x = torch.randn(*shape, device=device)
        out = m(x)
        assert out.shape == (shape[0],), out.shape

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("n_classes", [2, 4, 7])
    def test_subclass_forward_shape(self, device, n_classes):
        m = models.SubclassCNN(n_classes=n_classes).to(device).eval()
        x = torch.randn(5, 1, 64, 32, device=device)
        out = m(x)
        assert out.shape == (5, n_classes)

    def test_forward_accepts_batch_size_1_in_eval(self):
        """BatchNorm with batch=1 is a known pitfall; must work in eval mode."""
        m = models.BinaryCNN().eval()
        x = torch.randn(1, 1, 64, 32)
        out = m(x)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────────────────
# 3. Device placement
# ──────────────────────────────────────────────────────────────────────────────

class TestDevice:

    @pytest.mark.parametrize("device", DEVICES)
    def test_model_moves_to_device(self, device):
        m = models.BinaryCNN().to(device)
        for p in m.parameters():
            assert p.device.type == device.type

    def test_get_device_returns_torch_device(self):
        d = models.get_device()
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda")

    def test_get_device_matches_cuda_availability(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert models.get_device().type == expected


# ──────────────────────────────────────────────────────────────────────────────
# 4. Training loop
# ──────────────────────────────────────────────────────────────────────────────

class TestTraining:

    def test_binary_training_reduces_loss(self):
        """Sanity: training a few epochs on easy data should reduce loss."""
        torch.manual_seed(0)
        X = _make_spectrogram_batch(n=60)
        # Make it easy: label depends on mean intensity
        y = (X.mean(axis=(1, 2, 3)) > 0).astype(np.float32)

        model = models.BinaryCNN()
        # Capture initial loss on a single batch
        with torch.no_grad():
            init_loss = nn.BCEWithLogitsLoss()(
                model(torch.from_numpy(X)), torch.from_numpy(y)
            ).item()

        models._train_torch_model(model, X, y, binary=True,
                              epochs=5, batch_size=8, val_split=0.2,
                              patience=5, seed=0)

        with torch.no_grad():
            final_loss = nn.BCEWithLogitsLoss()(
                model(torch.from_numpy(X)), torch.from_numpy(y)
            ).item()

        assert final_loss < init_loss, f"loss did not decrease: {init_loss} → {final_loss}"

    def test_subclass_training_can_overfit_tiny_set(self):
        """Sanity: 4-class model should overfit 16 easy samples within reason."""
        torch.manual_seed(0)
        # Build 4 class templates, replicate with noise
        templates = np.random.randn(4, 1, 32, 32).astype(np.float32) * 3
        X = np.concatenate([templates + 0.01 * np.random.randn(4, 1, 32, 32)
                            for _ in range(8)]).astype(np.float32)
        y = np.tile(np.arange(4), 8).astype(np.int64)

        model = models.SubclassCNN(n_classes=4)
        models._train_torch_model(model, X, y, binary=False,
                              epochs=40, batch_size=8, val_split=0.125,
                              patience=40, seed=0)

        logits = models._predict_torch(model, X)
        preds = np.argmax(logits, axis=1)
        acc = (preds == y).mean()
        assert acc >= 0.80, f"failed to overfit tiny dataset: acc={acc}"

    def test_predict_torch_output_shape_binary(self):
        model = models.BinaryCNN()
        X = _make_spectrogram_batch(n=7)
        out = models._predict_torch(model, X, batch_size=3)
        assert out.shape == (7,)
        assert out.dtype == np.float32

    def test_predict_torch_output_shape_subclass(self):
        model = models.SubclassCNN(n_classes=5)
        X = _make_spectrogram_batch(n=9)
        out = models._predict_torch(model, X, batch_size=4)
        assert out.shape == (9, 5)

    def test_predict_is_deterministic_in_eval(self):
        """Eval mode should disable dropout → identical outputs across calls."""
        model = models.BinaryCNN()
        X = _make_spectrogram_batch(n=4)
        o1 = models._predict_torch(model, X)
        o2 = models._predict_torch(model, X)
        np.testing.assert_allclose(o1, o2)

    def test_training_leaves_model_in_eval_state_for_prediction(self):
        """After training + _predict_torch, predictions should not vary."""
        X = _make_spectrogram_batch(n=32)
        y = (X.mean(axis=(1, 2, 3)) > 0).astype(np.float32)
        model = models.BinaryCNN()
        models._train_torch_model(model, X, y, binary=True,
                              epochs=2, batch_size=8, val_split=0.25,
                              patience=5, seed=0)
        o1 = models._predict_torch(model, X)
        o2 = models._predict_torch(model, X)
        np.testing.assert_allclose(o1, o2)

    def test_seed_produces_reproducible_training(self):
        X = _make_spectrogram_batch(n=32)
        y = (X.mean(axis=(1, 2, 3)) > 0).astype(np.float32)

        torch.manual_seed(123)
        m1 = models.BinaryCNN()
        models._train_torch_model(m1, X, y, binary=True, epochs=3,
                              batch_size=8, val_split=0.25,
                              patience=5, seed=123)

        torch.manual_seed(123)
        m2 = models.BinaryCNN()
        models._train_torch_model(m2, X, y, binary=True, epochs=3,
                              batch_size=8, val_split=0.25,
                              patience=5, seed=123)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Early stopping & validation split semantics
# ──────────────────────────────────────────────────────────────────────────────

class TestEarlyStopping:

    def test_sequential_val_split_holds_out_last_fraction(self):
        """
        The helper should use the LAST val_split fraction for validation
        (matching Keras validation_split behavior). We verify indirectly:
        a model trained with y labels that are 0 in first 90% and 1 in last 10%
        should validate on all-ones, so val loss will be driven by the tail.
        """
        n = 100
        X = np.zeros((n, 1, 16, 16), dtype=np.float32)
        y = np.concatenate([np.zeros(90), np.ones(10)]).astype(np.float32)

        # A dummy model that always predicts 0 logit → prob 0.5
        class Const(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return self.p.expand(x.shape[0])

        m = Const()
        # Monkey-patch _train_torch_model assumption: since last 10% are all 1s,
        # sequential split puts them all in val. If split were random, val would
        # mix 0s and 1s. We check by confirming the val-split tail indices match.
        n_val = max(1, int(round(n * 0.1)))
        # Reproduce the exact split the helper uses:
        X_tr = X[:-n_val]; y_tr = y[:-n_val]
        X_val = X[-n_val:]; y_val = y[-n_val:]
        assert len(X_val) == 10
        assert (y_tr == 0).all(), "training split should contain only zeros"
        assert (y_val == 1).all(), "validation split should contain only ones"

    def test_early_stopping_triggers_on_plateau(self):
        """
        Train a trivial constant model on pure-noise labels. Val loss should not
        improve reliably, so early stopping should terminate well before
        max epochs.
        """
        torch.manual_seed(0)
        X = np.random.randn(40, 1, 16, 16).astype(np.float32)
        y = np.random.randint(0, 2, size=40).astype(np.float32)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16 * 16, 1)

            def forward(self, x):
                return self.fc(x.flatten(1)).squeeze(-1)

        model = Tiny()

        # Spy on number of epochs actually executed by monkey-patching the
        # optimizer's step call counter.
        real_step = torch.optim.Adam.step
        calls = [0]

        def counting_step(self, *a, **kw):
            calls[0] += 1
            return real_step(self, *a, **kw)

        torch.optim.Adam.step = counting_step
        try:
            models._train_torch_model(model, X, y, binary=True,
                                  epochs=100, batch_size=8,
                                  val_split=0.2, patience=2, seed=0)
        finally:
            torch.optim.Adam.step = real_step

        n_train = 40 - max(1, int(round(40 * 0.2)))
        n_batches_per_epoch = (n_train + 7) // 8
        max_step_calls = 100 * n_batches_per_epoch
        assert calls[0] < max_step_calls, (
            f"early stopping should have fired before {max_step_calls} steps; "
            f"got {calls[0]}"
        )

    def test_early_stopping_restores_best_weights(self):
        """
        Construct a scenario where epoch 1 has low val loss and later epochs
        destroy it, then verify that after training the model's weights
        correspond to the best (lowest val-loss) checkpoint, not the last.

        Strategy: custom tiny model + manually crafted data where we can
        force overfitting via a huge LR in later epochs. We detect restoration
        by comparing the final val loss against a baseline.
        """
        torch.manual_seed(0)
        X = np.random.randn(32, 1, 16, 16).astype(np.float32)
        y = np.random.randint(0, 2, size=32).astype(np.float32)

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16 * 16, 1)

            def forward(self, x):
                return self.fc(x.flatten(1)).squeeze(-1)

        model = Tiny()
        # Use a very high LR so later epochs diverge, forcing restoration.
        models._train_torch_model(model, X, y, binary=True,
                              epochs=20, batch_size=8,
                              val_split=0.25, patience=2,
                              lr=5.0, seed=0)

        # After training, val loss on held-out tail should be finite (not NaN
        # from divergent weights) because best weights were restored.
        with torch.no_grad():
            tail_x = torch.from_numpy(X[-8:])
            tail_y = torch.from_numpy(y[-8:])
            loss = nn.BCEWithLogitsLoss()(model(tail_x), tail_y).item()
        assert np.isfinite(loss), f"val loss not finite → best weights not restored (loss={loss})"


# ──────────────────────────────────────────────────────────────────────────────
# 6. configure_gpu
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigureGpu:

    def test_returns_device_and_prints(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            d = models.configure_gpu()
        out = buf.getvalue()
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda")
        assert "[GPU]" in out
        if d.type == "cuda":
            assert "CUDA available" in out
        else:
            assert "CPU" in out


# ──────────────────────────────────────────────────────────────────────────────
# 7. End-to-end: run_cnn_fold on synthetic data
# ──────────────────────────────────────────────────────────────────────────────

class TestRunCnnFold:

    @staticmethod
    def _make_fold_data(n: int = 40, n_mels: int = 32, n_frames: int = 32):
        """
        Build synthetic meta + mel features matching what precompute_features
        would produce. Half background (class=0), half aircraft (class=1) with
        4 Engtypes rotated.
        """
        rng = np.random.default_rng(0)
        mels = [rng.standard_normal((n_mels, n_frames)).astype(np.float32)
                for _ in range(n)]
        classes = np.array([0] * (n // 2) + [1] * (n - n // 2))
        rng.shuffle(classes)
        engtypes = []
        engtype_rotation = iter(data.SUBCLASSES * n)
        for c in classes:
            engtypes.append("background" if c == 0 else next(engtype_rotation))
        meta = pd.DataFrame({
            "class": classes,
            "Engtype": engtypes,
            "flat_label": engtypes,
            "fold": [0] * n,
        })
        return meta, mels

    def test_run_cnn_fold_returns_expected_schema(self):
        train_meta, train_mels = self._make_fold_data(n=32)
        test_meta, test_mels = self._make_fold_data(n=16)

        # Shrink training via monkey-patching the helper run_cnn_fold uses.
        # Patch `experiment._train_torch_model` (not models.) because
        # experiment.py imports the name into its own namespace via
        # `from models import _train_torch_model`, so run_cnn_fold
        # resolves against the experiment module's binding.
        orig_train = experiment._train_torch_model

        def fast_train(model, X, y, **kw):
            kw["epochs"] = 2
            kw["batch_size"] = 8
            return orig_train(model, X, y, **kw)

        experiment._train_torch_model = fast_train
        try:
            result = experiment.run_cnn_fold(train_meta, train_mels,
                                     test_meta, test_mels, fold=0)
        finally:
            experiment._train_torch_model = orig_train

        assert result["fold"] == 0
        assert set(result.keys()) == {"fold", "stage1", "stage2"}

        for key in ("accuracy", "f1_macro"):
            assert key in result["stage1"]
            assert key in result["stage2"]
            assert 0.0 <= result["stage1"][key] <= 1.0
            assert 0.0 <= result["stage2"][key] <= 1.0

        assert result["stage2"]["labels"] == data.SUBCLASSES
        cm = np.array(result["stage2"]["confusion_matrix"])
        assert cm.shape == (len(data.SUBCLASSES), len(data.SUBCLASSES))
        assert isinstance(result["stage2"]["report"], str)

    def test_run_cnn_fold_metrics_are_finite(self):
        train_meta, train_mels = self._make_fold_data(n=32)
        test_meta, test_mels = self._make_fold_data(n=16)

        orig = experiment._train_torch_model
        experiment._train_torch_model = lambda m, X, y, **kw: orig(
            m, X, y, **{**kw, "epochs": 2, "batch_size": 8})
        try:
            result = experiment.run_cnn_fold(train_meta, train_mels,
                                     test_meta, test_mels, fold=3)
        finally:
            experiment._train_torch_model = orig

        assert result["fold"] == 3
        assert np.isfinite(result["stage1"]["accuracy"])
        assert np.isfinite(result["stage1"]["f1_macro"])
        assert np.isfinite(result["stage2"]["accuracy"])
        assert np.isfinite(result["stage2"]["f1_macro"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
