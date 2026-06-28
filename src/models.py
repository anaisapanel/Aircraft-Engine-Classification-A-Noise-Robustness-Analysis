"""
CNN architecture and training helpers.

Hierarchical classification: a binary head (aircraft vs background)
and a 4-class head (engine subtype) share the same convolutional
trunk definition but are trained independently.

Training matches Keras semantics so results reproduce against the
original Keras-era experiments:
  - Adam, default lr=1e-3
  - sequential validation split (last val_split fraction)
  - early stopping with patience, restoring best weights
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_gpu() -> torch.device:
    """Check for CUDA availability and return the device the CNN will use."""
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"  [GPU] CUDA available - {n} device(s) detected "
              f"(torch {torch.__version__}, CUDA {torch.version.cuda}):")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            print(f"        • cuda:{i} {props.name} "
                  f"({props.total_memory / 1e9:.1f} GB)")
        return torch.device("cuda")
    print("  [GPU] No CUDA GPU detected - CNN will run on CPU.")
    return torch.device("cpu")


class _CNNBackbone(nn.Module):
    """
    Shared conv trunk matching the original Keras architecture:
        Conv(32)→ReLU→BN→MaxPool→Dropout(0.25)
        Conv(64)→ReLU→BN→MaxPool→Dropout(0.25)
        Conv(128)→ReLU→BN→GlobalAvgPool→Dropout(0.5)
        FC(128)→ReLU
    Produces a 128-d embedding; the head is added by subclasses.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.drop3 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop2(self.pool2(self.bn2(F.relu(self.conv2(x)))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.gap(x).flatten(1)
        x = self.drop3(x)
        return F.relu(self.fc1(x))


class BinaryCNN(nn.Module):
    """Binary aircraft-vs-background classifier (outputs raw logit)."""

    def __init__(self):
        super().__init__()
        self.backbone = _CNNBackbone()
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(-1)


class SubclassCNN(nn.Module):
    """Multi-class aircraft subtype classifier (outputs raw logits)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = _CNNBackbone()
        self.head = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def _train_torch_model(model: nn.Module,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       *,
                       binary: bool,
                       epochs: int = 30,
                       batch_size: int = 32,
                       val_split: float = 0.1,
                       patience: int = 4,
                       lr: float = 1e-3,
                       device: torch.device | None = None,
                       seed: int | None = 42) -> nn.Module:
    """
    Train `model` with Adam + early stopping (restore best weights).
    Mirrors Keras `model.fit(..., validation_split=0.1,
                              callbacks=[EarlyStopping(patience=4,
                                                       restore_best_weights=True)])`.

    Validation split is sequential (last `val_split` fraction), matching Keras.
    """
    device = device or get_device()
    model = model.to(device)

    if seed is not None:
        torch.manual_seed(seed)

    # Sequential val split (matches Keras validation_split semantics)
    n = len(X_train)
    n_val = max(1, int(round(n * val_split)))
    X_tr = torch.from_numpy(X_train[:-n_val]).float()
    X_val = torch.from_numpy(X_train[-n_val:]).float()
    if binary:
        y_tr = torch.from_numpy(y_train[:-n_val]).float()
        y_val = torch.from_numpy(y_train[-n_val:]).float()
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        y_tr = torch.from_numpy(y_train[:-n_val]).long()
        y_val = torch.from_numpy(y_train[-n_val:]).long()
        loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state: dict | None = None
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()

        # Validation (batched to avoid OOM with large inputs)
        model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                xb = X_val[i:i + batch_size].to(device)
                yb = y_val[i:i + batch_size].to(device)
                total_loss += loss_fn(model(xb), yb).item() * len(xb)
                total_n += len(xb)
        val_loss = total_loss / max(total_n, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch(model: nn.Module,
                   X: np.ndarray,
                   batch_size: int = 32,
                   device: torch.device | None = None) -> np.ndarray:
    """Run the model on `X` and return raw logits as a numpy array."""
    device = device or next(model.parameters()).device
    model.eval()
    X_t = torch.from_numpy(X).float()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i + batch_size].to(device)
            chunks.append(model(xb).cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,))
