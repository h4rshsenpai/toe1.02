"""Train a simple CNN on IDMT-SMT-Drums V2 for drum onset detection.

Classes predicted (3 outputs):
  0 = Kick (BD)
  1 = Snare (SD)
  2 = Hi-Hat (HH / CHH / OHH)

Usage:
    python train_cnn.py
    python train_cnn.py --epochs 50 --batch 512 --out output/drum_cnn.pt
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR  = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
XML_DIR   = DATA_DIR / "annotation_xml"

SR         = 44100
HOP        = 256
N_MELS     = 128
CONTEXT    = 13          # frames of context window (must be odd)
_HALF      = CONTEXT // 2

# Drum class index → class name (used at inference time)
CLASSES    = ["kick", "snare", "hihat"]
N_CLASSES  = len(CLASSES)

# IDMT label → class index
LABEL_MAP: dict[str, int] = {
    "KD":  0,   # kick
    "BD":  0,
    "SD":  1,   # snare
    "HH":  2,   # hi-hat
    "CHH": 2,   # closed hi-hat
    "OHH": 2,   # open hi-hat
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DrumCNN(nn.Module):
    """Compact 4-layer CNN for frame-level multi-label drum onset detection.

    Input:  (B, 1, CONTEXT, N_MELS) — single-channel log-mel context window
    Output: (B, N_CLASSES) — per-class onset probability (sigmoid applied)
    """

    def __init__(self, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                         # → (B, 16, 9, 64)

            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                         # → (B, 32, 9, 32)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((1, 2)),                         # → (B, 64, 9, 16)

            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                 # → (B, 64, 1, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.conv(x)))


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available in this PyTorch environment.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but Apple Metal (MPS) is not available in this PyTorch environment.")
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _parse_xml(xml_path: Path) -> dict[int, list[float]]:
    """Return {class_idx: [onset_times]} from an IDMT #MIX annotation XML."""
    tree = ET.parse(xml_path)
    onsets: dict[int, list[float]] = {i: [] for i in range(N_CLASSES)}
    for event in tree.getroot().findall(".//event"):
        label = event.findtext("instrument", "").strip().upper()
        onset_str = event.findtext("onsetSec", "")
        cls = LABEL_MAP.get(label)
        if cls is not None and onset_str:
            onsets[cls].append(float(onset_str))
    return onsets


def _compute_mel(audio_path: Path) -> np.ndarray:
    """Load audio and return log-mel spectrogram (n_frames, N_MELS)."""
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max).T  # (n_frames, N_MELS)


class IDMTDataset(Dataset):
    """Frame-level binary classification dataset built from IDMT-SMT-Drums V2.

    Each sample is a (CONTEXT × N_MELS) log-mel context window centred on one
    frame, plus a binary label vector of length N_CLASSES.
    """

    def __init__(self, xml_files: list[Path]) -> None:
        self.windows: list[np.ndarray] = []   # (CONTEXT, N_MELS) float32
        self.labels:  list[np.ndarray] = []   # (N_CLASSES,) float32

        for xml_path in tqdm(xml_files, desc="Processing files"):
            stem = xml_path.stem.replace("#MIX", "")
            wav  = AUDIO_DIR / f"{stem}#MIX.wav"
            if not wav.exists():
                continue

            mel    = _compute_mel(wav)          # (n_frames, N_MELS)
            n      = len(mel)
            onsets = _parse_xml(xml_path)

            # Build per-frame binary label matrix
            label_mat = np.zeros((n, N_CLASSES), dtype=np.float32)
            for cls, times in onsets.items():
                for t in times:
                    frame = int(round(t * SR / HOP))
                    if 0 <= frame < n:
                        label_mat[frame, cls] = 1.0

            # Pad mel at boundaries and extract context windows
            padded = np.pad(mel, ((_HALF, _HALF), (0, 0)), mode="edge")
            for i in range(n):
                window = padded[i : i + CONTEXT]           # (CONTEXT, N_MELS)
                self.windows.append(window.astype(np.float32))
                self.labels.append(label_mat[i])

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx]).unsqueeze(0)  # (1, CONTEXT, N_MELS)
        y = torch.from_numpy(self.labels[idx])
        return x, y

    def pos_weights(self) -> torch.Tensor:
        """Per-class pos_weight = n_neg / n_pos for BCEWithLogitsLoss."""
        labels = np.stack(self.labels)           # (N, N_CLASSES)
        pos = labels.sum(axis=0).clip(min=1)
        neg = (1 - labels).sum(axis=0).clip(min=1)
        return torch.tensor(neg / pos, dtype=torch.float32)

    def sample_weights(self) -> list[float]:
        """Per-sample weight: onset frames get higher probability of being sampled."""
        weights = []
        for lbl in self.labels:
            weights.append(10.0 if lbl.any() else 1.0)
        return weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs:     int  = 30,
    batch_size: int  = 256,
    lr:         float = 1e-3,
    out_path:   Path  = Path("output/drum_cnn.pt"),
    val_frac:   float = 0.2,
    device_arg: str = "auto",
) -> None:
    xml_files = sorted(XML_DIR.glob("*#MIX.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No annotation XML files found in {XML_DIR}")

    split     = int(len(xml_files) * (1 - val_frac))
    train_xml = xml_files[:split]
    val_xml   = xml_files[split:]

    print(f"Building training dataset ({len(train_xml)} files) …")
    train_ds = IDMTDataset(train_xml)
    print(f"Building validation dataset ({len(val_xml)} files) …")
    val_ds   = IDMTDataset(val_xml)

    print(f"Train frames: {len(train_ds):,}  |  Val frames: {len(val_ds):,}")

    pos_w   = train_ds.pos_weights()
    sampler = WeightedRandomSampler(train_ds.sample_weights(), num_samples=len(train_ds), replacement=True)

    device = _resolve_device(device_arg)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print(f"Training on {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"CUDA device: {gpu_name}")
    elif device.type == "mps":
        print("Using Apple Metal Performance Shaders (MPS)")

    model     = DrumCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    training_start = time.time()
    epoch_times = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc="Training", leave=False):
            x = x.to(device, non_blocking=pin_memory)
            y = y.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            logits = model.head(model.conv(x))   # raw logits for BCEWithLogitsLoss
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x = x.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)
                logits = model.head(model.conv(x))
                val_loss += criterion(logits, y).item() * len(x)
        val_loss /= len(val_ds)

        scheduler.step(val_loss)
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)
        
        # Calculate time estimates
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - epoch
        estimated_remaining = avg_epoch_time * remaining_epochs
        total_elapsed = epoch_end - training_start
        estimated_total = total_elapsed + estimated_remaining
        
        # Format time strings
        epoch_time_str = f"{epoch_time:.1f}s"
        remaining_str = f"{estimated_remaining/60:.1f}m" if estimated_remaining >= 60 else f"{estimated_remaining:.1f}s"
        total_str = f"{estimated_total/60:.1f}m" if estimated_total >= 60 else f"{estimated_total:.1f}s"
        
        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  epoch={epoch_time_str}  ETA={remaining_str} (total≈{total_str})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            print(f"  ✓ checkpoint saved → {out_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    total_time = time.time() - training_start
    total_str = f"{total_time/60:.1f}m" if total_time >= 60 else f"{total_time:.1f}s"
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    avg_str = f"{avg_epoch_time:.1f}s"
    print(f"Total time: {total_str}  |  Avg per epoch: {avg_str}")
    print(f"Checkpoint: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DrumCNN on IDMT-SMT-Drums V2")
    parser.add_argument("--epochs", type=int,   default=30)
    parser.add_argument("--batch",  type=int,   default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--out",    type=str,   default="output/drum_cnn.pt")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        out_path=Path(args.out),
        device_arg=args.device,
    )
