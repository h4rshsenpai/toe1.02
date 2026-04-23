"""Step 1: Transcribe drum audio to MIDI using DrumCNN."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import torch

from train_cnn import CLASSES, CONTEXT, HOP, N_CLASSES, N_MELS, SR, DrumCNN, _HALF

_HERE = Path(__file__).parent
DRUMS_WAV = _HERE / "output/htdemucs/キアロスクーロ/drums.wav"
DRUMS_MID = _HERE / "output/drums.mid"

CHECKPOINT = _HERE / "output" / "drum_cnn.pt"

# GM drum pitches, indexed to match CLASSES order
_GM_PITCHES: list[int] = [36, 38, 42]  # kick, snare, hihat

_THRESHOLD: float = 0.6
_MIN_GAP: float = 0.075  # seconds


def _load_model(checkpoint: Path = CHECKPOINT) -> DrumCNN:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"DrumCNN checkpoint not found: {checkpoint}\n"
            "Run  python train_cnn.py  to train the model first."
        )
    model = DrumCNN(n_classes=N_CLASSES)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu", weights_only=True))
    model.eval()
    return model


def _infer(model: DrumCNN, mel: np.ndarray) -> np.ndarray:
    """Run inference; returns (n_frames, N_CLASSES) onset probabilities."""
    n = len(mel)
    padded = np.pad(mel, ((_HALF, _HALF), (0, 0)), mode="edge")
    windows = np.stack([padded[i : i + CONTEXT] for i in range(n)], axis=0)
    x = torch.from_numpy(windows[:, np.newaxis, :, :])
    with torch.no_grad():
        probs = model(x).numpy()
    return probs


def _extract_onsets(
    probs: np.ndarray,
    threshold: float = _THRESHOLD,
    min_gap_frames: int = 0,
) -> dict[int, list[float]]:
    n_frames, n_cls = probs.shape
    min_gap_frames = max(min_gap_frames, 1)
    onsets: dict[int, list[float]] = {i: [] for i in range(n_cls)}
    for c in range(n_cls):
        last_onset_frame = -min_gap_frames
        for f in range(n_frames):
            if probs[f, c] >= threshold and (f - last_onset_frame) >= min_gap_frames:
                onsets[c].append(f * HOP / SR)
                last_onset_frame = f
    return onsets


def transcribe(
    audio_path: Path = DRUMS_WAV,
    midi_path: Path = DRUMS_MID,
    checkpoint: Path = CHECKPOINT,
    threshold: float = _THRESHOLD,
) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Input audio not found: {audio_path}")

    print(f"Loading DrumCNN from {checkpoint} …")
    model = _load_model(checkpoint)

    print(f"Computing log-mel spectrogram for {audio_path} …")
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=N_MELS),
        ref=np.max,
    ).T.astype(np.float32)

    print("Running CNN inference …")
    probs = _infer(model, mel)

    min_gap_frames = int(_MIN_GAP * SR / HOP)
    onsets = _extract_onsets(probs, threshold=threshold, min_gap_frames=min_gap_frames)

    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {len(onsets[i])} onsets")
    print(f"  Total: {sum(len(v) for v in onsets.values())} onsets")

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    pm   = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")

    for cls_idx, times in onsets.items():
        pitch = _GM_PITCHES[cls_idx]
        for t in times:
            note = pretty_midi.Note(velocity=80, pitch=pitch, start=t, end=t + 0.05)
            drum.notes.append(note)

    drum.notes.sort(key=lambda n: n.start)
    pm.instruments.append(drum)
    pm.write(str(midi_path))
    if not midi_path.exists():
        raise RuntimeError(f"transcribe() failed to write MIDI: {midi_path}")
    print(f"MIDI written to {midi_path}")

    return midi_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default=str(DRUMS_WAV))
    parser.add_argument("--out",   default=str(DRUMS_MID))
    args = parser.parse_args()
    transcribe(Path(args.audio), Path(args.out))
