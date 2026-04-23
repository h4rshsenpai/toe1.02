# toe — Drum Transcription Pipeline

Automatic drum transcription from audio to MIDI.

Current active flow: stem separation (optional) → note detection/transcription.

Quantization and notation rendering remain in the repository as deferred components and are not part of the default runtime path.

## Pipeline

```
audio.mp3
  ↓
[demucs] → drums.wav
  ↓
[DrumCNN] → MIDI
```

### Scripts

- **`main.py`** — active CLI pipeline (transcribe audio → MIDI)
- **`transcribe.py`** — DrumCNN inference to MIDI
- **`quantize.py`** — deferred Stage 2 module kept for future reactivation
- **`render.py`** — deferred notation rendering module
- **`evaluate.py`** — F1 benchmark against IDMT-SMT-Drums V2 (95 drum loops)

## Setup

Requires Python 3.14+, PyTorch, librosa, pretty-midi, and more. Install via [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Usage

### Active pipeline
```bash
python main.py \
  --audio drums.wav \      # Input audio (typically Demucs drum stem)
  --mid output/drums.mid    # Output MIDI
```

### Individual steps
```bash
# Transcribe drums stem to MIDI
python transcribe.py --audio drums.wav --out drums.mid

# Deferred stage (kept for future): quantize to grid
python quantize.py

# Evaluate on IDMT-SMT-Drums V2
python evaluate.py
```

## Performance

Run `python evaluate.py` after retraining the checkpoint to report precision, recall, and F1 for the current 3-class setup:

- Kick
- Snare
- Hi-Hat

The evaluation harness now collapses `HH`, `CHH`, and `OHH` into one hi-hat bucket so the reported macro average is computed across three classes.

## Deferred Components

- `quantize.py`: Beat This! beat tracking and 16th-note quantization logic (not currently invoked by `main.py`).
- `render.py`: PDF notation generation from quantized hits (not currently invoked by `main.py`).

See `DECISIONS.md` for rationale and reactivation notes.

## Dependencies

- **[demucs](https://github.com/facebookresearch/demucs)** — Source separation (4.0.1+)
- **PyTorch** — DrumCNN training and inference
- **[Beat This!](https://github.com/CPJKU/beat_this)** — Deferred beat tracking stage
- **[librosa](https://librosa.org/)** — Audio loading & analysis
- **[pretty-midi](https://github.com/craffel/pretty-midi)** — MIDI I/O
- **[matplotlib](https://matplotlib.org/)** — Visualization
- **[mir-eval](https://craffel.github.io/mir_eval/)** — Evaluation metrics

## License

[Your license here]
