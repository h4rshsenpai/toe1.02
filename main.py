"""Drum transcription pipeline: audio → MIDI → PDF notation."""

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).parent
DEFAULT_WAV = _HERE / "output/htdemucs/キアロスクーロ/drums.wav"
DEFAULT_MID = _HERE / "output/drums.mid"
DEFAULT_PDF = _HERE / "output/drums.pdf"
DEFAULT_SEP_OUT = _HERE / "output"


def _separate(audio: Path, out_dir: Path) -> Path:
    """Run demucs htdemucs on *audio* and return the drums stem path."""
    print(f"Separating drums from {audio} …")
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "drums", str(audio), "-o", str(out_dir)],
        check=True,
    )
    drums_stem = out_dir / "htdemucs" / audio.stem / "drums.wav"
    if not drums_stem.exists():
        raise FileNotFoundError(
            f"Expected demucs output at {drums_stem} — check demucs model name or output layout."
        )
    return drums_stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio",    default=str(DEFAULT_WAV),
                        help="Path to input audio. A separated drums stem, or a full mix when --separate is set.")
    parser.add_argument("--mid",      default=str(DEFAULT_MID),  help="Output path for transcribed MIDI")
    parser.add_argument("--pdf",      default=str(DEFAULT_PDF),  help="Output path for notation PDF")
    parser.add_argument("--separate", action="store_true",
                        help="Run demucs stem separation on --audio before transcribing")
    parser.add_argument("--sep-out",  default=str(DEFAULT_SEP_OUT),
                        help="Output directory for demucs separation (only used with --separate)")
    args = parser.parse_args()

    audio = Path(args.audio)
    mid   = Path(args.mid)
    pdf   = Path(args.pdf)

    if not audio.exists():
        raise FileNotFoundError(f"Input audio not found: {audio}")

    if args.separate:
        audio = _separate(audio, Path(args.sep_out))

    from transcribe import transcribe
    from drums import midi_to_hits
    from render import render

    # Stage 1: transcribe audio → MIDI
    transcribe(audio, mid)

    # Stage 3: render MIDI → PDF notation (Stage 2 / Beat This! quantization is deferred)
    hits, bpm = midi_to_hits(mid)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    render(hits, bpm, pdf)


if __name__ == "__main__":
    main()
