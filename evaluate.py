"""Evaluate CNN drum transcription against IDMT-SMT-Drums V2."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

from transcribe import transcribe
from drums import DrumType, _gm_to_drum

_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / "data"
AUDIO_DIR = DATA_DIR / "audio"
XML_DIR   = DATA_DIR / "annotation_xml"

# Dataset label → DrumType
LABEL_MAP = {
    "KD":  DrumType.KICK,
    "SD":  DrumType.SNARE,
    "HH":  DrumType.HIHAT_CLOSED,
    "CHH": DrumType.HIHAT_CLOSED,
    "OHH": DrumType.HIHAT_CLOSED,
}
EVAL_CLASSES = [DrumType.KICK, DrumType.SNARE, DrumType.HIHAT_CLOSED]
CLASS_NAMES  = {
    DrumType.KICK:         "Kick",
    DrumType.SNARE:        "Snare",
    DrumType.HIHAT_CLOSED: "Hi-Hat",
}

TOLERANCE = 0.05  # 50 ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_xml(xml_path: Path) -> dict[DrumType, np.ndarray]:
    """Return onset arrays per drum class from a #MIX annotation XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    onsets: dict[DrumType, list[float]] = defaultdict(list)
    for event in root.findall(".//event"):
        label = event.findtext("instrument", "").strip().upper()
        onset_str = event.findtext("onsetSec", "")
        drum = LABEL_MAP.get(label)
        if drum and onset_str:
            onsets[drum].append(float(onset_str))
    return {dt: np.array(sorted(v)) for dt, v in onsets.items()}


def transcribe_file(wav_path: Path) -> dict[DrumType, np.ndarray]:
    """Run DrumCNN on a WAV and return onset arrays per drum class."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        tmp_mid = Path(f.name)
    try:
        transcribe(wav_path, tmp_mid)
        pm = pretty_midi.PrettyMIDI(str(tmp_mid))
        onsets: dict[DrumType, list[float]] = defaultdict(list)
        for inst in pm.instruments:
            if not inst.is_drum:
                continue
            for note in inst.notes:
                drum = _gm_to_drum(note.pitch)
                if drum:
                    onsets[drum].append(note.start)
        return {dt: np.array(sorted(v)) for dt, v in onsets.items()}
    finally:
        tmp_mid.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate() -> None:
    xml_files = sorted(XML_DIR.glob("*#MIX.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No #MIX.xml files found in {XML_DIR}")

    print(f"Evaluating on {len(xml_files)} drum loops …\n")

    # Accumulators: sum of (tp, fp, fn) per class across all files
    totals: dict[DrumType, dict[str, float]] = {
        dt: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for dt in EVAL_CLASSES
    }

    for i, xml_path in enumerate(xml_files, 1):
        stem  = xml_path.stem.replace("#MIX", "")
        wav   = AUDIO_DIR / f"{stem}#MIX.wav"
        if not wav.exists():
            continue

        print(f"[{i:3d}/{len(xml_files)}] {stem}", end="  ", flush=True)
        try:
            ref = parse_xml(xml_path)
            est = transcribe_file(wav)
        except Exception as exc:
            print(f"SKIP ({exc})")
            continue

        for dt in EVAL_CLASSES:
            ref_onsets = ref.get(dt, np.array([]))
            est_onsets = est.get(dt, np.array([]))
            # Use match_events directly for exact integer tp/fp/fn counts
            if len(ref_onsets) > 0 and len(est_onsets) > 0:
                matching = mir_eval.util.match_events(ref_onsets, est_onsets,
                                                      TOLERANCE)
                tp = len(matching)
            else:
                tp = 0
            fp = len(est_onsets) - tp
            fn = len(ref_onsets) - tp
            totals[dt]["tp"] += tp
            totals[dt]["fp"] += fp
            totals[dt]["fn"] += fn

        print("✓")

    # --- Results table -------------------------------------------------------
    print()
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("─" * 48)

    macro_p = macro_r = macro_f = 0.0
    for dt in EVAL_CLASSES:
        tp = totals[dt]["tp"]
        fp = totals[dt]["fp"]
        fn = totals[dt]["fn"]
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        macro_p += p; macro_r += r; macro_f += f
        print(f"{CLASS_NAMES[dt]:<14} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

    n = len(EVAL_CLASSES)
    print("─" * 48)
    print(f"{'Macro avg':<14} {macro_p/n:>10.3f} {macro_r/n:>10.3f} {macro_f/n:>10.3f}")
    print(f"\nTolerance window: {TOLERANCE*1000:.0f} ms")


if __name__ == "__main__":
    evaluate()
