"""Shared drum data model and MIDI parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
import pretty_midi


# ---------------------------------------------------------------------------
# GM drum note → instrument type
# ---------------------------------------------------------------------------

class DrumType(Enum):
    KICK         = "kick"
    SNARE        = "snare"
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN   = "hihat_open"
    CRASH        = "crash"
    RIDE         = "ride"
    TOM_HI       = "tom_hi"
    TOM_MID      = "tom_mid"
    TOM_FLOOR    = "tom_floor"


_GM_MAP: dict[int, DrumType] = {
    35: DrumType.KICK,  36: DrumType.KICK,
    38: DrumType.SNARE, 40: DrumType.SNARE,
    37: DrumType.SNARE,  # cross-stick
    42: DrumType.HIHAT_CLOSED, 44: DrumType.HIHAT_CLOSED,
    46: DrumType.HIHAT_OPEN,
    49: DrumType.CRASH, 55: DrumType.CRASH, 57: DrumType.CRASH,
    51: DrumType.RIDE,  53: DrumType.RIDE,  59: DrumType.RIDE,
    47: DrumType.TOM_HI,  48: DrumType.TOM_HI,
    43: DrumType.TOM_MID, 45: DrumType.TOM_MID,
    41: DrumType.TOM_FLOOR,
}


def _gm_to_drum(pitch: int) -> DrumType | None:
    return _GM_MAP.get(pitch)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QuantizedHit:
    measure:   int    # 0-indexed
    sixteenth: int    # 0-indexed within measure (0–15 for 4/4)
    drum_type: DrumType
    velocity:  float  # 0–1


# ---------------------------------------------------------------------------
# MIDI → hits (no beat tracker; uses MIDI tempo map)
# ---------------------------------------------------------------------------

def midi_to_hits(midi_path: Path, meter: int = 4) -> tuple[List[QuantizedHit], float]:
    """Parse a MIDI file into QuantizedHit objects using the file's own tempo.

    Uses the MIDI's internal tempo events to build a 16th-note grid — no
    external beat tracker required.  Returns (hits, bpm) where bpm is the
    median tempo across the file.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Median BPM from MIDI tempo map
    _, tempos = pm.get_tempo_changes()
    if len(tempos) == 0:
        bpm = 120.0
    else:
        bpm = float(np.median(tempos))

    beat_dur = 60.0 / bpm
    sixteenth_dur = beat_dur / 4.0
    sixteenths_per_measure = meter * 4

    raw: list[tuple[float, DrumType, float]] = []
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            drum = _gm_to_drum(note.pitch)
            if drum is None:
                continue
            raw.append((note.start, drum, note.velocity / 127.0))

    if not raw:
        raise ValueError("No drum notes found in MIDI.")

    seen: dict[tuple[int, int, DrumType], QuantizedHit] = {}
    for time, drum_type, velocity in raw:
        grid_idx = int(round(time / sixteenth_dur))
        measure  = grid_idx // sixteenths_per_measure
        sixteenth = grid_idx % sixteenths_per_measure
        key = (measure, sixteenth, drum_type)
        if key not in seen or velocity > seen[key].velocity:
            seen[key] = QuantizedHit(measure, sixteenth, drum_type, velocity)

    hits = sorted(seen.values(), key=lambda h: (h.measure, h.sixteenth))
    return hits, bpm
