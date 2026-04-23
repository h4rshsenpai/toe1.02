"""Step 2: Parse MIDI, track beats with Beat This!, snap hits to 16th-note grid."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import mido
import numpy as np
import pretty_midi

from drums import DrumType, QuantizedHit, _gm_to_drum

_HERE = Path(__file__).parent
DRUMS_WAV = _HERE / "output/htdemucs/キアロスクーロ/drums.wav"
DRUMS_MID = _HERE / "output/drums.mid"


# ---------------------------------------------------------------------------
# Beat-aligned quantization
# ---------------------------------------------------------------------------

def _build_sixteenth_grid(beats: np.ndarray) -> np.ndarray:
    """
    Interpolate a 16th-note grid from beat times.
    Each beat is divided into 4 equal 16th-note slots using the local
    beat duration, so the grid adapts to natural tempo drift.

    One beat is extrapolated past the last detected beat so that notes
    near the end of the track have a complete set of grid points.

    Returns a 1-D array of 16th-note onset times.
    """
    if len(beats) < 2:
        # Fallback: single beat or empty — return what we have
        return beats.copy() if len(beats) else np.array([0.0])

    # Extrapolate one beat beyond the last to cover the final beat's subdivisions
    last_dur = float(beats[-1]) - float(beats[-2])
    extended = np.append(beats, beats[-1] + last_dur)

    grid: list[float] = []
    for i in range(len(extended) - 1):
        beat_start = float(extended[i])
        beat_dur   = float(extended[i + 1]) - float(extended[i])
        for sub in range(4):
            grid.append(beat_start + sub * beat_dur / 4)
    return np.array(grid)


def write_tempo_mapped_midi(
    src_midi:  Path,
    beats:     np.ndarray,
    dest_midi: Path,
    ticks_per_beat: int = 480,
    note_dur_ticks: int = 40,          # short fixed duration for drum hits
) -> None:
    """Rewrite MIDI replacing the hardcoded 120 BPM with Beat This! beat times.

    Each inter-beat interval becomes a tempo-change event, so the file plays
    back at the correct (and naturally varying) tempo in any MIDI player.
    """
    pm = pretty_midi.PrettyMIDI(str(src_midi))

    # Extend beats by one interval for extrapolation so notes near the end
    # of the track map cleanly without frac > 1.0 producing tick collisions.
    if len(beats) >= 2:
        last_dur = float(beats[-1]) - float(beats[-2])
        _beats_ext = np.append(beats, beats[-1] + last_dur)
    else:
        _beats_ext = beats

    # --- seconds → ticks converter using the beat map -----------------------
    def _to_ticks(t: float) -> int:
        if len(_beats_ext) < 2:
            return int(t * ticks_per_beat * 2)           # fallback: 120 BPM
        idx = int(np.searchsorted(_beats_ext, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(_beats_ext) - 2))
        beat_dur = float(_beats_ext[idx + 1] - _beats_ext[idx])
        frac     = (t - float(_beats_ext[idx])) / beat_dur
        return int(idx * ticks_per_beat + frac * ticks_per_beat)

    # --- build mido file -----------------------------------------------------
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)

    # Track 0: tempo map — one SetTempo per beat interval
    tempo_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    for i in range(len(beats) - 1):
        uspb  = int((float(beats[i + 1]) - float(beats[i])) * 1_000_000)
        delta = ticks_per_beat if i > 0 else 0
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=uspb, time=delta))
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))

    # Track 1: drum notes on channel 9
    drum_track = mido.MidiTrack()
    mid.tracks.append(drum_track)

    # Collect note-on / note-off events in absolute ticks
    events: list[tuple[int, str, int, int]] = []   # (abs_tick, type, pitch, vel)
    for inst in pm.instruments:
        if not inst.is_drum:
            continue
        for note in inst.notes:
            t_on  = _to_ticks(note.start)
            t_off = t_on + note_dur_ticks
            events.append((t_on,  "note_on",  note.pitch, note.velocity))
            events.append((t_off, "note_off", note.pitch, 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    prev_tick = 0
    for abs_tick, msg_type, pitch, velocity in events:
        delta = max(abs_tick - prev_tick, 0)
        drum_track.append(
            mido.Message(msg_type, channel=9, note=pitch,
                         velocity=velocity, time=delta)
        )
        prev_tick = abs_tick
    drum_track.append(mido.MetaMessage("end_of_track", time=0))

    dest_midi.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(dest_midi))
    print(f"Tempo-mapped MIDI written to {dest_midi}  "
          f"({len(beats)-1} tempo events)")


def quantize(
    midi_path:       Path = DRUMS_MID,
    audio_path:      Path = DRUMS_WAV,
    meter:           int  = 4,
    beat_audio_path: Path | None = None,
) -> Tuple[List[QuantizedHit], float, np.ndarray]:
    """Return (hits, bpm, beats) — hits snapped to a beat-aligned 16th-note grid.

    ``beat_audio_path``, if given, is used for beat tracking instead of
    ``audio_path``.  Pass the original (unseparated) mix here for more
    reliable tempo detection; ``audio_path`` (the drum stem) is still used
    for everything else.
    """
    beat_src = beat_audio_path if beat_audio_path is not None else audio_path

    # --- Beat This! -----------------------------------------------------------
    print("Running Beat This! beat tracker …")
    from beat_this.inference import File2Beats
    f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, downbeats = f2b(str(beat_src))

    if len(beats) < 2:
        raise ValueError(
            f"Beat This! detected fewer than 2 beats in {audio_path}. "
            "The audio may be too short or silent."
        )

    bpm = float(60.0 / np.median(np.diff(beats)))
    beats_per_measure = int(round(np.median([
        np.sum((beats >= downbeats[i]) & (beats < downbeats[i + 1]))
        for i in range(min(len(downbeats) - 1, 8))
    ])))
    if beats_per_measure < 2:
        beats_per_measure = meter  # fallback

    sixteenths_per_measure = beats_per_measure * 4
    print(f"BPM: {bpm:.1f}  |  meter: {beats_per_measure}/4  |  "
          f"downbeats: {len(downbeats)}")

    # Build the full 16th-note grid anchored to actual beat positions.
    # The grid is extrapolated one beat past the last detected beat so
    # notes at the end of the track are assigned correct grid positions.
    grid = _build_sixteenth_grid(beats)  # absolute times of every 16th note

    # Map each downbeat to its nearest grid index (nearest-neighbour, not
    # searchsorted, so a downbeat slightly before a grid point isn't shifted
    # forward by one 16th note).
    raw_idx   = np.searchsorted(grid, downbeats)
    raw_idx   = np.clip(raw_idx, 0, len(grid) - 1)
    prev_idx  = np.maximum(raw_idx - 1, 0)
    use_prev  = np.abs(grid[prev_idx] - downbeats) < np.abs(grid[raw_idx] - downbeats)
    downbeat_idx = np.where(use_prev, prev_idx, raw_idx)

    def _grid_to_measure_pos(grid_idx: int) -> tuple[int, int]:
        """Return (measure, sixteenth_within_measure) for a grid index."""
        measure = int(np.searchsorted(downbeat_idx, grid_idx, side="right")) - 1
        measure = max(measure, 0)
        offset  = grid_idx - downbeat_idx[measure]
        # Use abs() to avoid Python's negative-modulo wrapping pre-first-downbeat
        return measure, int(abs(offset) % sixteenths_per_measure)

    # --- Parse MIDI -----------------------------------------------------------
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    raw: list[tuple[float, DrumType, float]] = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            continue
        for note in instrument.notes:
            drum = _gm_to_drum(note.pitch)
            if drum is None:
                continue
            raw.append((note.start, drum, note.velocity / 127.0))

    if not raw:
        raise ValueError("No drum notes found in MIDI.")

    # --- Snap to grid ---------------------------------------------------------
    seen: dict[tuple[int, int, DrumType], QuantizedHit] = {}
    for time, drum_type, velocity in raw:
        # Find nearest 16th-note grid point
        grid_idx = int(np.argmin(np.abs(grid - time)))
        measure, sixteenth = _grid_to_measure_pos(grid_idx)
        key = (measure, sixteenth, drum_type)
        if key not in seen or velocity > seen[key].velocity:
            seen[key] = QuantizedHit(measure, sixteenth, drum_type, velocity)

    hits = sorted(seen.values(), key=lambda h: (h.measure, h.sixteenth))
    print(f"Quantized {len(hits)} unique hits across {hits[-1].measure + 1} measures")
    return hits, bpm, beats


if __name__ == "__main__":
    hits, bpm, beats = quantize()
    for h in hits[:20]:
        print(h)
