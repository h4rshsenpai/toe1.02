"""Failure-mode analysis for DrumCNN transcription.

Tests two hypotheses:
  1. Spectral Masking: F1 decays as simultaneous onset density (polyphony) increases.
  2. Temporal Smearing: F1 decays as BPM increases (shorter inter-onset intervals).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mir_eval.util
import numpy as np

from drums import DrumType
from evaluate import (
    AUDIO_DIR,
    EVAL_CLASSES,
    TOLERANCE,
    XML_DIR,
    parse_xml,
    transcribe_file,
)

_HERE = Path(__file__).parent
OUTPUT_DIR = _HERE / "output" / "diagnostic"
MIN_ONSETS_PER_WINDOW = 2
_EMPTY = np.array([], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-window metrics
# ---------------------------------------------------------------------------

def compute_window_polyphony(
    ref: dict[DrumType, np.ndarray],
    t_start: float,
    t_end: float,
) -> float:
    all_onsets: list[tuple[float, DrumType]] = []
    for dt in EVAL_CLASSES:
        arr = ref.get(dt, np.array([]))
        for onset in arr[(arr >= t_start) & (arr < t_end)]:
            all_onsets.append((float(onset), dt))

    if not all_onsets:
        return 0.0

    partners = []
    for t_i, cls_i in all_onsets:
        count = sum(
            1 for t_j, cls_j in all_onsets
            if cls_j != cls_i and abs(t_i - t_j) <= TOLERANCE
        )
        partners.append(count)
    return float(np.mean(partners))


def compute_window_f1(
    ref: dict[DrumType, np.ndarray],
    est: dict[DrumType, np.ndarray],
    t_start: float,
    t_end: float,
) -> float | None:
    f1s = []
    for dt in EVAL_CLASSES:
        ref_arr = ref.get(dt, np.array([]))
        est_arr = est.get(dt, np.array([]))
        ref_sl = ref_arr[(ref_arr >= t_start) & (ref_arr < t_end)]
        est_sl = est_arr[(est_arr >= t_start) & (est_arr < t_end)]

        if len(ref_sl) == 0:
            continue  # don't include in macro average

        if len(ref_sl) > 0 and len(est_sl) > 0:
            tp = len(mir_eval.util.match_events(ref_sl, est_sl, TOLERANCE))
        else:
            tp = 0

        fp = len(est_sl) - tp
        fn = len(ref_sl) - tp
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)

    return float(np.mean(f1s)) if f1s else None


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def _estimate_bpm(wav_path: Path, ref: dict[DrumType, np.ndarray]) -> float:
    from beat_this.inference import File2Beats  # lazy import — slow to load
    f2b = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, _ = f2b(str(wav_path))
    if len(beats) >= 2:
        return float(60.0 / np.median(np.diff(beats)))

    # fallback: use reference kick IOI
    kick = ref.get(DrumType.KICK, np.array([]))
    if len(kick) >= 2:
        return float(60.0 / np.median(np.diff(kick)))

    return 120.0


def analyze_file(
    wav_path: Path,
    xml_path: Path,
    window_size: float,
) -> list[dict]:
    ref = parse_xml(xml_path)
    bpm = _estimate_bpm(wav_path, ref)
    est = transcribe_file(wav_path)

    ref_times = [arr[-1] for arr in ref.values() if len(arr) > 0]
    if not ref_times:
        return []
    max_time = max(ref_times)

    data: list[dict] = []
    t = 0.0
    while t < max_time:
        t_end = t + window_size

        ref_slices = {
            dt: (arr := ref.get(dt, _EMPTY))[(arr >= t) & (arr < t_end)]
            for dt in EVAL_CLASSES
        }
        if sum(len(s) for s in ref_slices.values()) >= MIN_ONSETS_PER_WINDOW:
            macro_f1 = compute_window_f1(ref, est, t, t_end)
            if macro_f1 is not None:
                polyphony = compute_window_polyphony(ref, t, t_end)
                data.append({"bpm": bpm, "polyphony": polyphony, "macro_f1": macro_f1})

        t += window_size

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.3, s=15, edgecolors="none", color="#2196F3")
    ax.axhline(
        float(np.median(y)),
        color="red", linestyle="--", linewidth=1,
        label=f"Median F1 = {np.median(y):.3f}",
    )
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_console_report(data_points: list[dict]) -> None:
    bpms = np.array([d["bpm"] for d in data_points])
    polys = np.array([d["polyphony"] for d in data_points])
    f1s = np.array([d["macro_f1"] for d in data_points])

    print("\n=== Diagnostic Report ===")
    print(f"Total windows analyzed: {len(data_points)}\n")

    for label, values, metric_name in [
        ("Temporal Smearing (BPM)", bpms, "BPM"),
        ("Spectral Masking (Polyphony)", polys, "polyphony"),
    ]:
        median_val = float(np.median(values))
        below_mask = values <= median_val
        above_mask = values > median_val

        f1_below = float(np.mean(f1s[below_mask])) if below_mask.any() else float("nan")
        f1_above = float(np.mean(f1s[above_mask])) if above_mask.any() else float("nan")
        delta = f1_above - f1_below
        verdict = "SUPPORTED" if delta < 0 else "NOT SUPPORTED"

        print(f"--- {label} ---")
        print(f"Median {metric_name}: {median_val:.1f}")
        print(f"  Below median ({metric_name} <= {median_val:.1f}):  N={below_mask.sum():5d}  Macro F1 = {f1_below:.3f}")
        print(f"  Above median ({metric_name}  > {median_val:.1f}):  N={above_mask.sum():5d}  Macro F1 = {f1_above:.3f}")
        print(f"  F1 delta = {delta:+.3f}  → {verdict}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def diagnose() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose DrumCNN transcription failure modes."
    )
    parser.add_argument("--window", type=float, default=2.0,
                        help="Sliding window size in seconds (default: 2.0)")
    args = parser.parse_args()
    window_size: float = args.window

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(XML_DIR.glob("*#MIX.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No #MIX.xml files found in {XML_DIR}")

    print(f"Analyzing {len(xml_files)} files with {window_size}s windows …\n")

    all_data: list[dict] = []
    for i, xml_path in enumerate(xml_files, 1):
        stem = xml_path.stem.replace("#MIX", "")
        wav = AUDIO_DIR / f"{stem}#MIX.wav"
        if not wav.exists():
            print(f"[{i:3d}/{len(xml_files)}] {stem}  SKIP (no audio)")
            continue
        print(f"[{i:3d}/{len(xml_files)}] {stem}", end="  ", flush=True)
        try:
            file_data = analyze_file(wav, xml_path, window_size)
            all_data.extend(file_data)
            print(f"✓  ({len(file_data)} windows)")
        except Exception as exc:
            print(f"SKIP ({exc})")

    if not all_data:
        print("No data collected — nothing to plot.")
        return

    bpms = np.array([d["bpm"] for d in all_data])
    polys = np.array([d["polyphony"] for d in all_data])
    f1s = np.array([d["macro_f1"] for d in all_data])
    y_label = f"Macro F1 (per {window_size}s window)"

    bpm_path = OUTPUT_DIR / "f1_vs_bpm.png"
    poly_path = OUTPUT_DIR / "f1_vs_polyphony.png"

    plot_scatter(
        bpms, f1s,
        x_label="BPM (per file)",
        y_label=y_label,
        title=f"Temporal Smearing: F1 vs BPM (window={window_size}s)",
        output_path=bpm_path,
    )
    plot_scatter(
        polys, f1s,
        x_label="Mean Simultaneous Cross-Class Onsets (per window)",
        y_label=y_label,
        title=f"Spectral Masking: F1 vs Polyphony (window={window_size}s)",
        output_path=poly_path,
    )

    print_console_report(all_data)

    print(f"Saved: {bpm_path}")
    print(f"Saved: {poly_path}")


if __name__ == "__main__":
    diagnose()
