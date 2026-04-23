# DrumCNN Diagnostic Report

Date: 2026-04-22

## Scope

This report summarizes failure-mode diagnosis outputs for the retrained model with updated hop size and context window:
- output/diagnostic/new_hop_size_and_tol_window/f1_vs_bpm.png
- output/diagnostic/new_hop_size_and_tol_window/f1_vs_polyphony.png

And includes evaluation metrics from evaluation.txt in this directory.

## Parameter Changes vs Baseline

| Parameter | Baseline | This Run |
|---|---:|---:|
| HOP (samples) | 512 | 256 |
| CONTEXT (frames) | 9 | 13 |
| `_THRESHOLD` | 0.5 | 0.6 |
| `_MIN_GAP` | 50 ms | 75 ms |

The halved hop size doubles the temporal resolution of the mel spectrogram (~5.8 ms/frame vs ~11.6 ms/frame). The wider context window (13 vs 9 frames) compensates by covering a similar absolute time span (~75 ms at 256-sample hop vs ~104 ms). The higher onset threshold and longer suppression gap together reduce false positives at the cost of potentially missing closely-spaced onsets.

## Evaluation Snapshot

Source: evaluation.txt

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Kick | 0.863 | 1.000 | 0.926 |
| Snare | 0.876 | 0.992 | 0.931 |
| Hi-Hat | 0.826 | 0.995 | 0.903 |
| Macro avg | 0.855 | 0.996 | 0.920 |

### Comparison to Baseline

| Class | Baseline F1 | This Run F1 | Delta |
|---|---:|---:|---:|
| Kick | 0.972 | 0.926 | -0.046 |
| Snare | 0.930 | 0.931 | +0.001 |
| Hi-Hat | 0.950 | 0.903 | -0.047 |
| Macro avg | 0.951 | 0.920 | -0.031 |

Interpretation:
- Recall is near-perfect across all classes (0.992–1.000), higher than the baseline.
- Precision has dropped notably, especially for kick (0.863 vs 0.946) and hi-hat (0.826 vs 0.919).
- Macro F1 decreased by 0.031 relative to baseline, driven by precision regression.
- The higher threshold (0.6) and longer suppression gap (75 ms) were intended to reduce false positives, but the net effect is worse precision — suggesting the retrained model at HOP=256/CONTEXT=13 is producing more false positives overall, not fewer, compared to the baseline configuration.

## Diagnostic Outputs

Source: diagnostic.txt

- Total windows analyzed: 704
- (Median macro F1 not directly reported; implied ~0.91 from split medians)

### 1) Temporal Smearing Hypothesis (F1 vs BPM)

Plot: output/diagnostic/new_hop_size_and_tol_window/f1_vs_bpm.png

Console summary:
- Median BPM: 111.1
- F1 at/below median BPM: 0.880 (N=384)
- F1 above median BPM: 0.937 (N=320)
- Delta (high BPM - low BPM): +0.057
- Verdict: NOT SUPPORTED

Observed trend:
- F1 is slightly *higher* at above-median BPM, consistent with the baseline finding.
- Low-BPM windows show more variance and pull the below-median average down.
- No evidence of temporal smearing degradation at higher tempos.

### 2) Spectral Masking Hypothesis (F1 vs Polyphony)

Plot: output/diagnostic/new_hop_size_and_tol_window/f1_vs_polyphony.png

Console summary:
- Median polyphony: 0.6
- F1 at/below median polyphony: 0.886 (N=361)
- F1 above median polyphony: 0.927 (N=343)
- Delta (high polyphony - low polyphony): +0.041
- Verdict: NOT SUPPORTED

Observed trend:
- Higher polyphony windows again show marginally better F1, consistent with baseline.
- No evidence that simultaneous onset density degrades model performance.

## Parameters Used

### Diagnosis parameters (diagnose.py)

- Sliding window size: 1.0 s
- Window stride: 1.0 s (non-overlapping)
- Minimum onsets per window to include: 2
- Event matching tolerance: 50 ms
- Polyphony definition: mean number of cross-class onset partners within +/-50 ms
- BPM estimation:
  - Primary: Beat This! (checkpoint=final0, device=cpu, dbn=False)
  - Fallback 1: median kick inter-onset interval
  - Fallback 2: 120 BPM default

### Evaluation parameters (evaluate.py)

- Dataset paths:
  - Audio: data/audio/*#MIX.wav
  - Annotations: data/annotation_xml/*#MIX.xml
- Classes evaluated: Kick, Snare, Hi-Hat
- Label collapsing for hats: HH/CHH/OHH -> Hi-Hat class
- Matching method: mir_eval.util.match_events
- Tolerance window: 50 ms
- Aggregation: classwise TP/FP/FN summed across files, then precision/recall/F1 per class plus macro average

### Transcription/runtime parameters (this run)

- Checkpoint path: output/drum_cnn.pt
- Onset threshold: 0.6
- Minimum inter-onset suppression gap: 75 ms
- Sample rate: 44100 Hz
- Hop length: 256 samples (~5.8 ms)
- Mel bins: 128
- Context window: 13 frames
- Output classes: kick, snare, hihat (3-class setup)

## Bottom Line

Neither failure-mode hypothesis is supported under the new configuration, consistent with the baseline.

However, the new parameter set (HOP=256, CONTEXT=13, threshold=0.6, gap=75 ms) underperforms the baseline on macro F1 (0.920 vs 0.951). The primary regression is in precision for kick and hi-hat, suggesting the model trained at higher temporal resolution generates more false positives that the stricter threshold and longer suppression gap do not fully compensate for. The baseline configuration (HOP=512, CONTEXT=9, threshold=0.5, gap=50 ms) remains the stronger overall result.
