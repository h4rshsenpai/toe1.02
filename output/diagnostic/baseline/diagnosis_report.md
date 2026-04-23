# DrumCNN Diagnostic Report

Date: 2026-04-22

## Scope

This report summarizes failure-mode diagnosis outputs:
- output/diagnostic/f1_vs_bpm.png
- output/diagnostic/f1_vs_polyphony.png

And includes evaluation metrics from the latest paired measurement snapshot in measurements.txt.

## Evaluation Snapshot

Source: measurements.txt

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Kick | 0.946 | 0.999 | 0.972 |
| Snare | 0.878 | 0.989 | 0.930 |
| Hi-Hat | 0.919 | 0.983 | 0.950 |
| Macro avg | 0.914 | 0.990 | 0.951 |

Interpretation:
- Recall is very high across all classes.
- Precision is the main limiting factor, especially for snare.
- Macro F1 is high (0.951), with most misses likely due to false positives rather than missed onsets.

## Diagnostic Outputs

Source: diagnostic summary in measurements.txt and plots in output/diagnostic.

- Total windows analyzed: 704
- Median macro F1 across windows: 0.952

### 1) Temporal Smearing Hypothesis (F1 vs BPM)

Plot: output/diagnostic/f1_vs_bpm.png

Console summary:
- Median BPM: 111.1
- F1 at/below median BPM: 0.909 (N=384)
- F1 above median BPM: 0.959 (N=320)
- Delta (high BPM - low BPM): +0.050
- Verdict: NOT SUPPORTED

Observed trend:
- Points remain concentrated near high F1 at moderate and high BPM.
- While some low-F1 outliers exist around ~90-110 BPM, there is no overall downward trend as BPM increases.

### 2) Spectral Masking Hypothesis (F1 vs Polyphony)

Plot: output/diagnostic/f1_vs_polyphony.png

Console summary:
- Median polyphony: 0.6
- F1 at/below median polyphony: 0.910 (N=361)
- F1 above median polyphony: 0.954 (N=343)
- Delta (high polyphony - low polyphony): +0.043
- Verdict: NOT SUPPORTED

Observed trend:
- Higher polyphony does not show consistent degradation in macro F1.
- Most windows stay near high F1, with scattered low-F1 outliers across the polyphony range.

## Parameters Used

### Diagnosis parameters (diagnose.py)

- Sliding window size: 1.0 s (from plot titles: "window=1.0s")
- Window stride: 1.0 s (non-overlapping windows; loop increments by window size)
- Minimum onsets per window to include: 2
- Event matching tolerance: 50 ms (reused from evaluate.py)
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

### Transcription/runtime parameters affecting evaluation (transcribe.py, train_cnn.py)

- Checkpoint path: output/drum_cnn.pt (default for evaluate.py)
- Onset threshold: 0.5
- Minimum inter-onset suppression gap: 50 ms
- Sample rate: 44100 Hz
- Hop length: 512 samples (~11.6 ms)
- Mel bins: 128
- Context window: 9 frames
- Output classes: kick, snare, hihat (3-class setup)

## Bottom Line

For the analyzed run, neither tested failure mode is supported by the diagnostic split analysis:
- No clear F1 degradation at higher BPM
- No clear F1 degradation at higher polyphony

Current performance appears strong overall (macro F1 ~0.95), with precision (especially snare) as the main area for further improvement.
