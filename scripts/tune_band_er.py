"""Simulate alternative band ER segmentation parameters on all processed sessions.

Tries (min_peak_distance_s, amplitude_mode, ratio) combinations and reports
detected rep counts.  Runs in-memory only — does NOT modify anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.rep_segment import build_signal_band_er_side, smooth_signal
from src.utils.config import get_section


def segment_with(
    signal: np.ndarray,
    fps: int,
    min_peak_distance_s: float,
    duration_bounds_s: tuple[float, float],
    prominence: float,
    amplitude_mode: str,      # 'median', 'p75', 'max_median_p75'
    ratio: float,
) -> int:
    min_dist = max(1, int(min_peak_distance_s * fps))
    peaks, _ = find_peaks(signal, distance=min_dist, prominence=prominence)
    if len(peaks) == 0:
        return 0

    troughs: list[int] = [int(np.argmin(signal[: peaks[0] + 1]))]
    for i in range(len(peaks) - 1):
        segment = signal[peaks[i] : peaks[i + 1] + 1]
        troughs.append(peaks[i] + int(np.argmin(segment)))
    troughs.append(peaks[-1] + int(np.argmin(signal[peaks[-1] :])))

    min_dur, max_dur = duration_bounds_s
    candidates: list[tuple[int, int, float]] = []
    for i in range(len(peaks)):
        s, e = troughs[i], troughs[i + 1]
        if min_dur <= (e - s) / fps <= max_dur:
            seg = signal[s : e + 1]
            candidates.append((s, e, float(np.max(seg) - np.min(seg))))

    if not candidates:
        return 0

    amps = np.array([a for _, _, a in candidates])
    median_amp = float(np.median(amps))
    p75_amp = float(np.percentile(amps, 75))

    if amplitude_mode == "median":
        thr = ratio * median_amp
    elif amplitude_mode == "p75":
        thr = ratio * p75_amp
    elif amplitude_mode == "max_median_p75":
        thr = max(0.5 * median_amp, ratio * p75_amp)
    else:
        raise ValueError(amplitude_mode)

    return int(np.sum(amps >= thr))


def main() -> None:
    seg_cfg = get_section("segmentation")["band_er_side"]
    fps = 30
    sessions = sorted(
        p for p in Path("data/processed").iterdir() if p.name.startswith("band_er_side")
    )

    # Expected rep counts per user recording intent
    expected = {}
    for s in sessions:
        name = s.name
        if "good" in name:
            expected[name] = 15
        elif "bad_rom_partial" in name:
            expected[name] = 12
        elif "bad_tempo" in name:
            expected[name] = 12
        elif "elbow_drift" in name:
            expected[name] = 12

    signals = {}
    for s in sessions:
        pose_df = pd.read_parquet(s / "poses.parquet")
        raw = build_signal_band_er_side(pose_df)
        signals[s.name] = smooth_signal(
            raw, seg_cfg["smoothing_window"], seg_cfg["smoothing_polyorder"]
        )

    configs = [
        # (label, dist, mode, ratio)
        ("CURRENT 2.0s med×0.5",   2.0, "median", 0.5),
        ("1.3s p75×0.65",          1.3, "p75",    0.65),
        ("1.3s p75×0.7",           1.3, "p75",    0.7),
        ("1.3s p75×0.75",          1.3, "p75",    0.75),
        ("1.5s p75×0.7",           1.5, "p75",    0.7),
        ("1.5s p75×0.75",          1.5, "p75",    0.75),
    ]

    rows = []
    for s in sessions:
        name = s.name
        sig = signals[name]
        counts = [
            segment_with(
                sig, fps, dist,
                tuple(seg_cfg["rep_duration_bounds_s"]),
                seg_cfg["prominence"],
                mode, ratio,
            )
            for _, dist, mode, ratio in configs
        ]
        rows.append((name, expected.get(name, 0), counts))

    # Print
    print("EXPECTED rep counts: good=15, bad_*=12, elbow_drift=12")
    print("=" * 140)
    print(f'{"session":<42} {"exp":>4} ' + " ".join(f"{lbl:>24}" for lbl, *_ in configs))
    for name, exp, counts in rows:
        print(f'{name:<42} {exp:>4} ' + " ".join(f"{c:>24}" for c in counts))

    print("\nAbsolute deviation from expected (sum across sessions, lower=better):")
    for idx, (lbl, *_) in enumerate(configs):
        err = sum(abs(c[idx] - e) for _, e, c in rows if e > 0)
        over = sum(max(0, c[idx] - e) for _, e, c in rows if e > 0)
        under = sum(max(0, e - c[idx]) for _, e, c in rows if e > 0)
        print(f"  {lbl:<25} abs={err:>3}  over={over:>3}  under={under:>3}")

    # Per-category breakdown
    print("\nPer-category over/under (each config):")
    categories = ["good", "bad_rom_partial", "bad_tempo"]
    for idx, (lbl, *_) in enumerate(configs):
        line = f"  {lbl:<25} "
        for cat in categories:
            cat_rows = [(n, e, c) for n, e, c in rows if cat in n]
            tot_exp = sum(e for _, e, _ in cat_rows)
            tot_got = sum(c[idx] for _, _, c in cat_rows)
            line += f"{cat}: {tot_got}/{tot_exp}  "
        print(line)


if __name__ == "__main__":
    main()
