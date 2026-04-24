"""Rep segmentation: detect individual reps from pose landmarks.

Wall Slide (side view): tracks the better-visibility wrist's vertical travel,
normalised by torso height, inverted so that arms-raised = high signal.

Band ER Side (front view): lateral wrist displacement — added in Phase 7.

Usage:
    python src/pipeline/rep_segment.py \\
        --poses data/poses/<session>/poses.parquet \\
        --exercise wall_slide \\
        --fps 30 \\
        --output data/processed/<session>/reps.csv \\
        --plot  data/processed/<session>/segmentation_plot.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from src.utils.config import get_section


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Rep:
    rep_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float


# ── Arm / side selection ──────────────────────────────────────────────────────

def select_signal_arm(pose_df: pd.DataFrame) -> str:
    """Return 'left' or 'right' based on which wrist has higher mean visibility.

    In side-view exercises the near-side arm is well-tracked while the occluded
    arm has low visibility, so this reliably selects the visible wrist.

    Args:
        pose_df: Wide-format pose DataFrame from extract_poses().

    Returns:
        'left' or 'right'.
    """
    left_vis = pose_df.get("left_wrist_vis", pd.Series(dtype=float)).mean()
    right_vis = pose_df.get("right_wrist_vis", pd.Series(dtype=float)).mean()
    if pd.isna(right_vis):
        return "left"
    if pd.isna(left_vis):
        return "right"
    return "left" if left_vis >= right_vis else "right"


def select_working_arm_band_er(pose_df: pd.DataFrame) -> str:
    """Pick the working arm for Band ER Side based on lateral wrist travel.

    The working arm is the one whose wrist moves more relative to the elbow
    in the x-axis (lateral displacement).  From front view both arms are
    visible, so visibility is not a reliable differentiator — movement is.

    Args:
        pose_df: Wide-format pose DataFrame from extract_poses().

    Returns:
        'left' or 'right'.
    """
    left_lateral = (pose_df["left_wrist_x"] - pose_df["left_elbow_x"]).abs()
    right_lateral = (pose_df["right_wrist_x"] - pose_df["right_elbow_x"]).abs()

    left_travel = float(left_lateral.max() - left_lateral.min())
    right_travel = float(right_lateral.max() - right_lateral.min())

    return "left" if left_travel >= right_travel else "right"


# ── Active window detection ──────────────────────────────────────────────────

def detect_active_window(
    pose_df: pd.DataFrame,
    fps: int,
    settle_s: float = 1.0,
    multiplier: float = 3.0,
    min_threshold: float = 0.03,
) -> tuple[int, int]:
    """Detect the exercise window by finding when the torso stabilises.

    During setup/teardown the whole body translates (walking to position,
    reaching for the phone).  During exercise the torso stays roughly fixed
    while the limbs move.  We measure torso instability as the rolling
    standard deviation of the hip midpoint over a 2-second window, then
    trim leading/trailing frames where instability exceeds a threshold
    derived from the session median.

    Args:
        pose_df:       Wide-format pose DataFrame.
        fps:           Video frame rate.
        settle_s:      The torso must remain stable for this many seconds
                       before we consider the exercise started (and after
                       we consider it ended).
        multiplier:    Threshold = multiplier × median torso movement.
        min_threshold: Floor for the threshold to avoid over-trimming
                       very stable sessions.

    Returns:
        (start_frame, end_frame) inclusive bounds of the active window.
    """
    n = len(pose_df)
    if n < fps * 3:
        return 0, n - 1

    hip_x = (
        (pose_df["left_hip_x"].values + pose_df["right_hip_x"].values) / 2
    ).astype(float)
    hip_y = (
        (pose_df["left_hip_y"].values + pose_df["right_hip_y"].values) / 2
    ).astype(float)

    win = 2 * fps
    torso_move = np.sqrt(
        pd.Series(hip_x).rolling(win, center=True, min_periods=fps).std().values ** 2
        + pd.Series(hip_y).rolling(win, center=True, min_periods=fps).std().values ** 2
    )

    med = float(np.nanmedian(torso_move))
    threshold = max(multiplier * med, min_threshold)

    settle_frames = int(settle_s * fps)

    # Scan forward: first frame where torso stays stable for settle_s
    start_frame = 0
    for i in range(n - settle_frames):
        if np.isnan(torso_move[i]):
            continue
        if torso_move[i] < threshold:
            window = torso_move[i : i + settle_frames]
            if np.nanmax(window) < threshold:
                start_frame = i
                break

    # Scan backward: last frame where torso was still stable for settle_s
    end_frame = n - 1
    for i in range(n - 1, settle_frames, -1):
        if np.isnan(torso_move[i]):
            continue
        if torso_move[i] < threshold:
            window = torso_move[max(0, i - settle_frames) : i + 1]
            if np.nanmax(window) < threshold:
                end_frame = i
                break

    return start_frame, end_frame


# ── Signal construction ───────────────────────────────────────────────────────

def build_signal_wall_slide(pose_df: pd.DataFrame) -> np.ndarray:
    """Build a normalised wrist-height signal for the Wall Slide (side view).

    Uses the better-tracked wrist (highest mean visibility) and computes how
    high the wrist is relative to the shoulder, normalised by torso height so
    the signal is comparable across body sizes and camera distances.

    In image coordinates y increases downward, so arms-raised = smaller wrist_y.
    We compute (shoulder_y - wrist_y) / torso_height so that raising the arms
    produces a *positive, increasing* signal — peaks correspond to arm-up.

    NaN frames (undetected pose) are filled by linear interpolation so that
    smoothing and peak detection operate on a continuous signal.

    Args:
        pose_df: Wide-format pose DataFrame from extract_poses().

    Returns:
        1-D float array, one value per frame.  Higher = arm raised higher.
    """
    arm = select_signal_arm(pose_df)

    wrist_y = pose_df[f"{arm}_wrist_y"].values.astype(float)
    shoulder_y = pose_df[f"{arm}_shoulder_y"].values.astype(float)
    hip_y = pose_df[f"{arm}_hip_y"].values.astype(float)

    # torso_height is positive because hip is below shoulder in image coords
    torso_height = hip_y - shoulder_y
    median_torso = float(np.nanmedian(torso_height))
    if not (median_torso > 0):
        median_torso = 0.2          # fallback: ~20% of image height
    # Replace near-zero or NaN torso values to avoid division noise
    torso_height = np.where(
        ~np.isfinite(torso_height) | (np.abs(torso_height) < 1e-3),
        median_torso,
        torso_height,
    )

    signal = (shoulder_y - wrist_y) / torso_height   # positive = arm above shoulder

    # Linearly interpolate over undetected (NaN / non-finite) frames
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros(len(signal))
    if not finite.all():
        x = np.arange(len(signal))
        signal[~finite] = np.interp(x[~finite], x[finite], signal[finite])

    return signal


def build_signal_band_er_side(pose_df: pd.DataFrame) -> np.ndarray:
    """Build a normalised lateral-wrist-displacement signal for Band ER Side (front view).

    Uses the working arm (the one with more lateral wrist travel relative to
    the elbow).  The signal is the absolute lateral displacement of the wrist
    from the elbow, normalised by shoulder width.

    At rest the forearm points forward (wrist ≈ elbow in x), so the signal
    is near zero.  At max external rotation the wrist is furthest from the
    elbow laterally, producing a peak.  Peaks = max outward rotation.

    Args:
        pose_df: Wide-format pose DataFrame from extract_poses().

    Returns:
        1-D float array, one value per frame.  Higher = more outward rotation.
    """
    arm = select_working_arm_band_er(pose_df)

    wrist_x = pose_df[f"{arm}_wrist_x"].values.astype(float)
    elbow_x = pose_df[f"{arm}_elbow_x"].values.astype(float)

    # Shoulder width for normalisation
    left_sh_x = pose_df["left_shoulder_x"].values.astype(float)
    right_sh_x = pose_df["right_shoulder_x"].values.astype(float)
    shoulder_width = np.abs(left_sh_x - right_sh_x)
    med_width = float(np.nanmedian(shoulder_width))
    if not (med_width > 0):
        med_width = 0.2
    shoulder_width = np.where(
        ~np.isfinite(shoulder_width) | (shoulder_width < 1e-3),
        med_width,
        shoulder_width,
    )

    signal = np.abs(wrist_x - elbow_x) / shoulder_width

    # Interpolate NaN / non-finite frames
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros(len(signal))
    if not finite.all():
        x = np.arange(len(signal))
        signal[~finite] = np.interp(x[~finite], x[finite], signal[finite])

    return signal


# ── Smoothing ─────────────────────────────────────────────────────────────────

def smooth_signal(
    signal: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to suppress frame-level jitter.

    Args:
        signal:    1-D float array.
        window:    Window length (forced odd).  Clamped to signal length.
        polyorder: Polynomial order.  Clamped to window - 1.

    Returns:
        Smoothed signal of the same length as the input.
    """
    n = len(signal)
    if n < 3:
        return signal.copy()

    # window must be odd and ≤ signal length
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    p = min(polyorder, w - 1)

    return savgol_filter(signal, window_length=w, polyorder=p)


# ── Peak detection + rep assembly ─────────────────────────────────────────────

def find_rep_boundaries(
    signal: np.ndarray,
    fps: int,
    min_peak_distance_s: float,
    duration_bounds_s: tuple[float, float],
    prominence: float = 0.05,
    min_amplitude_ratio: float = 0.5,
    amplitude_reference_percentile: float = 50.0,
) -> list[Rep]:
    """Detect reps using trough-to-trough boundaries around each peak.

    Peaks in the signal correspond to the top-of-movement.  Between every
    pair of consecutive peaks there is a trough (bottom-of-movement / rest
    position) which becomes the shared boundary between adjacent reps.

    Steps:
        1. Find peaks via scipy.signal.find_peaks.
        2. Identify the trough before the first peak, between each pair of
           consecutive peaks, and after the last peak.
        3. Rep i spans from troughs[i] to troughs[i+1].
        4. Discard reps whose duration falls outside duration_bounds_s.
        5. Discard ghost reps whose signal amplitude (peak − trough within
           the rep) is below ``min_amplitude_ratio`` × a percentile of the
           amplitudes across all duration-valid candidates.

    Args:
        signal:                         Smoothed 1-D signal (higher = arm raised).
        fps:                            Frames per second.
        min_peak_distance_s:            Minimum time between consecutive peaks (s).
        duration_bounds_s:              (min_s, max_s) acceptable rep duration.
        prominence:                     Minimum peak prominence for find_peaks.
        min_amplitude_ratio:            Fraction of the reference amplitude below
                                        which a rep is discarded.
        amplitude_reference_percentile: Which percentile of candidate amplitudes
                                        to use as the reference (50 = median,
                                        robust to normal spread; 75 = p75,
                                        robust when ghost reps dominate the
                                        candidate set and drag the median down).

    Returns:
        List of Rep dataclasses, rep_id assigned sequentially from 0.
    """
    min_distance = max(1, int(min_peak_distance_s * fps))
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)

    if len(peaks) == 0:
        return []

    # ── Build trough list ─────────────────────────────────────────────────────
    # One trough before the first peak, one between each pair, one after last.
    troughs: list[int] = []

    # Before first peak
    troughs.append(int(np.argmin(signal[: peaks[0] + 1])))

    # Between consecutive peaks (inclusive of both peak frames for argmin)
    for i in range(len(peaks) - 1):
        segment = signal[peaks[i] : peaks[i + 1] + 1]
        troughs.append(peaks[i] + int(np.argmin(segment)))

    # After last peak
    troughs.append(peaks[-1] + int(np.argmin(signal[peaks[-1] :])))

    # ── Assemble candidates (duration filter) ────────────────────────────────
    min_dur, max_dur = duration_bounds_s
    candidates: list[tuple[int, int, float]] = []  # (start, end, amplitude)
    for i in range(len(peaks)):
        start = troughs[i]
        end = troughs[i + 1]
        duration_s = (end - start) / fps
        if min_dur <= duration_s <= max_dur:
            seg = signal[start : end + 1]
            amplitude = float(np.max(seg) - np.min(seg))
            candidates.append((start, end, amplitude))

    if not candidates:
        return []

    # ── Ghost-rep filter: discard reps with very low amplitude ────────────────
    # Using a high percentile (e.g. p75) as the reference is robust to ghost
    # reps dominating the candidate set — the median collapses toward ghost
    # amplitudes, but p75 stays anchored to real-rep amplitudes as long as
    # ghosts are < 25% of candidates per quartile.
    amplitudes = np.array([a for _, _, a in candidates])
    reference_amp = float(np.percentile(amplitudes, amplitude_reference_percentile))
    amp_threshold = min_amplitude_ratio * reference_amp

    reps: list[Rep] = []
    for start, end, amplitude in candidates:
        if amplitude >= amp_threshold:
            reps.append(
                Rep(
                    rep_id=len(reps),
                    start_frame=int(start),
                    end_frame=int(end),
                    start_time_s=round(start / fps, 3),
                    end_time_s=round(end / fps, 3),
                )
            )

    return reps


# ── Top-level dispatcher ──────────────────────────────────────────────────────

def segment_reps(
    pose_df: pd.DataFrame,
    exercise: str,
    fps: int,
) -> list[Rep]:
    """Segment reps for a given exercise using pose data.

    Args:
        pose_df:  Wide-format pose DataFrame from extract_poses().
        exercise: Exercise identifier ('wall_slide' or 'band_er_side').
        fps:      Video frame rate.

    Returns:
        List of Rep dataclasses detected in the session.

    Raises:
        NotImplementedError: For 'band_er_side' (support added in Phase 7).
        ValueError:          For unknown exercise identifiers.
    """
    seg_cfg = get_section("segmentation")

    if exercise == "wall_slide":
        signal_raw = build_signal_wall_slide(pose_df)
    elif exercise == "band_er_side":
        signal_raw = build_signal_band_er_side(pose_df)
    else:
        raise ValueError(
            f"Unknown exercise: '{exercise}'. Supported: wall_slide, band_er_side"
        )

    # Trim setup / teardown before peak detection
    start_frame, end_frame = detect_active_window(pose_df, fps)

    cfg = seg_cfg[exercise]
    smoothed = smooth_signal(
        signal_raw,
        window=cfg["smoothing_window"],
        polyorder=cfg["smoothing_polyorder"],
    )

    # Run peak detection on the trimmed window only
    trimmed = smoothed[start_frame : end_frame + 1]
    reps = find_rep_boundaries(
        trimmed,
        fps=fps,
        min_peak_distance_s=cfg["min_peak_distance_s"],
        duration_bounds_s=tuple(cfg["rep_duration_bounds_s"]),
        prominence=cfg.get("prominence", 0.05),
        min_amplitude_ratio=cfg.get("min_amplitude_ratio", 0.5),
        amplitude_reference_percentile=cfg.get("amplitude_reference_percentile", 50.0),
    )

    # Shift frame offsets back to full-video coordinates
    for r in reps:
        r.start_frame += start_frame
        r.end_frame += start_frame
        r.start_time_s = round(r.start_frame / fps, 3)
        r.end_time_s = round(r.end_frame / fps, 3)

    return reps


# ── Debug plot ────────────────────────────────────────────────────────────────

def plot_segmentation(
    signal: np.ndarray,
    reps: list[Rep],
    fps: int,
    title: str,
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    """Generate a debug plot showing the signal and detected rep boundaries.

    Rep spans are shaded; boundary lines are drawn at start and end of each
    rep; rep index labels appear above the midpoint of each span.

    Args:
        signal:    The (smoothed) signal used for segmentation.
        reps:      Detected reps from find_rep_boundaries() or segment_reps().
        fps:       Frames per second (used to convert frame indices to seconds).
        title:     Plot title string.
        save_path: If given, save the figure to this path (PNG).
        show:      If True, display the figure interactively.
    """
    import matplotlib.pyplot as plt

    time = np.arange(len(signal)) / fps
    y_max = float(np.nanmax(signal)) if len(signal) else 1.0
    y_min = float(np.nanmin(signal)) if len(signal) else 0.0
    label_y = y_max + 0.05 * (y_max - y_min + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, signal, color="steelblue", linewidth=1.0, label="Wrist signal")

    palette = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854",
               "#FFD92F", "#E5C494", "#B3B3B3"]
    for i, rep in enumerate(reps):
        color = palette[i % len(palette)]
        ax.axvspan(rep.start_time_s, rep.end_time_s, alpha=0.22, color=color)
        ax.axvline(rep.start_time_s, color="dimgray", linestyle="--", linewidth=0.7)
        ax.axvline(rep.end_time_s, color="dimgray", linestyle="--", linewidth=0.7)
        mid = (rep.start_time_s + rep.end_time_s) / 2.0
        ax.text(mid, label_y, f"#{rep.rep_id}", ha="center", va="bottom",
                fontsize=8, color="black")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised wrist height")
    ax.set_title(f"{title}  —  {len(reps)} rep(s) detected")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_reps_csv(reps: list[Rep], out_path: Path) -> None:
    """Save detected reps to a CSV file.

    Args:
        reps:     List of Rep dataclasses.
        out_path: Destination path; parent directories are created if absent.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _COLS = ["rep_id", "start_frame", "end_frame", "start_time_s", "end_time_s"]
    rows = [
        {
            "rep_id": r.rep_id,
            "start_frame": r.start_frame,
            "end_frame": r.end_frame,
            "start_time_s": r.start_time_s,
            "end_time_s": r.end_time_s,
        }
        for r in reps
    ]
    pd.DataFrame(rows, columns=_COLS).to_csv(out_path, index=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment reps from a pose Parquet file."
    )
    parser.add_argument("--poses", required=True, help="Path to poses.parquet")
    parser.add_argument(
        "--exercise", required=True, choices=["wall_slide", "band_er_side"]
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default 30)")
    parser.add_argument("--output", required=True, help="Path for output reps.csv")
    parser.add_argument(
        "--plot", default=None, help="Path for segmentation plot PNG (optional)"
    )
    args = parser.parse_args()

    from src.pipeline.pose_extract import load_poses

    pose_df = load_poses(Path(args.poses))
    reps = segment_reps(pose_df, args.exercise, args.fps)
    save_reps_csv(reps, Path(args.output))

    if args.plot:
        cfg = get_section("segmentation")[args.exercise]
        if args.exercise == "wall_slide":
            raw = build_signal_wall_slide(pose_df)
        elif args.exercise == "band_er_side":
            raw = build_signal_band_er_side(pose_df)
        else:
            raise ValueError(f"Unknown exercise: {args.exercise}")
        smoothed = smooth_signal(raw, cfg["smoothing_window"], cfg["smoothing_polyorder"])
        plot_segmentation(
            smoothed, reps, args.fps,
            title=f"{args.exercise} — segmentation",
            save_path=Path(args.plot),
        )
        print(f"Plot saved to : {args.plot}")

    print(f"Reps detected : {len(reps)}")
    for rep in reps:
        dur = rep.end_time_s - rep.start_time_s
        print(
            f"  Rep {rep.rep_id}: frames {rep.start_frame}–{rep.end_frame}"
            f"  ({rep.start_time_s:.1f}s – {rep.end_time_s:.1f}s, {dur:.1f}s)"
        )
    print(f"Output saved  : {args.output}")
