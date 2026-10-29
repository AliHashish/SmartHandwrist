"""One core analysis tool; chest/reference signals never enter the estimator.

PPG-DaLiA reference labels correspond to 8-second windows shifting every 2s.
Only the wrist BVP is passed to heart-rate estimation. ACC is used only for
motion-artifact warnings; labels are used *after* prediction for evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import pickle
import re
from typing import Any

import numpy as np
from scipy import signal

PPG_HZ = 64
ACC_HZ = 32
WIN_SECONDS = 8
STEP_SECONDS = 2
MIN_HZ = 0.65
MAX_HZ = 3.5


@dataclass(frozen=True)
class Subject:
    name: str
    bvp: np.ndarray
    acc: np.ndarray
    reference: np.ndarray


def validate_subject(subject: str) -> str:
    if not re.fullmatch(r"S(?:[1-9]|1[0-5])", subject):
        raise ValueError("subject must be S1 through S15")
    return subject


@lru_cache(maxsize=1)
def load_subject(path_str: str) -> Subject:
    """Load a *trusted* official synchronized subject .pkl.

    WARNING: pickle deserialization can execute arbitrary code. Run only with
    an official downloaded dataset in an isolated environment. Never accept
    arbitrary user-supplied pickle paths from the LLM/user interface.
    """
    path = Path(path_str)
    name = validate_subject(path.stem)
    if path.name != f"{name}.pkl":
        raise ValueError("Unexpected file name")
    with path.open("rb") as file:
        record = pickle.load(file, encoding="latin1")  # trusted source only
    bvp = np.asarray(record["signal"]["wrist"]["BVP"], dtype=float).reshape(-1)
    acc = np.asarray(record["signal"]["wrist"]["ACC"], dtype=float)
    reference = np.asarray(record["label"], dtype=float).reshape(-1)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"Unexpected ACC shape: {acc.shape}")
    if bvp.size < WIN_SECONDS * PPG_HZ or acc.shape[0] < WIN_SECONDS * ACC_HZ:
        raise ValueError("Insufficient wrist data")
    if reference.size < 1:
        raise ValueError("No reference labels")
    return Subject(name, bvp, acc, reference)


def _band_peak(x: np.ndarray, fs: int, *, filtered: bool) -> tuple[float, float, float]:
    """Return (BPM, peak-share, peak-frequency-Hz). No reference inputs.

    The peak-share is a spectral concentration heuristic, NOT a calibrated
    probability or medical confidence score.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size < fs * 4 or not np.all(np.isfinite(x)) or np.std(x) < 1e-9:
        return math.nan, math.nan, math.nan
    if filtered:
        f, p = signal.welch(x, fs=fs, window="hann", nperseg=x.size,
                            nfft=4096, detrend="constant")
    else:
        f, p = signal.periodogram(x, fs=fs, window="hann", nfft=4096,
                                  detrend="constant")
    band = (f >= MIN_HZ) & (f <= MAX_HZ)
    fb, pb = f[band], p[band]
    if not pb.size or np.sum(pb) <= 1e-12:
        return math.nan, math.nan, math.nan
    peak_idx = int(np.argmax(pb))
    peak_hz = float(fb[peak_idx])
    concentration = float(np.sum(pb[np.abs(fb - peak_hz) <= 0.1]) / np.sum(pb))
    return round(60.0 * peak_hz, 2), round(concentration, 3), peak_hz


def estimate_hr_from_ppg(ppg_window: np.ndarray, *, prefiltered: bool = False) -> tuple[float, float, float]:
    """PPG-only pulse estimation; reference and ECG are not accepted inputs.

    For `prefiltered=False`, computes a raw spectrum baseline. For
    `prefiltered=True`, expects caller to have applied the bandpass to a longer
    BVP segment to avoid per-window filtering edge artifacts.
    """
    return _band_peak(ppg_window, PPG_HZ, filtered=prefiltered)


def filter_bvp(bvp: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [MIN_HZ, MAX_HZ], btype="bandpass", fs=PPG_HZ,
                        output="sos")
    return signal.sosfiltfilt(sos, np.asarray(bvp, dtype=float))


def motion_indicators(acc_window: np.ndarray, ppg_peak_hz: float) -> tuple[float, bool | None]:
    """ACC RMS is in uncalibrated sensor units, not a clinical movement score."""
    if not np.isfinite(acc_window).all():
        return math.nan, None
    demeaned = acc_window - np.mean(acc_window, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(demeaned**2, axis=1))))
    if not np.isfinite(ppg_peak_hz):
        return round(rms, 4), None
    # Any strong accelerometer axis at a similar frequency is a warning,
    # not proof that the PPG peak is motion-driven.
    overlap = False
    for axis in range(3):
        axis_signal = demeaned[:, axis]
        if np.std(axis_signal) < 1e-9:
            continue
        f, p = signal.periodogram(axis_signal, fs=ACC_HZ, nfft=2048,
                                  window="hann", detrend=False)
        band = (f >= MIN_HZ) & (f <= MAX_HZ)
        if p[band].size and np.sum(p[band]) > 1e-10:
            axis_peak = float(f[band][np.argmax(p[band])])
            overlap |= abs(axis_peak - ppg_peak_hz) < 0.12
    return round(rms, 4), bool(overlap)


def _summary(errors: list[float]) -> dict[str, Any]:
    if not errors:
        return {"mae_bpm": None, "mean_signed_error_bpm": None, "evaluated_windows": 0}
    e = np.asarray(errors, dtype=float)
    return {"mae_bpm": round(float(np.mean(np.abs(e))), 2),
            "mean_signed_error_bpm": round(float(np.mean(e)), 2),
            "evaluated_windows": len(errors)}


def analyze_loaded_subject(record: Subject, start_seconds: float = 0,
                           duration_seconds: float = 120,
                           *, return_trace: bool = False) -> dict[str, Any]:
    """Core tool: process wrist-only estimates, then evaluate against labels.

    Selection includes full 8-s windows whose starts are 2s apart.
    """
    if not 0 <= start_seconds < 24 * 3600 or not 8 <= duration_seconds <= 300:
        raise ValueError("start must be non-negative and duration 8–300 seconds")
    max_time = min(len(record.bvp) / PPG_HZ,
                   len(record.acc) / ACC_HZ)
    end_seconds = min(float(start_seconds + duration_seconds), max_time)
    left = max(0, math.ceil(start_seconds / STEP_SECONDS))
    right = min(len(record.reference) - 1,
                math.floor((end_seconds - WIN_SECONDS) / STEP_SECONDS))
    if right < left:
        raise ValueError("No complete, reference-aligned 8-second windows in selection")

    # Filter full BVP series, not independently for each tiny window. Detect
    # nonfinite values explicitly instead of silently filling a recording.
    if not np.all(np.isfinite(record.bvp)):
        raise ValueError("BVP contains nonfinite values; select a clean recording")
    filtered = filter_bvp(record.bvp)

    rows = []
    raw_errors, filtered_errors = [], []
    for idx in range(left, right + 1):
        start = idx * STEP_SECONDS
        i, j = start * PPG_HZ, (start + WIN_SECONDS) * PPG_HZ
        ia, ja = start * ACC_HZ, (start + WIN_SECONDS) * ACC_HZ
        bvp_raw = record.bvp[i:j]
        bvp_clean = filtered[i:j]
        acc = record.acc[ia:ja]
        raw_bpm, _, _ = estimate_hr_from_ppg(bvp_raw)
        filt_bpm, share, peak_hz = estimate_hr_from_ppg(bvp_clean, prefiltered=True)
        motion_rms, motion_overlap = motion_indicators(acc, peak_hz)
        ref = float(record.reference[idx])
        valid_ref = np.isfinite(ref) and 30 <= ref <= 240
        if valid_ref and np.isfinite(raw_bpm):
            raw_errors.append(raw_bpm - ref)
        if valid_ref and np.isfinite(filt_bpm):
            filtered_errors.append(filt_bpm - ref)
        rows.append({"start_seconds": start,
                     "midpoint_seconds": start + WIN_SECONDS / 2,
                     "raw_bpm": float(raw_bpm) if np.isfinite(raw_bpm) else None,
                     "filtered_bpm": float(filt_bpm) if np.isfinite(filt_bpm) else None,
                     "reference_bpm": round(ref, 2) if valid_ref else None,
                     "spectral_peak_share": share if np.isfinite(share) else None,
                     "motion_rms_relative_units": motion_rms if np.isfinite(motion_rms) else None,
                     "motion_frequency_overlap": motion_overlap})

    finite_filtered = [r["filtered_bpm"] for r in rows if r["filtered_bpm"] is not None]
    finite_reference = [r["reference_bpm"] for r in rows if r["reference_bpm"] is not None]
    peak_shares = [r["spectral_peak_share"] for r in rows if r["spectral_peak_share"] is not None]
    overlap_values = [r["motion_frequency_overlap"] for r in rows if r["motion_frequency_overlap"] is not None]
    result = {
        "subject": record.name,
        "selected_start_seconds": float(start_seconds),
        "selected_end_seconds": float(end_seconds),
        "complete_windows": len(rows),
        "window_seconds": WIN_SECONDS,
        "stride_seconds": STEP_SECONDS,
        "estimator_input": "wrist BVP only; ACC for contextual flags only",
        "reference_usage": "comparison and evaluation only; never passed to estimator",
        "raw_periodogram": _summary(raw_errors),
        "filtered_welch": _summary(filtered_errors),
        "mean_estimate_bpm": round(float(np.mean(finite_filtered)), 2) if finite_filtered else None,
        "mean_reference_bpm": round(float(np.mean(finite_reference)), 2) if finite_reference else None,
        "median_peak_share_heuristic": round(float(np.median(peak_shares)), 3) if peak_shares else None,
        "motion_frequency_overlap_percent": round(100 * sum(overlap_values) / len(overlap_values), 1) if overlap_values else None,
        "invalid_reference_windows": len(rows) - len(finite_reference),
        "limitations": [
            "This is a simple frequency-peak baseline; motion and harmonics can dominate the PPG spectrum.",
            "Spectral peak-share and accelerometer overlap are heuristics, not calibrated confidence probabilities.",
            "Results cover only selected windows of one participant, not a subject-independent evaluation.",
            "Window errors are correlated because windows overlap by 6 seconds.",
            "Heart-rate estimates are not medical measurements or diagnoses.",
        ],
    }
    if return_trace:
        result["windows"] = rows
        # Caller may use signals for plotting; never pass raw signals to an LLM.
        result["_plot_signals"] = {"bvp": record.bvp, "filtered": filtered,
                                    "acc": record.acc}
    return result


def analyze_segment(data_root: str | Path, subject: str = "S1",
                    start_seconds: float = 0, duration_seconds: float = 120,
                    *, return_trace: bool = False) -> dict[str, Any]:
    """Single analysis function usable directly or through an AI tool call."""
    subject = validate_subject(subject)
    root = Path(data_root).expanduser().resolve()
    path = root / subject / f"{subject}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Extract official UCI data.zip first.")
    return analyze_loaded_subject(load_subject(str(path)), float(start_seconds),
                                  float(duration_seconds), return_trace=return_trace)
