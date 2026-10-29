"""Synthetic-only smoke-test fixture. Never use for PPG-DaLiA results."""
from __future__ import annotations
import numpy as np
from .analysis import PPG_HZ, ACC_HZ, WIN_SECONDS, STEP_SECONDS, Subject, validate_subject


def synthetic_record(seconds: int = 1100, subject: str = 'S1', bpm: float = 90.0,
                     movement_from_seconds: float = 600) -> Subject:
    """Known-frequency pulse + optional later movement artifact, labeled SYNTHETIC."""
    validate_subject(subject)
    if seconds < WIN_SECONDS + STEP_SECONDS:
        raise ValueError('Record must contain at least one 8-second window')
    rng = np.random.default_rng(42)
    t = np.arange(seconds * PPG_HZ) / PPG_HZ
    ppg = np.sin(2 * np.pi * (bpm / 60) * t) + 0.08 * rng.normal(size=t.size)
    motion_mask = t >= movement_from_seconds
    ppg[motion_mask] += 2.5 * np.sin(2 * np.pi * 1.8 * t[motion_mask])
    a = np.arange(seconds * ACC_HZ) / ACC_HZ
    motion_a = a >= movement_from_seconds
    acc = np.stack((.01 * np.sin(2*np.pi*.4*a),
                    .012 * np.sin(2*np.pi*.6*a),
                    .01 * np.sin(2*np.pi*.3*a)), axis=1)
    acc[motion_a, 0] += 0.7*np.sin(2*np.pi*1.8*a[motion_a])
    acc[motion_a, 1] += 0.4*np.cos(2*np.pi*1.8*a[motion_a])
    n = int((seconds - WIN_SECONDS) // STEP_SECONDS) + 1
    labels = np.full(n, bpm)
    return Subject(subject, ppg, acc, labels)
