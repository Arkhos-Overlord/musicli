"""Pre-calculated waveform peak generation for visualisation."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# How many amplitude columns for the waveform display
WAVEFORM_COLUMNS = 80


def compute_waveform(
    filepath: str,
    columns: int = WAVEFORM_COLUMNS,
) -> Optional[np.ndarray]:
    """Read an audio file with soundfile and compute RMS-normalised
    amplitude peaks across `columns` bins.

    Returns a 1-D float32 array of shape (columns,) with values in [0, 1],
    or None if reading fails.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    except Exception as exc:
        logger.warning("Could not read %s for waveform: %s", filepath, exc)
        return None

    samples = len(data)
    if samples < columns:
        return None

    # Average L+R to mono
    mono = data.mean(axis=1)

    # Split into `columns` chunks and take max absolute amplitude per chunk
    chunk_size = samples // columns
    peaks = np.zeros(columns, dtype=np.float32)

    for i in range(columns):
        chunk = mono[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk):
            peaks[i] = np.max(np.abs(chunk))

    # Normalise to [0, 1]
    peak_max = peaks.max()
    if peak_max > 0:
        peaks /= peak_max

    return peaks


# ── Background pre-calculation ──────────────────────────────────────────────


def precompute_waveform_async(
    filepath: str,
    callback,
    columns: int = WAVEFORM_COLUMNS,
) -> None:
    """Compute waveform in a background thread; call `callback(peaks)` when done."""

    def _worker() -> None:
        try:
            peaks = compute_waveform(filepath, columns)
            callback(peaks if peaks is not None else np.zeros(0))
        except Exception:
            callback(np.zeros(0))

    threading.Thread(target=_worker, daemon=True).start()
