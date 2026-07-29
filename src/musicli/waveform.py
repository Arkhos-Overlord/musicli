"""Pre-calculated waveform peak generation for visualisation."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

WAVEFORM_COLUMNS = 80


def compute_waveform(
    filepath: str,
    columns: int = WAVEFORM_COLUMNS,
) -> Optional[np.ndarray]:
    """Read an audio file and compute RMS-normalised amplitude peaks.

    Returns a 1-D float32 array of shape (columns,) in [0, 1], or None.
    """
    try:
        import soundfile as sf
        data, _sr = sf.read(filepath, dtype="float32", always_2d=True)
    except Exception as exc:
        logger.warning("Could not read %s for waveform: %s", filepath, exc)
        return None

    samples = len(data)
    if samples < columns:
        return None

    mono = data.mean(axis=1)
    chunk_size = samples // columns
    peaks = np.zeros(columns, dtype=np.float32)

    for i in range(columns):
        chunk = mono[i * chunk_size : (i + 1) * chunk_size]
        if len(chunk):
            peaks[i] = np.max(np.abs(chunk))

    peak_max = peaks.max()
    if peak_max > 0:
        peaks /= peak_max

    return peaks


def precompute_waveform_async(
    filepath: str,
    callback: Callable[[np.ndarray], None],
    columns: int = WAVEFORM_COLUMNS,
    generation: int = 0,
    generation_check: Optional[Callable[[], int]] = None,
) -> None:
    """Compute waveform in a background thread; drop stale results via generation."""

    def _worker() -> None:
        try:
            peaks = compute_waveform(filepath, columns)
            result = peaks if peaks is not None else np.zeros(0, dtype=np.float32)
        except Exception:
            result = np.zeros(0, dtype=np.float32)

        if generation_check is not None and generation_check() != generation:
            return
        try:
            callback(result)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()
