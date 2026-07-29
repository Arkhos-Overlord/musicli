"""High-fidelity DSP pipeline: Parametric EQ, ReplayGain, TPDF dithering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


# ── Biquad Filter Design (RBJ Audio EQ Cookbook — digital coefficients) ─────
# These formulas already produce digital b/a coeffs (w0 = 2πf/fs).
# Convert to SOS with tf2sos — do NOT apply bilinear again.


def design_low_shelf(
    freq: float = 150.0,
    gain_db: float = 0.0,
    q: float = 0.7,
    fs: float = 96000.0,
) -> np.ndarray:
    """Low-shelf biquad. freq=corner, gain_db=boost/cut, q=slope."""
    w0 = 2.0 * np.pi * freq / fs
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    return signal.tf2sos([b0, b1, b2], [a0, a1, a2])


def design_peaking(
    freq: float = 1000.0,
    gain_db: float = 0.0,
    q: float = 1.0,
    fs: float = 96000.0,
) -> np.ndarray:
    """Peaking/band EQ biquad."""
    w0 = 2.0 * np.pi * freq / fs
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A

    return signal.tf2sos([b0, b1, b2], [a0, a1, a2])


def design_high_shelf(
    freq: float = 8000.0,
    gain_db: float = 0.0,
    q: float = 0.7,
    fs: float = 96000.0,
) -> np.ndarray:
    """High-shelf biquad."""
    w0 = 2.0 * np.pi * freq / fs
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    return signal.tf2sos([b0, b1, b2], [a0, a1, a2])


# ── EQ Pipeline ─────────────────────────────────────────────────────────────


@dataclass
class EQPipeline:
    """Pre-computed 3-band EQ cascade (low-shelf + peaking + high-shelf)."""

    sos: np.ndarray  # combined SOS matrix
    zi_per_channel: list  # filter state for each channel
    fs: float = 96000.0

    @classmethod
    def create(
        cls,
        bass_db: float = 0.0,
        mid_db: float = 0.0,
        treble_db: float = 0.0,
        fs: float = 96000.0,
    ) -> EQPipeline:
        """Pre-compute the cascaded SOS filters for given EQ settings."""
        # Bypass identity filter when all bands are flat (cheaper + no phase shift)
        if bass_db == 0.0 and mid_db == 0.0 and treble_db == 0.0:
            sos = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        else:
            sos_low = design_low_shelf(freq=150.0, gain_db=bass_db, q=0.7, fs=fs)
            sos_peak = design_peaking(freq=1000.0, gain_db=mid_db, q=1.0, fs=fs)
            sos_high = design_high_shelf(freq=8000.0, gain_db=treble_db, q=0.7, fs=fs)
            sos = np.vstack([sos_low, sos_peak, sos_high])
        n_sections = sos.shape[0]
        return cls(
            sos=sos,
            zi_per_channel=[np.zeros((n_sections, 2)) for _ in range(2)],
            fs=fs,
        )

    def process(self, samples: np.ndarray, channels: int = 2) -> np.ndarray:
        """Apply the EQ cascade to a mono or interleaved-stereo buffer.

        `samples` is a 1D float32 array; if stereo it's interleaved L,R,L,R,...
        Returns the filtered array (same shape).
        """
        if channels == 1:
            out, self.zi_per_channel[0] = signal.sosfilt(
                self.sos, samples, zi=self.zi_per_channel[0]
            )
            return out.astype(np.float32, copy=False)
        else:
            left = samples[0::2]
            right = samples[1::2]
            left_out, self.zi_per_channel[0] = signal.sosfilt(
                self.sos, left, zi=self.zi_per_channel[0]
            )
            right_out, self.zi_per_channel[1] = signal.sosfilt(
                self.sos, right, zi=self.zi_per_channel[1]
            )
            out = np.empty(len(samples), dtype=np.float32)
            out[0::2] = left_out
            out[1::2] = right_out
            return out


# ── ReplayGain ──────────────────────────────────────────────────────────────


def parse_replaygain_db(tag_str: str | None) -> float:
    """Parse a ReplayGain tag string like '-4.53 dB' into a float. Returns 0.0 on failure."""
    if tag_str is None:
        return 0.0
    try:
        return float(str(tag_str).replace(" dB", "").replace("db", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def apply_replaygain(
    samples: np.ndarray,
    gain_db: float,
) -> np.ndarray:
    """Multiply float32 audio by the linear gain factor from a dB value."""
    if gain_db == 0.0:
        return samples
    linear = 10.0 ** (gain_db / 20.0)
    np.multiply(samples, linear, out=samples)
    return samples


# ── Soft clip / output ──────────────────────────────────────────────────────


def soft_clip_float32(samples: np.ndarray) -> np.ndarray:
    """Clamp float32 samples to [-1, 1] with a soft knee near the rails.

    Keeps output in FLOAT32 device format (no int16 quantization).
    Soft clip prevents hard digital clipping when EQ/ReplayGain push peaks.
    """
    # tanh soft-clip only samples that would clip; leave the rest alone
    abs_s = np.abs(samples)
    mask = abs_s > 0.95
    if np.any(mask):
        out = samples.copy()
        # Map overshoot into a gentle curve approaching ±1
        over = out[mask]
        sign = np.sign(over)
        mag = np.abs(over)
        # Blend linear → tanh beyond 0.95
        t = (mag - 0.95) / 0.05  # 0 at 0.95, 1 at 1.0+
        t = np.clip(t, 0.0, 1.0)
        soft = np.tanh(mag)
        out[mask] = sign * ((1.0 - t) * mag + t * soft)
        return np.clip(out, -1.0, 1.0).astype(np.float32)
    return np.clip(samples, -1.0, 1.0).astype(np.float32)


def apply_tpdf_dither(samples: np.ndarray) -> np.ndarray:
    """Legacy helper: soft-clip float32 for device output.

    Kept for import compatibility. Device is FLOAT32 so we no longer
    quantize to int16 (that was a format mismatch bug).
    """
    return soft_clip_float32(samples)
