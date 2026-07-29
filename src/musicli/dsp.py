"""High-fidelity DSP pipeline: Parametric EQ, ReplayGain, TPDF dithering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


# ── Biquad Filter Design (RBJ Audio EQ Cookbook via bilinear transform) ─────


def _design_biquad(
    b_analog: np.ndarray,
    a_analog: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Convert analog biquad coefficients to digital SOS via bilinear transform."""
    # Normalize constants so b[0] / a[0] = 1.0
    g = b_analog[0] / a_analog[0]
    zeros = np.roots(b_analog)
    poles = np.roots(a_analog)
    zpk = signal.bilinear_zpk(zeros, poles, g, fs)
    sos = signal.zpk2sos(*zpk)
    return sos


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

    b = np.array([
        A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha),
        2 * A * ((A - 1) - (A + 1) * np.cos(w0)),
        A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha),
    ])
    a = np.array([
        (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha,
        -2 * ((A - 1) + (A + 1) * np.cos(w0)),
        (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha,
    ])
    return _design_biquad(b, a, fs)


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

    b = np.array([
        1 + alpha * A,
        -2 * np.cos(w0),
        1 - alpha * A,
    ])
    a = np.array([
        1 + alpha / A,
        -2 * np.cos(w0),
        1 - alpha / A,
    ])
    return _design_biquad(b, a, fs)


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

    b = np.array([
        A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha),
        -2 * A * ((A - 1) + (A + 1) * np.cos(w0)),
        A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha),
    ])
    a = np.array([
        (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha,
        2 * ((A - 1) - (A + 1) * np.cos(w0)),
        (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha,
    ])
    return _design_biquad(b, a, fs)


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
        sos_low = design_low_shelf(freq=150.0, gain_db=bass_db, q=0.7, fs=fs)
        sos_peak = design_peaking(freq=1000.0, gain_db=mid_db, q=1.0, fs=fs)
        sos_high = design_high_shelf(freq=8000.0, gain_db=treble_db, q=0.7, fs=fs)
        sos = np.vstack([sos_low, sos_peak, sos_high])
        # zi per channel = n_sections × 2
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
            return out
        else:
            # Deinterleave stereo, filter each channel, re-interleave
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
        # Strip ' dB' suffix
        return float(str(tag_str).replace(" dB", "").strip())
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
    # Use in-place multiplication for speed
    np.multiply(samples, linear, out=samples)
    return samples


# ── TPDF Dithering ──────────────────────────────────────────────────────────


def apply_tpdf_dither(samples: np.ndarray) -> np.ndarray:
    """Apply Triangular PDF dither to float32 audio, returning int16.

    TPDF = sum of two uniform random variables, giving a triangular
    distribution with ±1 LSB range — eliminates harmonic distortion
    from quantization while adding only constant-noise-floor hiss.
    """
    scale = 32767.0
    scaled = samples * scale

    # Two independent uniform distributions [−0.5, 0.5)
    rng = np.random.default_rng()
    noise1 = rng.random(len(scaled), dtype=np.float32) - 0.5
    noise2 = rng.random(len(scaled), dtype=np.float32) - 0.5

    tpdf_noise = noise1 + noise2  # triangular, range [−1.0, 1.0) LSB

    dithered = np.round(scaled + tpdf_noise)
    return np.clip(dithered, -32768, 32767).astype(np.int16)



