"""Audio playback engine: miniaudio decoder, DSP pipeline, FFT data."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import miniaudio
import numpy as np

from musicli.dsp import EQPipeline, soft_clip_float32

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

TARGET_SAMPLE_RATE = 96000  # High-Fidelity standard for clean, crisp audio
CHANNELS = 2
BYTES_PER_SAMPLE = 4  # float32
BUFFER_FRAMES = 1024
CROSSFADE_FRAMES = 2000  # ~21ms at 96kHz


# ── Engine ──────────────────────────────────────────────────────────────────


class AudioEngine:
    """Manages miniaudio playback with integrated DSP and FFT data sharing.

    Supports gapless playback via dual-decoder architecture: while the
    current track plays, the next track is pre-decoded and stored in
    `_next_decoder`. The callback automatically switches decoders when
    the current one runs dry, enabling zero-gap transitions.
    """

    def __init__(self) -> None:
        self._device: Optional[miniaudio.PlaybackDevice] = None
        self._current_decoder: Optional[miniaudio.DecodedSoundFile] = None
        self._next_decoder: Optional[miniaudio.DecodedSoundFile] = None
        self._next_path: str = ""
        self._next_replaygain_db: float = 0.0
        self._next_duration: float = 0.0
        self._lock = threading.RLock()

        # ── Playback state ──────────────────────────────────────────────
        self._playing = False
        self._paused = False
        self._volume: float = 0.8
        self._muted: bool = False

        # ── DSP state ───────────────────────────────────────────────────
        self._eq: EQPipeline = EQPipeline.create(fs=TARGET_SAMPLE_RATE)
        self._replaygain_db: float = 0.0

        # ── Crossfade state ───────────────────────────────────────────
        self._crossfade_enabled: bool = True
        self._crossfade_remaining: int = 0
        self._tail_buffer = np.zeros(CROSSFADE_FRAMES * CHANNELS, dtype=np.float32)
        self._tail_write: int = 0

        # ── Track info ──────────────────────────────────────────────────
        self._current_path: str = ""
        self._duration: float = 0.0
        self._position_sec: float = 0.0

        # ── Auto-switch notification (gapless) ─────────────────────────
        self._just_switched: bool = False
        self._switched_to_path: str = ""

        # ── FFT / visualiser data ──────────────────────────────────────
        self._fft_buffer = np.zeros(BUFFER_FRAMES * CHANNELS, dtype=np.float32)
        self._fft_lock = threading.Lock()

        # ── Decoded sample cursor (for DecodedSoundFile which is not a stream) ─
        self._sample_pos: int = 0  # index into interleaved float samples
        self._samples: Optional[np.ndarray] = None
        self._next_samples: Optional[np.ndarray] = None
        self._next_sample_pos: int = 0

    # ── Public properties ─────────────────────────────────────────────────

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def current_path(self) -> str:
        return self._current_path

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        with self._lock:
            return self._position_sec

    @property
    def volume(self) -> float:
        return self._volume

    # ── FFT data access ──────────────────────────────────────────────────

    def get_fft_data(self) -> np.ndarray:
        """Return mono float32 array of last frame for FFT visualisation."""
        with self._fft_lock:
            frame = self._fft_buffer.copy()
        mono = (frame[0::2] + frame[1::2]) * 0.5
        return mono

    def get_fft_stereo(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return left and right channels separately for stereo FFT."""
        with self._fft_lock:
            frame = self._fft_buffer.copy()
        return frame[0::2].copy(), frame[1::2].copy()

    # ── Track loading ────────────────────────────────────────────────────

    def _decode_file(self, path: str) -> Tuple[np.ndarray, float]:
        """Decode an audio file to interleaved float32 stereo at target rate.

        Returns (samples, duration_seconds).
        """
        decoded = miniaudio.decode_file(
            path,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=CHANNELS,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        # decoded.samples is array.array or bytes-like of float32
        samples = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        # Ensure even length (complete frames)
        if len(samples) % CHANNELS:
            samples = samples[: len(samples) - (len(samples) % CHANNELS)]
        duration = len(samples) / (TARGET_SAMPLE_RATE * CHANNELS)
        return samples, duration

    def load_track(self, path: str) -> bool:
        """Load an audio file for playback. Returns True on success."""
        try:
            samples, duration = self._decode_file(path)
        except Exception as exc:
            logger.error("Failed to decode %s: %s", path, exc)
            return False

        with self._lock:
            self._samples = samples
            self._sample_pos = 0
            self._current_path = path
            self._duration = duration
            self._position_sec = 0.0
            self._next_samples = None
            self._next_sample_pos = 0
            self._next_path = ""
            self._next_replaygain_db = 0.0
            self._next_duration = 0.0
            self._crossfade_remaining = 0
            self._tail_write = 0
            self._current_decoder = None  # unused; kept for API compat
            self._next_decoder = None

        logger.info(
            "Loaded: %s (%.1fs, %d Hz, %d ch)",
            Path(path).name,
            duration,
            TARGET_SAMPLE_RATE,
            CHANNELS,
        )
        return True

    def preload_next(self, path: str, replaygain_db: float = 0.0) -> bool:
        """Pre-decode the next track for gapless playback."""
        try:
            samples, duration = self._decode_file(path)
        except Exception as exc:
            logger.debug("Failed to preload %s: %s", path, exc)
            return False

        with self._lock:
            self._next_samples = samples
            self._next_sample_pos = 0
            self._next_path = path
            self._next_replaygain_db = replaygain_db
            self._next_duration = duration

        logger.debug("Preloaded next: %s", Path(path).name)
        return True

    def has_just_switched(self) -> Tuple[bool, str]:
        """Check if the callback auto-switched to a preloaded track.

        Returns (switched, path). Reading this clears the flag.
        """
        with self._lock:
            switched = self._just_switched
            path = self._switched_to_path
            self._just_switched = False
            self._switched_to_path = ""
        return switched, path

    # ── Playback control ─────────────────────────────────────────────────

    def play(self) -> None:
        """Start or resume playback."""
        if self._device is None or not self._device.running:
            self._start_device()
        self._playing = True
        self._paused = False

    def pause(self) -> None:
        """Toggle pause state."""
        self._paused = not self._paused
        logger.debug("Paused: %s", self._paused)

    def resume(self) -> None:
        """Resume if paused; start if stopped with a loaded track."""
        if self._samples is None:
            return
        if self._device is None or not self._device.running:
            self._start_device()
        self._playing = True
        self._paused = False

    def stop(self) -> None:
        """Stop playback and release device."""
        self._playing = False
        self._paused = False
        with self._lock:
            self._next_samples = None
            self._next_path = ""
            self._next_duration = 0.0
            self._crossfade_remaining = 0
            self._tail_write = 0
            self._position_sec = 0.0
        if self._device and self._device.running:
            try:
                self._device.stop()
                self._device.close()
            except Exception:
                pass
            self._device = None

    def seek(self, seconds: float) -> None:
        """Seek to an absolute position in seconds."""
        with self._lock:
            if self._samples is None:
                return
            total_frames = len(self._samples) // CHANNELS
            frame = int(max(0.0, min(seconds, self._duration)) * TARGET_SAMPLE_RATE)
            frame = min(frame, max(0, total_frames - 1))
            self._sample_pos = frame * CHANNELS
            self._position_sec = frame / TARGET_SAMPLE_RATE
            n_sections = self._eq.sos.shape[0]
            self._eq.zi_per_channel = [
                np.zeros((n_sections, 2)) for _ in range(2)
            ]

    def set_volume(self, vol: float) -> None:
        """Set volume 0.0–1.0 (linear)."""
        self._volume = max(0.0, min(1.0, vol))

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def set_eq(self, bass_db: float, mid_db: float, treble_db: float) -> None:
        """Update EQ settings. Replaces pipeline under lock."""
        new_eq = EQPipeline.create(
            bass_db=bass_db,
            mid_db=mid_db,
            treble_db=treble_db,
            fs=TARGET_SAMPLE_RATE,
        )
        with self._lock:
            self._eq = new_eq

    def set_replaygain(self, gain_db: float) -> None:
        """Set the ReplayGain value for the current track."""
        self._replaygain_db = gain_db

    def set_crossfade(self, enabled: bool) -> None:
        """Enable or disable crossfade transitions."""
        self._crossfade_enabled = enabled

    def compute_rms_normalization(self) -> None:
        """Compute RMS normalization gain when no ReplayGain tags exist."""
        if self._replaygain_db != 0.0:
            return
        with self._lock:
            samples = self._samples
        if samples is None or len(samples) == 0:
            return
        try:
            # First ~5 seconds
            max_samples = int(5.0 * TARGET_SAMPLE_RATE * CHANNELS)
            chunk = samples[: min(max_samples, len(samples))]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > 1e-9:
                target_rms = 10.0 ** (-18.0 / 20.0)
                gain_db = 20.0 * np.log10(target_rms / rms)
                # Clamp extreme gains
                gain_db = max(-20.0, min(20.0, gain_db))
                self._replaygain_db = gain_db
                logger.info("RMS normalization: %.1f dB", gain_db)
        except Exception as exc:
            logger.debug("RMS normalization failed: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────

    def _start_device(self) -> None:
        """Create and start the miniaudio playback device."""
        if self._device:
            try:
                if self._device.running:
                    self._device.stop()
                self._device.close()
            except Exception:
                pass

        self._device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=CHANNELS,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        gen = self._callback_generator()
        next(gen)  # prime
        self._device.start(gen)

    def _read_frames(self, frame_count: int) -> Tuple[Optional[np.ndarray], bool]:
        """Read frame_count frames from current sample buffer.

        Returns (interleaved float32 samples or None if exhausted, ended).
        """
        if self._samples is None:
            return None, True
        needed = frame_count * CHANNELS
        pos = self._sample_pos
        end = len(self._samples)
        if pos >= end:
            return None, True
        chunk = self._samples[pos : pos + needed]
        self._sample_pos = pos + len(chunk)
        ended = self._sample_pos >= end
        if len(chunk) < needed:
            pad = np.zeros(needed - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        return chunk, ended

    def _switch_to_next(self) -> bool:
        """Switch to preloaded next track. Returns True if switched."""
        if self._next_samples is None:
            return False
        do_crossfade = self._crossfade_enabled and self._tail_write > 0
        if do_crossfade:
            cf_frames = min(CROSSFADE_FRAMES, self._tail_write // CHANNELS)
            self._crossfade_remaining = cf_frames
        else:
            self._crossfade_remaining = 0

        self._samples = self._next_samples
        self._sample_pos = 0
        self._current_path = self._next_path
        self._duration = self._next_duration
        self._position_sec = 0.0
        self._replaygain_db = self._next_replaygain_db
        self._just_switched = True
        self._switched_to_path = self._next_path

        self._next_samples = None
        self._next_sample_pos = 0
        self._next_path = ""
        self._next_duration = 0.0
        self._next_replaygain_db = 0.0
        self._tail_write = 0

        n_sections = self._eq.sos.shape[0]
        self._eq.zi_per_channel = [
            np.zeros((n_sections, 2)) for _ in range(2)
        ]
        return True

    def _callback_generator(self):
        """Generator-based audio callback for miniaudio.
        
        Single-yield pattern:
        1. First call to next(gen) primes it.
        2. The driver calls .send(frame_count), which returns the value of the yield.
        3. The generator resumes, calculates the NEXT output, and yields it.
        """
        # Initial priming buffer
        current_output = b"\x00" * (BUFFER_FRAMES * CHANNELS * BYTES_PER_SAMPLE)
        
        while True:
            # The driver sends frame_count via .send()
            frame_count = yield current_output
            
            # Handle edge cases where frame_count might be None or 0
            if frame_count is None or frame_count <= 0:
                frame_count = BUFFER_FRAMES
                
            needed_bytes = frame_count * CHANNELS * BYTES_PER_SAMPLE

            with self._lock:
                paused = self._paused
                playing = self._playing
                samples_loaded = self._samples is not None

            if not playing or paused or not samples_loaded:
                current_output = b"\x00" * needed_bytes
                continue

            with self._lock:
                chunk, ended = self._read_frames(frame_count)
                switched = False
                tail_for_blend: Optional[np.ndarray] = None

                if chunk is None or (ended and self._sample_pos >= len(self._samples or [])):
                    if self._next_samples is not None:
                        if self._crossfade_enabled and self._tail_write > 0:
                            cf_frames = min(
                                CROSSFADE_FRAMES, self._tail_write // CHANNELS
                            )
                            cf_len = cf_frames * CHANNELS
                            src_end = self._tail_write
                            src_start = max(0, src_end - cf_len)
                            tail_for_blend = self._tail_buffer[src_start:src_end].copy()
                        
                        if self._switch_to_next():
                            switched = True
                            chunk, ended = self._read_frames(frame_count)
                        else:
                            self._playing = False
                            current_output = b"\x00" * needed_bytes
                            continue
                    else:
                        self._playing = False
                        current_output = b"\x00" * needed_bytes
                        continue

                if chunk is None:
                    self._playing = False
                    current_output = b"\x00" * needed_bytes
                    continue

                # Crossfade blend
                if switched and tail_for_blend is not None and self._crossfade_remaining > 0:
                    actual_frames = len(chunk) // CHANNELS
                    blend_frames = min(actual_frames, self._crossfade_remaining)
                    blend_len = blend_frames * CHANNELS
                    if blend_len > 0 and len(tail_for_blend) >= blend_len:
                        ramp = np.linspace(0.0, 1.0, blend_frames, dtype=np.float32)
                        ramp_stereo = np.repeat(ramp, CHANNELS)
                        chunk = chunk.copy()
                        chunk[:blend_len] = (
                            tail_for_blend[:blend_len] * (1.0 - ramp_stereo)
                            + chunk[:blend_len] * ramp_stereo
                        )
                        self._crossfade_remaining -= blend_frames

                # Update tail ring
                if not switched:
                    samples_len = len(chunk)
                    tail_buf = self._tail_buffer
                    tail_len = len(tail_buf)
                    if samples_len >= tail_len:
                        tail_buf[:] = chunk[-tail_len:]
                        self._tail_write = tail_len
                    else:
                        w = self._tail_write
                        remaining = tail_len - w
                        if samples_len <= remaining:
                            tail_buf[w : w + samples_len] = chunk
                            self._tail_write = w + samples_len
                        else:
                            tail_buf[w:] = chunk[:remaining]
                            overflow = samples_len - remaining
                            tail_buf[:overflow] = chunk[remaining:]
                            self._tail_write = overflow

                actual_frames = len(chunk) // CHANNELS
                self._position_sec = self._sample_pos / (TARGET_SAMPLE_RATE * CHANNELS)

                with self._fft_lock:
                    n = min(len(chunk), len(self._fft_buffer))
                    self._fft_buffer[:n] = chunk[:n]
                    if n < len(self._fft_buffer):
                        self._fft_buffer[n:] = 0

                # DSP
                out = chunk.astype(np.float32, copy=True)
                if self._replaygain_db != 0.0:
                    linear = 10.0 ** (self._replaygain_db / 20.0)
                    np.multiply(out, linear, out=out)

                out = self._eq.process(out, CHANNELS)

                vol = 0.0 if self._muted else self._volume
                if vol != 1.0:
                    np.multiply(out, vol, out=out)

                out = soft_clip_float32(out)
                current_output = out.tobytes()
