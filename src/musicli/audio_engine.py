"""Audio playback engine: native-rate decoding, DSP pipeline, FFT data."""

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

DEFAULT_SAMPLE_RATE = 44100  # fallback
CHANNELS = 2
BYTES_PER_SAMPLE = 4  # float32
BUFFER_FRAMES = 1024
CROSSFADE_FRAMES = 2000  # ~45ms at 44.1k, ~21ms at 96k
FADE_SAMPLES = 256  # anti-pop fade on device stop


# ── Engine ──────────────────────────────────────────────────────────────────


class AudioEngine:
    """Manages miniaudio playback with native sample-rate support.

    Detects each file's original sample rate and re-initialises the
    hardware device to match — no resampling, no aliasing, bit-perfect
    path with optional pure-bypass mode.
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
        self._fs: int = DEFAULT_SAMPLE_RATE  # current hardware rate
        self._eq: EQPipeline = EQPipeline.create(fs=self._fs)
        self._replaygain_db: float = 0.0

        # ── Bit-perfect mode (bypasses all DSP) ───────────────────────
        self._bitperfect: bool = False

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

        # ── Decoded sample cursor ──────────────────────────────────────
        self._sample_pos: int = 0
        self._samples: Optional[np.ndarray] = None
        self._samplerate: int = DEFAULT_SAMPLE_RATE  # native rate of current track

        # ── Next-track native info ─────────────────────────────────────
        self._next_samples: Optional[np.ndarray] = None
        self._next_samplerate: int = DEFAULT_SAMPLE_RATE

        # ── Anti-pop fade state ───────────────────────────────────────
        self._fade_samples_left: int = 0

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
        with self._lock:
            return self._position_sec

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def sample_rate(self) -> int:
        """Current hardware sample rate (native to the playing track)."""
        return self._fs

    @property
    def bitperfect(self) -> bool:
        return self._bitperfect

    # ── FFT data access ──────────────────────────────────────────────────

    def get_fft_data(self) -> np.ndarray:
        with self._fft_lock:
            frame = self._fft_buffer.copy()
        mono = (frame[0::2] + frame[1::2]) * 0.5
        return mono

    def get_fft_stereo(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._fft_lock:
            frame = self._fft_buffer.copy()
        return frame[0::2].copy(), frame[1::2].copy()

    # ── Track loading ────────────────────────────────────────────────────

    def _decode_file_native_rate(self, path: str) -> int:
        """Peek at a file's native sample rate without full decoding."""
        try:
            info = miniaudio.get_file_info(path)
            return info.sample_rate
        except Exception:
            logger.debug("Could not read file info for %s, using default", path)
            return DEFAULT_SAMPLE_RATE

    def _decode_file(self, path: str, native_rate: int) -> Tuple[np.ndarray, float, int]:
        """Decode audio to interleaved float32 stereo at the given native rate.

        Returns (samples, duration_seconds, actual_sample_rate).
        """
        decoded = miniaudio.decode_file(
            path,
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=CHANNELS,
            sample_rate=native_rate,
        )
        samples = np.frombuffer(decoded.samples, dtype=np.float32).copy()
        if len(samples) % CHANNELS:
            samples = samples[: len(samples) - (len(samples) % CHANNELS)]
        duration = len(samples) / (native_rate * CHANNELS)
        return samples, duration, native_rate

    def load_track(self, path: str) -> bool:
        """Load an audio file for playback. Detects native rate and
        re-inits the hardware device to match. Returns True on success."""
        try:
            native_rate = self._decode_file_native_rate(path)
            samples, duration, actual_rate = self._decode_file(path, native_rate)
        except Exception as exc:
            logger.error("Failed to decode %s: %s", path, exc)
            return False

        # Restart device at native rate (outside lock to keep speed)
        if self._device:
            self._stop_and_fade_device()

        self._fs = actual_rate
        self._eq = EQPipeline.create(fs=self._fs)

        with self._lock:
            self._samples = samples
            self._sample_pos = 0
            self._samplerate = actual_rate
            self._current_path = path
            self._duration = duration
            self._position_sec = 0.0
            self._next_samples = None
            self._next_samplerate = DEFAULT_SAMPLE_RATE
            self._next_path = ""
            self._next_replaygain_db = 0.0
            self._next_duration = 0.0
            self._crossfade_remaining = 0
            self._tail_write = 0
            self._current_decoder = None
            self._next_decoder = None

        logger.info(
            "Loaded: %s (%.1fs, %d Hz native, %d ch)",
            Path(path).name,
            duration,
            actual_rate,
            CHANNELS,
        )
        return True

    def preload_next(self, path: str, replaygain_db: float = 0.0) -> bool:
        """Pre-decode the next track for gapless playback. If the next track
        has a different native rate, gapless won't happen (device restart needed)."""
        try:
            native_rate = self._decode_file_native_rate(path)
            samples, duration, actual_rate = self._decode_file(path, native_rate)
        except Exception as exc:
            logger.debug("Failed to preload %s: %s", path, exc)
            return False

        with self._lock:
            self._next_samples = samples
            self._next_samplerate = actual_rate
            self._next_path = path
            self._next_replaygain_db = replaygain_db
            self._next_duration = duration

        logger.debug("Preloaded next: %s (%d Hz)", Path(path).name, actual_rate)
        return True

    def has_just_switched(self) -> Tuple[bool, str]:
        with self._lock:
            switched = self._just_switched
            path = self._switched_to_path
            self._just_switched = False
            self._switched_to_path = ""
        return switched, path

    # ── Playback control ─────────────────────────────────────────────────

    def play(self) -> None:
        if self._device is None or not self._device.running:
            self._start_device(self._fs)
        self._playing = True
        self._paused = False

    def pause(self) -> None:
        self._paused = not self._paused
        logger.debug("Paused: %s", self._paused)

    def resume(self) -> None:
        if self._samples is None:
            return
        if self._device is None or not self._device.running:
            self._start_device(self._fs)
        self._playing = True
        self._paused = False

    def stop(self) -> None:
        self._playing = False
        self._paused = False
        with self._lock:
            self._next_samples = None
            self._next_path = ""
            self._next_duration = 0.0
            self._crossfade_remaining = 0
            self._tail_write = 0
            self._position_sec = 0.0
        self._stop_and_fade_device()

    def _stop_and_fade_device(self) -> None:
        """Stop device with a tiny fade to prevent DC pops."""
        if self._device and self._device.running:
            self._fade_samples_left = FADE_SAMPLES
            # Let a few buffer cycles pass for the fade to render
            import time
            time.sleep(FADE_SAMPLES / max(self._fs, 1) * 2)
            try:
                self._device.stop()
                self._device.close()
            except Exception:
                pass
            self._device = None

    def seek(self, seconds: float) -> None:
        with self._lock:
            if self._samples is None:
                return
            total_frames = len(self._samples) // CHANNELS
            frame = int(max(0.0, min(seconds, self._duration)) * self._samplerate)
            frame = min(frame, max(0, total_frames - 1))
            self._sample_pos = frame * CHANNELS
            self._position_sec = frame / self._samplerate
            n_sections = self._eq.sos.shape[0]
            self._eq.zi_per_channel = [
                np.zeros((n_sections, 2)) for _ in range(2)
            ]

    def set_volume(self, vol: float) -> None:
        self._volume = max(0.0, min(1.0, vol))

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def set_eq(self, bass_db: float, mid_db: float, treble_db: float) -> None:
        new_eq = EQPipeline.create(
            bass_db=bass_db,
            mid_db=mid_db,
            treble_db=treble_db,
            fs=self._fs,
        )
        with self._lock:
            self._eq = new_eq

    def set_replaygain(self, gain_db: float) -> None:
        self._replaygain_db = gain_db

    def set_crossfade(self, enabled: bool) -> None:
        self._crossfade_enabled = enabled

    def set_bitperfect(self, enabled: bool) -> None:
        """Enable/disable bit-perfect bypass mode.

        When enabled: EQ, ReplayGain, and soft-clip are all bypassed.
        Raw decoded samples go straight to the DAC."""
        self._bitperfect = enabled
        logger.info("Bit-perfect mode: %s", "on" if enabled else "off")

    def compute_rms_normalization(self) -> None:
        if self._replaygain_db != 0.0:
            return
        with self._lock:
            samples = self._samples
        if samples is None or len(samples) == 0:
            return
        try:
            max_samples = int(5.0 * self._samplerate * CHANNELS)
            chunk = samples[: min(max_samples, len(samples))]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            if rms > 1e-9:
                target_rms = 10.0 ** (-18.0 / 20.0)
                gain_db = 20.0 * np.log10(target_rms / rms)
                gain_db = max(-20.0, min(20.0, gain_db))
                self._replaygain_db = gain_db
                logger.info("RMS normalization: %.1f dB", gain_db)
        except Exception as exc:
            logger.debug("RMS normalization failed: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────

    def _start_device(self, rate: int) -> None:
        """Create and start miniaudio device at the given sample rate."""
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
            sample_rate=rate,
        )
        gen = self._callback_generator()
        next(gen)
        self._device.start(gen)
        logger.debug("Device started at %d Hz", rate)

    def _read_frames(self, frame_count: int) -> Tuple[Optional[np.ndarray], bool]:
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
        self._samplerate = self._next_samplerate
        self._current_path = self._next_path
        self._duration = self._next_duration
        self._position_sec = 0.0
        self._replaygain_db = self._next_replaygain_db
        self._just_switched = True
        self._switched_to_path = self._next_path

        self._next_samples = None
        self._next_samplerate = DEFAULT_SAMPLE_RATE
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

        Supports native-rate switching, bit-perfect bypass, and anti-pop fade."""
        current_output = b"\x00" * (BUFFER_FRAMES * CHANNELS * BYTES_PER_SAMPLE)

        while True:
            frame_count = yield current_output

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
                self._position_sec = self._sample_pos / (self._samplerate * CHANNELS)

                with self._fft_lock:
                    n = min(len(chunk), len(self._fft_buffer))
                    self._fft_buffer[:n] = chunk[:n]
                    if n < len(self._fft_buffer):
                        self._fft_buffer[n:] = 0

                # ── Anti-pop fade ────────────────────────────────────────
                if self._fade_samples_left > 0:
                    fade = min(actual_frames, self._fade_samples_left)
                    fade_len = fade * CHANNELS
                    ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
                    ramp_stereo = np.repeat(ramp, CHANNELS)
                    chunk = chunk.copy()
                    chunk[:fade_len] *= ramp_stereo
                    self._fade_samples_left -= fade
                    if self._fade_samples_left <= 0:
                        self._fade_samples_left = 0

                # ── DSP Pipeline (or bit-perfect bypass) ──────────────────
                if self._bitperfect:
                    out = chunk.astype(np.float32, copy=True)
                    vol = 0.0 if self._muted else self._volume
                    if vol != 1.0:
                        np.multiply(out, vol, out=out)
                    out = np.clip(out, -1.0, 1.0).astype(np.float32)
                else:
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