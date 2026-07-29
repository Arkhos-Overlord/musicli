"""Audio playback engine: miniaudio decoder, DSP pipeline, FFT data."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import miniaudio
import numpy as np

from musicli.dsp import EQPipeline, apply_tpdf_dither

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

TARGET_SAMPLE_RATE = 96000
CHANNELS = 2
BUFFER_FRAMES = 1024  # frames per callback invocation
CROSSFADE_FRAMES = 2000  # ~21ms crossfade duration at 96kHz


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
        self._current_decoder: Optional[miniaudio.Decoder] = None
        self._next_decoder: Optional[miniaudio.Decoder] = None
        self._next_path: str = ""
        self._next_replaygain_db: float = 0.0
        self._next_duration: float = 0.0
        self._lock = threading.Lock()

        # ── Playback state ──────────────────────────────────────────────
        self._playing = False
        self._paused = False
        self._volume: float = 0.8  # linear, 0.0–1.0
        self._muted: bool = False

        # ── DSP state ───────────────────────────────────────────────────
        self._eq: EQPipeline = EQPipeline.create(fs=TARGET_SAMPLE_RATE)
        self._replaygain_db: float = 0.0

        # ── Crossfade state ───────────────────────────────────────────
        self._crossfade_enabled: bool = True
        self._crossfade_remaining: int = 0
        # Ring buffer holding the last CROSSFADE_FRAMES samples (interleaved)
        self._tail_buffer = np.zeros(CROSSFADE_FRAMES * CHANNELS, dtype=np.float32)
        self._tail_write: int = 0  # current write position in ring buffer

        # ── Track info ──────────────────────────────────────────────────
        self._current_path: str = ""
        self._duration: float = 0.0
        self._position_callback_frame: float = 0.0  # updated in callback

        # ── Auto-switch notification (gapless) ─────────────────────────
        self._just_switched: bool = False
        self._switched_to_path: str = ""

        # ── FFT / visualiser data ──────────────────────────────────────
        # Shared buffer: last frame of float32 audio for the UI to FFT
        self._fft_buffer = np.zeros(BUFFER_FRAMES * CHANNELS, dtype=np.float32)
        self._fft_lock = threading.Lock()

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
        """Current playback position in seconds (from callback)."""
        with self._lock:
            return self._position_callback_frame

    @property
    def volume(self) -> float:
        return self._volume

    # ── FFT data access ──────────────────────────────────────────────────

    def get_fft_data(self) -> np.ndarray:
        """Return a copy of the last audio frame for FFT visualisation.

        Returns a mono float32 array of `BUFFER_FRAMES` samples (averaged
        L+R), suitable for `np.fft.rfft`.
        """
        with self._fft_lock:
            frame = self._fft_buffer.copy()
        # Convert stereo interleaved → mono
        mono = (frame[0::2] + frame[1::2]) * 0.5
        return mono

    def get_fft_stereo(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return left and right channels separately for stereo FFT."""
        with self._fft_lock:
            frame = self._fft_buffer.copy()
        left = frame[0::2].copy()
        right = frame[1::2].copy()
        return left, right

    # ── Track loading ────────────────────────────────────────────────────

    def load_track(self, path: str) -> bool:
        """Load an audio file for playback. Returns True on success."""
        try:
            decoder = miniaudio.decode_file(
                path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=CHANNELS,
                sample_rate=TARGET_SAMPLE_RATE,
            )
        except Exception as exc:
            logger.error("Failed to decode %s: %s", path, exc)
            return False

        with self._lock:
            self._current_decoder = decoder
            self._current_path = path
            self._duration = decoder.duration
            self._position_callback_frame = 0.0
            # Clear stale preloaded next track and crossfade state
            self._next_decoder = None
            self._next_path = ""
            self._next_replaygain_db = 0.0
            self._crossfade_remaining = 0
            self._tail_write = 0

        logger.info(
            "Loaded: %s (%.1fs, %d Hz, %d ch)",
            Path(path).name,
            decoder.duration,
            decoder.sample_rate,
            decoder.nchannels,
        )
        return True

    def preload_next(self, path: str, replaygain_db: float = 0.0) -> bool:
        """Pre-decode the next track for gapless playback.

        Call this after `load_track()` with the path of the track that
        should play next. The callback will auto-switch when the current
        decoder runs dry.
        """
        try:
            decoder = miniaudio.decode_file(
                path,
                output_format=miniaudio.SampleFormat.FLOAT32,
                nchannels=CHANNELS,
                sample_rate=TARGET_SAMPLE_RATE,
            )
        except Exception as exc:
            logger.debug("Failed to preload %s: %s", path, exc)
            return False

        with self._lock:
            self._next_decoder = decoder
            self._next_path = path
            self._next_replaygain_db = replaygain_db
            self._next_duration = decoder.duration

        logger.debug("Preloaded next: %s", Path(path).name)
        return True

    def has_just_switched(self) -> Tuple[bool, str]:
        """Check if the callback auto-switched to a preloaded track.

        Returns (switched, path). Reading this clears the flag so each
        switch is only reported once.
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

    def stop(self) -> None:
        """Stop playback and release device."""
        self._playing = False
        self._paused = False
        with self._lock:
            self._next_decoder = None
            self._next_path = ""
            self._next_duration = 0.0
            self._crossfade_remaining = 0
            self._tail_write = 0
        if self._device and self._device.running:
            self._device.stop()
            self._device.close()
            self._device = None

    def seek(self, seconds: float) -> None:
        """Seek to an absolute position in seconds."""
        with self._lock:
            if self._current_decoder:
                try:
                    self._current_decoder.seek(int(seconds * TARGET_SAMPLE_RATE))
                    self._position_callback_frame = seconds
                    # Reset EQ filter state after seek to avoid transients
                    n_sections = self._eq.sos.shape[0]
                    self._eq.zi_per_channel = [
                        np.zeros((n_sections, 2)) for _ in range(2)
                    ]
                except Exception as exc:
                    logger.warning("Seek failed: %s", exc)

    def set_volume(self, vol: float) -> None:
        """Set volume 0.0–1.0 (linear)."""
        self._volume = max(0.0, min(1.0, vol))

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def set_eq(self, bass_db: float, mid_db: float, treble_db: float) -> None:
        """Update EQ settings. Pre-computes new SOS coefficients outside
        the audio callback so the hot path stays fast."""
        self._eq = EQPipeline.create(
            bass_db=bass_db,
            mid_db=mid_db,
            treble_db=treble_db,
            fs=TARGET_SAMPLE_RATE,
        )

    def set_replaygain(self, gain_db: float) -> None:
        """Set the ReplayGain value for the current track."""
        self._replaygain_db = gain_db

    def set_crossfade(self, enabled: bool) -> None:
        """Enable or disable crossfade transitions."""
        self._crossfade_enabled = enabled

    def compute_rms_normalization(self) -> None:
        """Compute on-the-fly RMS normalization gain when no ReplayGain
        tags are available. Samples the first 5 seconds of the already-
        loaded decoder for a fast RMS estimate."""
        if self._replaygain_db != 0.0:
            return  # Already have ReplayGain, skip
        with self._lock:
            decoder = self._current_decoder
        if decoder is None:
            return
        try:
            # Read first 5 seconds (or whole track if shorter) from decoder
            max_frames = int(5.0 * TARGET_SAMPLE_RATE)
            total_frames = int(decoder.duration * TARGET_SAMPLE_RATE)
            frames_to_read = min(max_frames, total_frames)

            # Save current position, read the chunk, then seek back
            saved_pos = self._position_callback_frame
            with self._lock:
                decoder.seek(0)
                raw = decoder.read(frames_to_read)
                decoder.seek(int(saved_pos * TARGET_SAMPLE_RATE))
                self._position_callback_frame = saved_pos

            if raw is None:
                return
            samples = np.frombuffer(raw, dtype=np.float32)
            if len(samples) == 0:
                return

            # Compute RMS and target -18 dBFS (a common loudness target)
            rms = np.sqrt(np.mean(samples ** 2))
            if rms > 0:
                target_rms = 10.0 ** (-18.0 / 20.0)  # -18 dBFS
                gain_db = 20.0 * np.log10(target_rms / rms)
                self._replaygain_db = gain_db
                logger.info("RMS normalization: %.1f dB", gain_db)
        except Exception as exc:
            logger.debug("RMS normalization failed: %s", exc)

    # ── Internal ──────────────────────────────────────────────────────────

    def _start_device(self) -> None:
        """Create and start the miniaudio playback device."""
        if self._device:
            if self._device.running:
                self._device.stop()
            self._device.close()

        self._device = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=CHANNELS,
            sample_rate=TARGET_SAMPLE_RATE,
        )
        gen = self._callback_generator()
        next(gen)  # prime the generator
        self._device.start(gen)

    def _callback_generator(self):
        """Generator-based audio callback for new miniaudio API.

        Uses the single-yield pattern: ``frame_count = yield output``.
        Each ``.send(frame_count)`` call receives the frame count AND
        yields the processed audio bytes in one step.
        """
        # Initial silence buffer to yield on the priming call
        output = b"\x00" * (BUFFER_FRAMES * CHANNELS * 2)
        while True:
            frame_count = yield output
            needed = frame_count * CHANNELS

            with self._lock:
                decoder = self._current_decoder
                paused = self._paused

            if decoder is None or paused:
                self._crossfade_remaining = 0
                self._tail_write = 0
                output = b"\x00" * (needed * 2)
                continue

            # ── Read decoded float32 samples ────────────────────────────
            try:
                with self._lock:
                    result = decoder.read(frame_count)
            except Exception:
                output = b"\x00" * (needed * 2)
                continue

            switched_this_call = False

            if result is None:
                # Track finished — try auto-switch to preloaded next
                with self._lock:
                    next_dec = self._next_decoder
                    next_path = self._next_path
                    next_rg = self._next_replaygain_db
                    next_duration = self._next_duration

                if next_dec is not None:
                    do_crossfade = self._crossfade_enabled and self._tail_write > 0
                    if do_crossfade:
                        cf_frames = min(CROSSFADE_FRAMES,
                                        self._tail_write // CHANNELS)
                        cf_len = cf_frames * CHANNELS
                        tail = np.zeros(cf_len, dtype=np.float32)
                        src_end = self._tail_write
                        src_start = max(0, src_end - cf_len)
                        tail[:cf_len] = self._tail_buffer[src_start:src_end]
                        self._crossfade_remaining = cf_frames
                    else:
                        self._crossfade_remaining = 0
                    self._tail_write = 0

                    # Gapless switch
                    with self._lock:
                        self._current_decoder = next_dec
                        self._current_path = next_path
                        self._next_decoder = None
                        self._next_path = ""
                        self._replaygain_db = next_rg
                        self._duration = next_duration
                        self._position_callback_frame = 0.0
                        self._just_switched = True
                        self._switched_to_path = next_path
                        n_sections = self._eq.sos.shape[0]
                        self._eq.zi_per_channel = [
                            np.zeros((n_sections, 2)) for _ in range(2)
                        ]
                        decoder = self._current_decoder

                    try:
                        with self._lock:
                            result = decoder.read(frame_count)
                    except Exception:
                        output = b"\x00" * (needed * 2)
                        continue

                    # ── Apply crossfade blend ───────────────────────────
                    if do_crossfade and self._crossfade_remaining > 0 and result is not None:
                        new_samples = np.frombuffer(result, dtype=np.float32)
                        actual_frames = len(new_samples) // CHANNELS
                        blend_frames = min(actual_frames, self._crossfade_remaining)
                        blend_len = blend_frames * CHANNELS
                        ramp = np.linspace(0.0, 1.0, blend_frames, dtype=np.float32)
                        ramp_stereo = np.repeat(ramp, CHANNELS)
                        result_bytes = bytearray(result)
                        new_view = np.frombuffer(result_bytes, dtype=np.float32)
                        if blend_len <= len(tail):
                            new_view[:blend_len] = (
                                tail[:blend_len] * (1.0 - ramp_stereo)
                                + new_view[:blend_len] * ramp_stereo
                            )
                        result = bytes(result_bytes)
                        self._crossfade_remaining -= blend_frames

                    switched_this_call = True
                else:
                    self._playing = False
                    self._crossfade_remaining = 0
                    output = b"\x00" * (needed * 2)
                    continue

            if result is None:
                self._playing = False
                self._crossfade_remaining = 0
                output = b"\x00" * (needed * 2)
                continue

            samples = np.frombuffer(result, dtype=np.float32).copy()

            # Pad with silence if we got fewer frames
            actual_frames = len(samples) // CHANNELS
            if actual_frames < frame_count:
                pad = np.zeros((frame_count - actual_frames) * CHANNELS, dtype=np.float32)
                samples = np.concatenate([samples, pad])
                if not switched_this_call:
                    with self._lock:
                        next_dec = self._next_decoder
                    if next_dec is None:
                        self._playing = False

            # ── Update tail buffer (ring buffer) ────────────────────────
            if not switched_this_call:
                samples_len = len(samples)
                tail_buf = self._tail_buffer
                tail_len = len(tail_buf)
                w = self._tail_write
                remaining = tail_len - w
                if samples_len <= remaining:
                    tail_buf[w:w + samples_len] = samples
                    self._tail_write = w + samples_len
                else:
                    tail_buf[w:] = samples[:remaining]
                    overflow = samples_len - remaining
                    tail_buf[:overflow] = samples[remaining:]
                    self._tail_write = overflow

            # ── Update position ─────────────────────────────────────────
            with self._lock:
                self._position_callback_frame += actual_frames / TARGET_SAMPLE_RATE

            # ── Copy to FFT buffer (thread-safe) ────────────────────────
            with self._fft_lock:
                self._fft_buffer[: len(samples)] = samples

            # ── DSP Pipeline (all numpy) ────────────────────────────────
            if self._replaygain_db != 0.0:
                linear = 10.0 ** (self._replaygain_db / 20.0)
                np.multiply(samples, linear, out=samples)

            samples = self._eq.process(samples, CHANNELS)

            vol = 0.0 if self._muted else self._volume
            if vol != 1.0:
                np.multiply(samples, vol, out=samples)

            output = apply_tpdf_dither(samples).tobytes()
