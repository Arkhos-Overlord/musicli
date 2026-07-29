#!/usr/bin/env python
"""musicli — premium retro-CRT CLI music player.

Usage:
    python -m musicli.main              # Launch the TUI player
    python -m musicli.main --scan       # Force re-scan audio files
    python -m musicli.main --version    # Show version
"""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from musicli.audio_engine import AudioEngine
from musicli.dsp import parse_replaygain_db
from musicli.lyrics import fetch_lyrics_async
from musicli.scanner import scan_directory
from musicli.state import PlayerState, TrackMeta, load_cache, load_state, save_state
from musicli.themes import THEMES, WAVEFORM_STYLES
from musicli.ui import MusicliUI, read_key
from musicli.waveform import precompute_waveform_async

logger = logging.getLogger("musicli")

# ── Constants ───────────────────────────────────────────────────────────────

FPS = 15
FRAME_TIME = 1.0 / FPS

# EQ adjustment step
EQ_STEP = 2.0  # dB

# Seek step
SEEK_SEC = 10.0

# ── Key code mappings ───────────────────────────────────────────────────────

# Keys that return multi-char on Windows get mapped in ui.py
ARROW_UP = ("UP", "\x1b[A", "k")
ARROW_DOWN = ("DOWN", "\x1b[B", "j")
ARROW_LEFT = ("LEFT", "\x1b[D")
ARROW_RIGHT = ("RIGHT", "\x1b[C")


# ── Main App ────────────────────────────────────────────────────────────────


class MusicliApp:
    """Orchestrator: wires together scanner, audio engine, lyrics, and UI."""

    def __init__(self, force_scan: bool = False) -> None:
        self.force_scan = force_scan

        # ── State ──────────────────────────────────────────────────────
        self.state: PlayerState = load_state()
        self.tracks: list[TrackMeta] = []
        self.current_idx: int = 0
        self.current_track: Optional[TrackMeta] = None

        # ── Queue (indices into self.tracks) ───────────────────────────
        self.queue: list[int] = []

        # ── Lyrics ─────────────────────────────────────────────────────
        self.lyrics: Dict[float, str] = {}

        # ── Waveform ───────────────────────────────────────────────────
        self.waveform_peaks: Optional[np.ndarray] = None

        # ── Search ─────────────────────────────────────────────────────
        self.filter_text: str = ""

        # ── Modules ────────────────────────────────────────────────────
        self.engine = AudioEngine()
        self.ui = MusicliUI(app_name="musicli")

        # ── Restore theme / waveform ────────────────────────────────────
        if 0 <= self.state.theme_index < len(THEMES):
            self.ui.theme = THEMES[self.state.theme_index]
        if 0 <= self.state.waveform_index < len(WAVEFORM_STYLES):
            self.ui.waveform_style = WAVEFORM_STYLES[self.state.waveform_index]

        # ── Restore engine state ───────────────────────────────────────
        self.engine.set_volume(self.state.volume)
        self.engine.set_muted(self.state.muted)
        self.engine.set_crossfade(self.state.crossfade)
        self.engine.set_eq(
            self.state.eq_bass,
            self.state.eq_mid,
            self.state.eq_treble,
        )

    # ── Initialisation ───────────────────────────────────────────────────

    def init(self) -> None:
        """Scan library and load last track if available."""
        # Scan
        raw = scan_directory(Path.cwd(), force=self.force_scan)
        self.tracks = raw

        if not self.tracks:
            logger.warning("No audio files found in current directory.")

        # ── Restore queue from saved paths ────────────────────────────
        path_to_idx = {t.path: i for i, t in enumerate(self.tracks)}
        for saved_path in self.state.queue_paths:
            idx = path_to_idx.get(saved_path)
            if idx is not None:
                self.queue.append(idx)

        # Find last-played track
        if self.state.current_track_path:
            for i, t in enumerate(self.tracks):
                if t.path == self.state.current_track_path:
                    self.current_idx = i
                    self._load_track(i)
                    break

    # ── Track operations ─────────────────────────────────────────────────

    def _load_track(self, idx: int) -> None:
        """Load a track by library index."""
        if idx < 0 or idx >= len(self.tracks):
            return

        track = self.tracks[idx]
        self.current_idx = idx
        self.current_track = track

        # ── Extract cover art ─────────────────────────────────────────
        self._extract_cover_art(track.path)

        # Load into engine
        ok = self.engine.load_track(track.path)
        if not ok:
            return

        # ReplayGain — use tag if available, else compute RMS fallback
        rg = parse_replaygain_db(track.replaygain_track)
        if rg != 0.0 or track.replaygain_track is not None:
            self.engine.set_replaygain(rg)
        else:
            self.engine.compute_rms_normalization()

        # Seek to saved position if resuming
        if track.path == self.state.current_track_path:
            self.engine.seek(self.state.position_sec)
            # Reset saved position so it only applies once
            self.state.position_sec = 0.0

        # Fetch lyrics in background
        self.lyrics = {}
        fetch_lyrics_async(
            track.artist,
            track.title,
            callback=lambda lrc: self._on_lyrics(lrc),
        )

        # Pre-compute waveform in background
        self.waveform_peaks = None
        precompute_waveform_async(
            track.path,
            callback=lambda peaks: self._on_waveform(peaks),
        )

        # ── Preload next track for gapless playback ──────────────────
        next_idx = self._compute_next_index()
        if next_idx is not None:
            next_track = self.tracks[next_idx]
            next_rg = parse_replaygain_db(next_track.replaygain_track)
            self.engine.preload_next(next_track.path, next_rg)

        self.engine.play()

    def _on_lyrics(self, lrc: Dict[float, str]) -> None:
        self.lyrics = lrc

    def _on_waveform(self, peaks: np.ndarray) -> None:
        if len(peaks) > 0:
            self.waveform_peaks = peaks

    def _extract_cover_art(self, path: str) -> None:
        """Extract album art from audio file and push to UI."""
        try:
            from mutagen import File as MFile
            mf = MFile(path)
            if mf is None:
                self.ui.set_cover_art(None)
                return
            art = None
            # MP3/ID3: APIC keys
            for key in mf.keys():
                if key.startswith("APIC:"):
                    art = mf[key].data
                    break
            # MP4/M4A: covr key
            if art is None and "covr" in mf:
                covr = mf["covr"]
                if covr:
                    art = covr[0] if hasattr(covr[0], 'data') else bytes(covr[0])
            if art:
                self.ui.set_cover_art(bytes(art) if not isinstance(art, bytes) else art)
            else:
                self.ui.set_cover_art(None)
        except Exception:
            self.ui.set_cover_art(None)

    def _compute_next_index(self) -> Optional[int]:
        """Determine the next track index. Queue takes priority over
        natural sequence."""
        n = len(self.tracks)
        if n == 0:
            return None

        # Queue first
        if self.queue:
            return self.queue[0]

        # Repeat: single track
        if self.state.repeat_mode == "track":
            return self.current_idx

        # Shuffle
        if self.state.shuffle:
            return random.randrange(n)

        # Next in natural sequence (wrap around)
        return (self.current_idx + 1) % n

    def _play_index(self, idx: int) -> None:
        """Play track at library index, wrapping around."""
        n = len(self.tracks)
        if n == 0:
            return
        idx = idx % n
        self._load_track(idx)

    # ── Handle input ────────────────────────────────────────────────────

    def _handle_key(self, key: str) -> bool:
        """Process a keypress. Returns False to quit."""
        # ── Quit ────────────────────────────────────────────────────────
        if key in ("q", "Q", "\x1b"):  # q or Esc
            if self.filter_text:
                self.filter_text = ""
                return True
            return False

        # ── Search mode ────────────────────────────────────────────────
        if key == "/":
            self.filter_text = ""
            return True

        # If in search, accumulate filter text
        if key == "\x08" or key == "\x7f":  # Backspace
            self.filter_text = self.filter_text[:-1]
            return True

        if len(key) == 1 and key.isprintable() and key not in " \r\n\t":
            self.filter_text += key
            return True

        # ── Queue ────────────────────────────────────────────────────
        if key in ("a", "A"):  # Add highlighted track to queue
            if self.current_idx not in self.queue and self.tracks:
                self.queue.append(self.current_idx)
            return True

        if key in ("d", "D"):  # Remove highlighted track from queue
            if self.current_idx in self.queue:
                self.queue.remove(self.current_idx)
            return True

        if key == "[":  # Move highlighted queue item earlier
            if self.current_idx in self.queue:
                pos = self.queue.index(self.current_idx)
                if pos > 0:
                    self.queue[pos], self.queue[pos - 1] = (
                        self.queue[pos - 1], self.queue[pos]
                    )
            return True

        if key == "]":  # Move highlighted queue item later
            if self.current_idx in self.queue:
                pos = self.queue.index(self.current_idx)
                if pos < len(self.queue) - 1:
                    self.queue[pos], self.queue[pos + 1] = (
                        self.queue[pos + 1], self.queue[pos]
                    )
            return True

        if key in ("x", "X"):  # Clear entire queue
            self.queue.clear()
            return True

        # ── Playback ────────────────────────────────────────────────────
        if key == " ":  # Space = Play/Pause
            if self.engine.is_playing:
                self.engine.pause()
            else:
                self._load_track(self.current_idx)
            return True

        if key in ("\r", "\n"):  # Enter = play selected track
            self._load_track(self.current_idx)
            return True

        if key in ARROW_UP:  # Up
            self.current_idx = max(0, self.current_idx - 1)
            return True

        if key in ARROW_DOWN:  # Down
            self.current_idx = min(len(self.tracks) - 1, self.current_idx + 1)
            return True

        if key in ("n", "N"):
            self._play_index(self.current_idx + 1)
            return True

        if key in ("p", "P"):
            self._play_index(self.current_idx - 1)
            return True

        if key in ARROW_LEFT:  # Seek -10s
            pos = self.engine.position - SEEK_SEC
            self.engine.seek(max(0, pos))
            return True

        if key in ARROW_RIGHT:  # Seek +10s
            pos = self.engine.position + SEEK_SEC
            dur = self.engine.duration
            self.engine.seek(min(dur, pos))
            return True

        # ── Volume ──────────────────────────────────────────────────────
        if key in ("+", "="):
            self.engine.set_volume(self.engine.volume + 0.05)
            self.state.volume = self.engine.volume
            if self.engine.volume > 0.01:
                self.state.muted = False
            return True

        if key in ("-", "_"):
            self.engine.set_volume(self.engine.volume - 0.05)
            self.state.volume = self.engine.volume
            return True

        if key in ("m", "M"):
            self.state.muted = not self.state.muted
            self.engine.set_muted(self.state.muted)
            return True

        # ── EQ ──────────────────────────────────────────────────────────
        if key == "1":  # Bass boost
            self.state.eq_bass = min(12.0, self.state.eq_bass + EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "!":
            self.state.eq_bass = max(-12.0, self.state.eq_bass - EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True

        if key == "2":  # Mid
            self.state.eq_mid = min(12.0, self.state.eq_mid + EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "@":
            self.state.eq_mid = max(-12.0, self.state.eq_mid - EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True

        if key == "3":  # Treble
            self.state.eq_treble = min(12.0, self.state.eq_treble + EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "#":
            self.state.eq_treble = max(-12.0, self.state.eq_treble - EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True

        # ── Help overlay ───────────────────────────────────────────────
        if key == "?":
            self.ui.show_help = not self.ui.show_help
            return True

        # ── Repeat / Shuffle ──────────────────────────────────────────────
        if key in ("r", "R"):
            modes = ["off", "track", "album"]
            idx = modes.index(self.state.repeat_mode) if self.state.repeat_mode in modes else 0
            self.state.repeat_mode = modes[(idx + 1) % len(modes)]
            return True

        if key in ("s", "S"):
            self.state.shuffle = not self.state.shuffle
            return True

        # ── Crossfade ──────────────────────────────────────────────────
        if key in ("c", "C"):
            self.state.crossfade = not self.state.crossfade
            self.engine.set_crossfade(self.state.crossfade)
            return True

        # ── Theme / Waveform ────────────────────────────────────────────
        if key in ("t", "T"):
            self.ui.cycle_theme()
            self.state.theme_index = THEMES.index(self.ui.theme)
            return True

        if key in ("w", "W"):
            self.ui.cycle_waveform()
            self.state.waveform_index = WAVEFORM_STYLES.index(self.ui.waveform_style)
            return True

        return True

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Launch the full-screen TUI and run the event loop."""
        self.init()

        with self.ui:
            self.ui.start_live()

            try:
                while True:
                    frame_start = time.monotonic()

                    # ── Process input ──────────────────────────────────
                    key = read_key()
                    while key is not None:
                        if not self._handle_key(key):
                            # Quit
                            self._save_and_quit()
                            return
                        key = read_key()

                    # ── Auto-advance track on end ──────────────────────
                    if (
                        self.engine.is_playing
                        and not self.engine.is_paused
                        and self.engine.position >= self.engine.duration - 0.1
                        and self.engine.duration > 0
                    ):
                        self._handle_track_end()

                    # ── Gapless auto-switch detection ───────────────────
                    switched, new_path = self.engine.has_just_switched()
                    if switched and new_path:
                        # Find the track index for the auto-switched path
                        for i, t in enumerate(self.tracks):
                            if t.path == new_path:
                                self.current_idx = i
                                self.current_track = t
                                self.lyrics = {}
                                self.waveform_peaks = None
                                fetch_lyrics_async(
                                    t.artist, t.title,
                                    callback=lambda lrc: self._on_lyrics(lrc),
                                )
                                precompute_waveform_async(
                                    t.path,
                                    callback=lambda peaks: self._on_waveform(peaks),
                                )
                                # If this was a queue item, remove it
                                if i in self.queue:
                                    self.queue.remove(i)
                                # Preload next
                                next_idx = self._compute_next_index()
                                if next_idx is not None:
                                    nt = self.tracks[next_idx]
                                    nrg = parse_replaygain_db(nt.replaygain_track)
                                    self.engine.preload_next(nt.path, nrg)
                                break

                    # ── Gather state for rendering ─────────────────────
                    position = self.engine.position
                    duration = self.engine.duration
                    fft_data = self.engine.get_fft_data() if self.engine.is_playing else None
                    fft_stereo = self.engine.get_fft_stereo() if self.engine.is_playing else None

                    # ── Update UI ──────────────────────────────────────
                    self.ui.update(
                        tracks=self.tracks,
                        current_idx=self.current_idx,
                        current_track=self.current_track,
                        position_sec=position,
                        duration=duration,
                        volume=self.engine.volume,
                        muted=self.state.muted,
                        eq_bass=self.state.eq_bass,
                        eq_mid=self.state.eq_mid,
                        eq_treble=self.state.eq_treble,
                        lyrics=self.lyrics,
                        filter_text=self.filter_text,
                        fft_data=fft_data,
                        waveform_peaks=self.waveform_peaks,
                        is_playing=self.engine.is_playing and not self.engine.is_paused,
                        repeat_mode=self.state.repeat_mode,
                        shuffle=self.state.shuffle,
                        queue_indices=self.queue,
                        crossfade=self.state.crossfade,
                        fft_stereo=fft_stereo,
                    )

                    # ── Pacing ─────────────────────────────────────────
                    elapsed = time.monotonic() - frame_start
                    sleep_time = max(0, FRAME_TIME - elapsed)
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                pass
            finally:
                self._save_and_quit()
                self.ui.stop_live()

    def _handle_track_end(self) -> None:
        """Called when the current track finishes playing."""
        # Pop queue first if there are items
        if self.queue:
            next_idx = self.queue.pop(0)
            self._load_track(next_idx)
            return

        if self.state.repeat_mode == "track":
            self._load_track(self.current_idx)
        elif self.state.shuffle:
            idx = random.randrange(len(self.tracks))
            self._play_index(idx)
        else:
            self._play_index(self.current_idx + 1)

    def _save_and_quit(self) -> None:
        """Persist state and stop engine."""
        self.state.volume = self.engine.volume
        self.state.position_sec = self.engine.position
        if self.current_track:
            self.state.current_track_path = self.current_track.path
        self.state.theme_index = THEMES.index(self.ui.theme)
        self.state.waveform_index = WAVEFORM_STYLES.index(self.ui.waveform_style)
        # Save queue as paths (indices shift between sessions)
        self.state.queue_paths = [
            self.tracks[i].path for i in self.queue if 0 <= i < len(self.tracks)
        ]

        save_state(self.state)
        self.engine.stop()
        logger.info("State saved. Goodbye!")


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry: parse args, launch TUI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="musicli — premium retro-CRT CLI music player"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Force re-scan of audio files (ignore cache)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    args = parser.parse_args()

    if args.version:
        print(f"musicli v0.2.0")
        return

    # ── Setup logging (non-intrusive) ───────────────────────────────────
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler("musicli.log", encoding="utf-8")],
    )

    app = MusicliApp(force_scan=args.scan)
    app.run()


if __name__ == "__main__":
    main()
