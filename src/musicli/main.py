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

from musicli import __version__
from musicli.audio_engine import AudioEngine
from musicli.dsp import parse_replaygain_db
from musicli.lyrics import fetch_lyrics_async
from musicli.scanner import scan_directory
from musicli.state import PlayerState, TrackMeta, load_state, save_state
from musicli.themes import THEMES, WAVEFORM_STYLES
from musicli.ui import MusicliUI, filter_tracks, read_key
from musicli.waveform import precompute_waveform_async

logger = logging.getLogger("musicli")

FPS = 15
FRAME_TIME = 1.0 / FPS
EQ_STEP = 2.0
SEEK_SEC = 10.0

ARROW_UP = ("UP", "\x1b[A", "k")
ARROW_DOWN = ("DOWN", "\x1b[B", "j")
ARROW_LEFT = ("LEFT", "\x1b[D")
ARROW_RIGHT = ("RIGHT", "\x1b[C")


class MusicliApp:
    """Orchestrator: wires together scanner, audio engine, lyrics, and UI."""

    def __init__(self, force_scan: bool = False) -> None:
        self.force_scan = force_scan

        self.state: PlayerState = load_state()
        self.tracks: list[TrackMeta] = []
        self.current_idx: int = 0
        self.current_track: Optional[TrackMeta] = None

        self.queue: list[int] = []
        self.lyrics: Dict[float, str] = {}
        self.waveform_peaks: Optional[np.ndarray] = None

        # Explicit search mode (activated by /)
        self.search_active: bool = False
        self.filter_text: str = ""

        # Generation counter to drop stale async callbacks
        self._track_generation: int = 0

        self.engine = AudioEngine()
        self.ui = MusicliUI(app_name="musicli")

        if 0 <= self.state.theme_index < len(THEMES):
            self.ui.theme = THEMES[self.state.theme_index]
        if 0 <= self.state.waveform_index < len(WAVEFORM_STYLES):
            self.ui.waveform_style = WAVEFORM_STYLES[self.state.waveform_index]

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
        raw = scan_directory(Path.cwd(), force=self.force_scan)
        self.tracks = raw

        if not self.tracks:
            logger.warning("No audio files found in current directory.")

        path_to_idx = {t.path: i for i, t in enumerate(self.tracks)}
        for saved_path in self.state.queue_paths:
            idx = path_to_idx.get(saved_path)
            if idx is not None:
                self.queue.append(idx)

        if self.state.current_track_path:
            for i, t in enumerate(self.tracks):
                if t.path == self.state.current_track_path:
                    self.current_idx = i
                    self._load_track(i, autoplay=False)
                    # Restore position without auto-playing
                    if self.state.position_sec > 0:
                        self.engine.seek(self.state.position_sec)
                    break

    # ── Track operations ─────────────────────────────────────────────────

    def _load_track(self, idx: int, autoplay: bool = True) -> None:
        """Load a track by library index."""
        if not self.tracks or idx < 0 or idx >= len(self.tracks):
            return

        track = self.tracks[idx]
        self.current_idx = idx
        self.current_track = track
        self._track_generation += 1
        gen = self._track_generation

        self._extract_cover_art(track.path)

        ok = self.engine.load_track(track.path)
        if not ok:
            return

        rg = parse_replaygain_db(track.replaygain_track)
        if rg != 0.0 or track.replaygain_track is not None:
            self.engine.set_replaygain(rg)
        else:
            self.engine.set_replaygain(0.0)
            self.engine.compute_rms_normalization()

        # Resume saved position only once on session restore
        if (
            track.path == self.state.current_track_path
            and self.state.position_sec > 0
            and not autoplay
        ):
            self.engine.seek(self.state.position_sec)
            self.state.position_sec = 0.0

        self.lyrics = {}
        fetch_lyrics_async(
            track.artist,
            track.title,
            callback=lambda lrc: self._on_lyrics(lrc),
            generation=gen,
            generation_check=lambda: self._track_generation,
        )

        self.waveform_peaks = None
        precompute_waveform_async(
            track.path,
            callback=lambda peaks: self._on_waveform(peaks),
            generation=gen,
            generation_check=lambda: self._track_generation,
        )

        next_idx = self._compute_next_index()
        if next_idx is not None and next_idx != idx:
            next_track = self.tracks[next_idx]
            next_rg = parse_replaygain_db(next_track.replaygain_track)
            self.engine.preload_next(next_track.path, next_rg)

        if autoplay:
            self.engine.play()

    def _on_lyrics(self, lrc: Dict[float, str]) -> None:
        self.lyrics = lrc

    def _on_waveform(self, peaks: np.ndarray) -> None:
        if len(peaks) > 0:
            self.waveform_peaks = peaks

    def _extract_cover_art(self, path: str) -> None:
        """Extract album art from audio file (MP3/MP4/FLAC) and push to UI."""
        try:
            from mutagen import File as MFile
            mf = MFile(path)
            if mf is None:
                self.ui.set_cover_art(None)
                return
            art = None

            # MP3/ID3: APIC
            try:
                for key in mf.keys():
                    if str(key).startswith("APIC"):
                        art = mf[key].data
                        break
            except Exception:
                pass

            # MP4/M4A: covr
            if art is None:
                try:
                    if "covr" in mf:
                        covr = mf["covr"]
                        if covr:
                            c0 = covr[0]
                            art = bytes(c0) if not isinstance(c0, bytes) else c0
                except Exception:
                    pass

            # FLAC / Vorbis pictures
            if art is None and hasattr(mf, "pictures") and mf.pictures:
                try:
                    art = mf.pictures[0].data
                except Exception:
                    pass

            if art:
                self.ui.set_cover_art(bytes(art) if not isinstance(art, bytes) else art)
            else:
                self.ui.set_cover_art(None)
        except Exception:
            self.ui.set_cover_art(None)

    def _filtered(self) -> List[tuple[int, TrackMeta]]:
        return filter_tracks(self.tracks, self.filter_text)

    def _compute_next_index(self, direction: int = 1) -> Optional[int]:
        """Next track index. Queue > repeat-track > shuffle > sequential/album."""
        n = len(self.tracks)
        if n == 0:
            return None

        if direction > 0 and self.queue:
            return self.queue[0]

        if self.state.repeat_mode == "track":
            return self.current_idx

        if self.state.shuffle:
            if n == 1:
                return self.current_idx
            # Avoid immediate repeat of current
            choices = [i for i in range(n) if i != self.current_idx]
            return random.choice(choices) if choices else self.current_idx

        if self.state.repeat_mode == "album" and self.current_track:
            album = self.current_track.album
            artist = self.current_track.artist
            album_idxs = [
                i for i, t in enumerate(self.tracks)
                if t.album == album and t.artist == artist
            ]
            if album_idxs:
                try:
                    pos = album_idxs.index(self.current_idx)
                except ValueError:
                    return album_idxs[0]
                next_pos = pos + direction
                if 0 <= next_pos < len(album_idxs):
                    return album_idxs[next_pos]
                # Wrap within album
                return album_idxs[0] if direction > 0 else album_idxs[-1]

        return (self.current_idx + direction) % n

    def _play_next(self, direction: int = 1) -> None:
        """Play next/previous respecting queue, shuffle, and repeat."""
        if not self.tracks:
            return

        if direction > 0 and self.queue:
            next_idx = self.queue.pop(0)
            self._load_track(next_idx)
            return

        next_idx = self._compute_next_index(direction)
        if next_idx is not None:
            # When advancing via N and queue was empty, don't pop
            if direction > 0 and self.queue and next_idx == self.queue[0]:
                self.queue.pop(0)
            self._load_track(next_idx)

    def _move_selection(self, delta: int) -> None:
        """Move highlight within the filtered list using global indices."""
        indexed = self._filtered()
        if not indexed:
            return
        # Find current position in filtered list
        positions = [i for i, (orig, _) in enumerate(indexed) if orig == self.current_idx]
        if positions:
            pos = positions[0]
        else:
            pos = 0
        pos = max(0, min(len(indexed) - 1, pos + delta))
        self.current_idx = indexed[pos][0]

    # ── Handle input ────────────────────────────────────────────────────

    def _handle_key(self, key: str) -> bool:
        """Process a keypress. Returns False to quit."""
        if not key:
            return True

        # ── Search mode ────────────────────────────────────────────────
        if self.search_active:
            if key in ("\x1b", "\r", "\n"):  # Esc / Enter exits search
                self.search_active = False
                return True
            if key in ("\x08", "\x7f"):  # Backspace
                self.filter_text = self.filter_text[:-1]
                return True
            if key in ("q", "Q") and not self.filter_text:
                self.search_active = False
                return True
            if len(key) == 1 and key.isprintable() and key not in "\t":
                self.filter_text += key
                # Snap selection to first match
                indexed = self._filtered()
                if indexed:
                    self.current_idx = indexed[0][0]
                return True
            return True

        # ── Quit ────────────────────────────────────────────────────────
        if key in ("q", "Q"):
            if self.filter_text:
                self.filter_text = ""
                return True
            return False

        if key == "\x1b":  # Esc
            if self.filter_text:
                self.filter_text = ""
                return True
            if self.ui.show_help:
                self.ui.show_help = False
                return True
            return False

        # ── Start search ────────────────────────────────────────────────
        if key == "/":
            self.search_active = True
            self.filter_text = ""
            return True

        # ── Queue ────────────────────────────────────────────────────
        if key in ("a", "A"):
            if self.tracks and self.current_idx not in self.queue:
                self.queue.append(self.current_idx)
            return True

        if key in ("d", "D"):
            if self.current_idx in self.queue:
                self.queue.remove(self.current_idx)
            return True

        if key == "[":
            if self.current_idx in self.queue:
                pos = self.queue.index(self.current_idx)
                if pos > 0:
                    self.queue[pos], self.queue[pos - 1] = (
                        self.queue[pos - 1], self.queue[pos]
                    )
            return True

        if key == "]":
            if self.current_idx in self.queue:
                pos = self.queue.index(self.current_idx)
                if pos < len(self.queue) - 1:
                    self.queue[pos], self.queue[pos + 1] = (
                        self.queue[pos + 1], self.queue[pos]
                    )
            return True

        if key in ("x", "X"):
            self.queue.clear()
            return True

        # ── Playback ────────────────────────────────────────────────────
        if key == " ":
            if self.engine.is_playing and not self.engine.is_paused:
                self.engine.pause()
            elif self.engine.is_playing and self.engine.is_paused:
                self.engine.resume()
            elif self.current_track is not None:
                # Track loaded but stopped — resume from position or restart
                self.engine.resume()
                if not self.engine.is_playing:
                    self._load_track(self.current_idx)
            elif self.tracks:
                self._load_track(self.current_idx)
            return True

        if key in ("\r", "\n"):
            self._load_track(self.current_idx)
            return True

        if key in ARROW_UP:
            self._move_selection(-1)
            return True

        if key in ARROW_DOWN:
            self._move_selection(1)
            return True

        if key in ("n", "N"):
            self._play_next(1)
            return True

        if key in ("p", "P"):
            self._play_next(-1)
            return True

        if key in ARROW_LEFT:
            pos = self.engine.position - SEEK_SEC
            self.engine.seek(max(0, pos))
            return True

        if key in ARROW_RIGHT:
            pos = self.engine.position + SEEK_SEC
            dur = self.engine.duration
            self.engine.seek(min(dur, pos) if dur > 0 else max(0, pos))
            return True

        # ── Volume ──────────────────────────────────────────────────────
        if key in ("+", "="):
            self.engine.set_volume(self.engine.volume + 0.05)
            self.state.volume = self.engine.volume
            if self.engine.volume > 0.01:
                self.state.muted = False
                self.engine.set_muted(False)
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
        if key == "1":
            self.state.eq_bass = min(12.0, self.state.eq_bass + EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "!":
            self.state.eq_bass = max(-12.0, self.state.eq_bass - EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "2":
            self.state.eq_mid = min(12.0, self.state.eq_mid + EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "@":
            self.state.eq_mid = max(-12.0, self.state.eq_mid - EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "3":
            self.state.eq_treble = min(12.0, self.state.eq_treble + EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True
        if key == "#":
            self.state.eq_treble = max(-12.0, self.state.eq_treble - EQ_STEP)
            self.engine.set_eq(self.state.eq_bass, self.state.eq_mid, self.state.eq_treble)
            return True

        if key == "?":
            self.ui.show_help = not self.ui.show_help
            return True

        if key in ("r", "R"):
            modes = ["off", "track", "album"]
            idx = modes.index(self.state.repeat_mode) if self.state.repeat_mode in modes else 0
            self.state.repeat_mode = modes[(idx + 1) % len(modes)]
            return True

        if key in ("s", "S"):
            self.state.shuffle = not self.state.shuffle
            return True

        if key in ("c", "C"):
            self.state.crossfade = not self.state.crossfade
            self.engine.set_crossfade(self.state.crossfade)
            return True

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
        _saved = False

        def _quit_once() -> None:
            nonlocal _saved
            if not _saved:
                _saved = True
                self._save_and_quit()

        with self.ui:
            self.ui.start_live()

            try:
                while True:
                    frame_start = time.monotonic()

                    key = read_key()
                    while key is not None:
                        if not self._handle_key(key):
                            _quit_once()
                            return
                        key = read_key()

                    # Auto-advance when track ends
                    if (
                        self.engine.is_playing
                        and not self.engine.is_paused
                        and self.engine.duration > 0
                        and self.engine.position >= self.engine.duration - 0.15
                    ):
                        self._handle_track_end()

                    # Gapless auto-switch
                    switched, new_path = self.engine.has_just_switched()
                    if switched and new_path:
                        for i, t in enumerate(self.tracks):
                            if t.path == new_path:
                                self.current_idx = i
                                self.current_track = t
                                self._track_generation += 1
                                gen = self._track_generation
                                self.lyrics = {}
                                self.waveform_peaks = None
                                self._extract_cover_art(t.path)
                                fetch_lyrics_async(
                                    t.artist, t.title,
                                    callback=lambda lrc: self._on_lyrics(lrc),
                                    generation=gen,
                                    generation_check=lambda: self._track_generation,
                                )
                                precompute_waveform_async(
                                    t.path,
                                    callback=lambda peaks: self._on_waveform(peaks),
                                    generation=gen,
                                    generation_check=lambda: self._track_generation,
                                )
                                if i in self.queue:
                                    self.queue.remove(i)
                                next_idx = self._compute_next_index()
                                if next_idx is not None and next_idx != i:
                                    nt = self.tracks[next_idx]
                                    nrg = parse_replaygain_db(nt.replaygain_track)
                                    self.engine.preload_next(nt.path, nrg)
                                break

                    position = self.engine.position
                    duration = self.engine.duration
                    playing = self.engine.is_playing and not self.engine.is_paused
                    fft_data = self.engine.get_fft_data() if playing else None
                    fft_stereo = self.engine.get_fft_stereo() if playing else None

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
                        is_playing=playing,
                        repeat_mode=self.state.repeat_mode,
                        shuffle=self.state.shuffle,
                        queue_indices=self.queue,
                        crossfade=self.state.crossfade,
                        fft_stereo=fft_stereo,
                        search_active=self.search_active,
                    )

                    elapsed = time.monotonic() - frame_start
                    time.sleep(max(0, FRAME_TIME - elapsed))

            except KeyboardInterrupt:
                pass
            finally:
                _quit_once()
                self.ui.stop_live()

    def _handle_track_end(self) -> None:
        """Called when the current track finishes playing."""
        if not self.tracks:
            return

        if self.queue:
            next_idx = self.queue.pop(0)
            self._load_track(next_idx)
            return

        if self.state.repeat_mode == "track":
            self._load_track(self.current_idx)
            return

        next_idx = self._compute_next_index(1)
        if next_idx is not None:
            # Stop if we wrapped to start with repeat off
            if (
                self.state.repeat_mode == "off"
                and not self.state.shuffle
                and next_idx <= self.current_idx
                and self.current_idx == len(self.tracks) - 1
            ):
                # Natural end of library — stop
                self.engine.stop()
                return
            self._load_track(next_idx)

    def _save_and_quit(self) -> None:
        """Persist state and stop engine."""
        try:
            self.state.volume = self.engine.volume
            self.state.position_sec = self.engine.position
            if self.current_track:
                self.state.current_track_path = self.current_track.path
            try:
                self.state.theme_index = THEMES.index(self.ui.theme)
            except ValueError:
                self.state.theme_index = 0
            try:
                self.state.waveform_index = WAVEFORM_STYLES.index(self.ui.waveform_style)
            except ValueError:
                self.state.waveform_index = 0
            self.state.queue_paths = [
                self.tracks[i].path for i in self.queue if 0 <= i < len(self.tracks)
            ]
            save_state(self.state)
        except Exception as exc:
            logger.warning("Failed to save state: %s", exc)
        try:
            self.engine.stop()
        except Exception:
            pass
        logger.info("State saved. Goodbye!")


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
        print(f"musicli v{__version__}")
        return

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
