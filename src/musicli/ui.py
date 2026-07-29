"""Retro CRT/VGA terminal UI using rich: Layout, Live, Panel, box.DOUBLE."""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional

import numpy as np
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from musicli.state import TrackMeta
from musicli.themes import Theme, WAVEFORM_STYLES, WaveformStyle, DEFAULT_THEME

logger = logging.getLogger(__name__)

# ── Platform-specific non-blocking input ────────────────────────────────────

if sys.platform == "win32":
    import msvcrt

    def _kbhit() -> bool:
        return msvcrt.kbhit()

    def _getch() -> str:
        ch = msvcrt.getch()
        try:
            return ch.decode("utf-8")
        except UnicodeDecodeError:
            # Special keys come as multibyte; handle arrows, etc.
            if ch == b"\xe0" or ch == b"\x00":
                ch2 = msvcrt.getch()
                mapping = {
                    b"H": "UP",
                    b"P": "DOWN",
                    b"M": "RIGHT",
                    b"K": "LEFT",
                }
                return mapping.get(ch2, "")
            return ""

    def read_key() -> Optional[str]:
        if _kbhit():
            return _getch()
        return None

else:
    import select
    import termios
    import tty

    _old_settings: list = []

    def _setup_term() -> None:
        fd = sys.stdin.fileno()
        _old_settings.append(termios.tcgetattr(fd))
        tty.setcbreak(fd)

    def _restore_term() -> None:
        if _old_settings:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_settings.pop())

    def read_key() -> Optional[str]:
        """Read a keypress, handling multi-byte escape sequences."""
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None

        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return ch

        # Escape sequence — try to read more bytes without blocking
        # Arrows: \x1b[A (Up), \x1b[B (Down), \x1b[C (Right), \x1b[D (Left)
        if select.select([sys.stdin], [], [], 0.01)[0]:
            seq = sys.stdin.read(2)
            if len(seq) >= 2 and seq[0] == "[":
                key_map = {
                    "A": "UP",
                    "B": "DOWN",
                    "C": "RIGHT",
                    "D": "LEFT",
                }
                return key_map.get(seq[1], "\x1b")
            return "\x1b"
        # Stray escape (Esc key)
        return "\x1b"


# ── UI Helpers ──────────────────────────────────────────────────────────────


def _progress_bar(
    value: float,
    maximum: float,
    width: int = 30,
    filled_char: str = "\u2588",
    empty_char: str = "\u2591",
) -> str:
    """Render a blocky progress bar: ████████░░░░░░."""
    if maximum <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / maximum))
    filled = int(round(ratio * width))
    return filled_char * filled + empty_char * (width - filled)


def _format_time(seconds: float) -> str:
    """Format seconds as mm:ss."""
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _waveform_bar(value: float, style: WaveformStyle) -> str:
    """Map a [0, 1] amplitude to a waveform character."""
    chars = style.chars
    idx = min(int(value * (len(chars) - 1)), len(chars) - 1)
    return chars[idx]


# ── Render helpers ──────────────────────────────────────────────────────────


def _render_library(
    tracks: List[TrackMeta],
    current_idx: int,
    filter_text: str,
    theme: Theme,
    is_playing: bool,
    queue_indices: Optional[List[int]] = None,
) -> Panel:
    """Render the library (left panel) with hierarchical Artist→Album→Track."""
    # Filter
    if filter_text:
        ft = filter_text.lower()
        filtered = [
            t for t in tracks
            if ft in t.title.lower()
            or ft in t.artist.lower()
            or ft in t.album.lower()
        ]
    else:
        filtered = tracks

    qset = set(queue_indices or [])

    # Build text
    lines: list[Text] = []
    last_artist = ""
    last_album = ""

    for i, track in enumerate(filtered):
        # Hierarchy headers
        if track.artist != last_artist:
            lines.append(Text(f"\n{track.artist}", style=theme.accent))
            last_artist = track.artist
            last_album = ""
        if track.album != last_album and track.album != "Unknown Album":
            lines.append(Text(f"  {track.album}", style=theme.text_secondary))
            last_album = track.album

        # Track line
        q_mark = "[Q] " if i in qset else ""
        prefix = "▶ " if (i == current_idx and is_playing) else "  "
        tn = f"{track.track_num:02d}. " if track.track_num else ""
        dur = _format_time(track.duration)

        line = Text()
        if q_mark and i != current_idx:
            line.append(q_mark, style=theme.text_secondary)
        if i == current_idx:
            line.append(f"{prefix}{q_mark}{tn}{track.title}", style=theme.accent)
        else:
            line.append(f"{prefix}{q_mark}{tn}{track.title}", style=theme.text_primary)
        line.append(f"  {dur}", style=theme.text_secondary)
        lines.append(line)

    content = Text("\n").join(lines) if lines else Text("No tracks found.\nDrop .mp3/.flac/.wav files here.", style=theme.text_secondary)

    # Search indicator
    title = "LIBRARY"
    if filter_text:
        title = f"LIBRARY [/{filter_text}]"
    if qset:
        title += f"  Q:{len(qset)}"

    return Panel(
        content,
        title=title,
        border_style=theme.border,
        box=box.DOUBLE,
        padding=(0, 1),
        style=f"on {theme.bg}",
    )


def _render_now_playing(
    track: Optional[TrackMeta],
    position_sec: float,
    duration: float,
    volume: float,
    muted: bool,
    eq_bass: float,
    eq_mid: float,
    eq_treble: float,
    lyrics: Dict[float, str],
    theme: Theme,
    repeat_mode: str,
    shuffle: bool,
) -> Panel:
    """Render the Now Playing panel (top-right)."""
    lines: list[Text] = []

    if track is None:
        lines.append(Text("No track loaded", style=theme.text_secondary))
        return Panel(
            Text("\n").join(lines),
            title="NOW PLAYING",
            border_style=theme.border,
            box=box.DOUBLE,
            padding=(0, 1),
        )

    # ── Track info ──────────────────────────────────────────────────────
    lines.append(Text(f"{track.title}", style=theme.accent))
    lines.append(Text(f"{track.artist}  |  {track.album}", style=theme.text_primary))
    lines.append(Text(""))

    # ── Seek bar ────────────────────────────────────────────────────────
    pos_str = _format_time(position_sec)
    dur_str = _format_time(duration)
    bar = _progress_bar(position_sec, duration, width=40)
    lines.append(Text(f"{pos_str} {bar} {dur_str}", style=theme.text_primary))
    lines.append(Text(""))

    # ── Volume bar ──────────────────────────────────────────────────────
    vol_label = "MUTE" if muted else f"VOL {int(volume*100):d}%"
    vol_bar = _progress_bar(0.0 if muted else volume, 1.0, width=20)
    lines.append(Text(f"{vol_label} {vol_bar}", style=theme.text_primary))

    # ── EQ indicators ───────────────────────────────────────────────────
    def _eq_bar(val: float, label: str, width: int = 12) -> str:
        val_clamped = max(-12, min(12, val))
        ratio = (val_clamped + 12) / 24
        filled = int(round(ratio * width))
        bar = "─" * filled + " " * (width - filled)
        return f"{label} [{bar}] {val:+.0f}dB"

    eq_line = (
        f"{_eq_bar(eq_bass, 'B')}  "
        f"{_eq_bar(eq_mid, 'M')}  "
        f"{_eq_bar(eq_treble, 'T')}"
    )
    lines.append(Text(eq_line, style=theme.text_secondary))

    # ── Status line ─────────────────────────────────────────────────────
    flags = []
    if repeat_mode != "off":
        flags.append(f"RPT:{repeat_mode[0].upper()}")
    if shuffle:
        flags.append("SHF")
    status = " | ".join(flags) if flags else ""
    lines.append(Text(status, style=theme.text_secondary))

    # ── Synced lyrics ──────────────────────────────────────────────────
    if lyrics:
        lines.append(Text(""))
        lines.append(Text("─" * 40, style=theme.text_secondary))
        sorted_times = sorted(lyrics.keys())

        # Find active and next lyric
        active_lyric = ""
        next_lyric = ""
        for t in sorted_times:
            if t <= position_sec:
                active_lyric = lyrics[t]
            elif next_lyric == "":
                next_lyric = lyrics[t]

        # Past lyric (dim)
        if active_lyric:
            lines.append(Text(f"  {active_lyric}", style=theme.accent_dim))

        # Current lyric (highlighted with block cursor)
        if next_lyric:
            lines.append(Text(f"▌ {next_lyric}", style=theme.accent))

    return Panel(
        Text("\n").join(lines),
        title="NOW PLAYING",
        border_style=theme.border,
        box=box.DOUBLE,
        padding=(0, 1),
        style=f"on {theme.bg}",
    )


def _render_visualizer(
    fft_data: Optional[np.ndarray],
    waveform_peaks: Optional[np.ndarray],
    position_sec: float,
    duration: float,
    theme: Theme,
    waveform_style: WaveformStyle,
) -> Panel:
    """Render the visualizer panel (bottom-right): VU meter + waveform."""
    lines: list[Text] = []

    # ── Real-time VU / EQ meter ────────────────────────────────────────
    if fft_data is not None and len(fft_data) > 0:
        # Simple FFT for frequency bins
        try:
            fft = np.abs(np.fft.rfft(fft_data))
            # Group into 8 frequency bands
            n_bins = 8
            band_size = max(1, len(fft) // n_bins)
            bands = np.zeros(n_bins)
            for i in range(n_bins):
                start = i * band_size
                end = start + band_size
                bands[i] = np.mean(fft[start:end]) if end <= len(fft) else 0

            # Normalise
            bmax = bands.max()
            if bmax > 0:
                bands /= bmax

            # Render as vertical bars
            bar_chars = "▁▂▃▄▅▆▇█"
            viz = ""
            for b in bands:
                idx = min(int(b * 7), 7)
                viz += bar_chars[idx] * 2
            lines.append(Text(viz, style=theme.accent))
            lines.append(Text(""))
        except Exception:
            pass

    # ── Waveform ────────────────────────────────────────────────────────
    if waveform_peaks is not None and len(waveform_peaks) > 0:
        wf_line = ""
        for peak in waveform_peaks:
            wf_line += _waveform_bar(float(peak), waveform_style)
        lines.append(Text(wf_line, style=theme.text_primary))

        # Position marker
        if duration > 0:
            ratio = position_sec / duration
            marker_pos = min(int(ratio * len(waveform_peaks)), len(waveform_peaks) - 1)
            marker_line = " " * marker_pos + "▲"
            lines.append(Text(marker_line, style=theme.accent))

    return Panel(
        Text("\n").join(lines) if lines else Text("", style=theme.text_secondary),
        title=f"VISUALIZER [{waveform_style.name}]",
        border_style=theme.border,
        box=box.DOUBLE,
        padding=(0, 1),
        style=f"on {theme.bg}",
    )


def _render_header(theme: Theme, app_name: str = "musicli", crossfade: bool = True) -> Panel:
    """Top bar with app name and controls hint."""
    text = Text()
    text.append(f"  {app_name}  ", style=theme.accent)
    text.append("│", style=theme.text_secondary)
    cf_str = "XF:ON " if crossfade else "XF:OFF"
    text.append(f" {cf_str} ", style=theme.accent if crossfade else theme.text_secondary)
    text.append("│", style=theme.text_secondary)
    text.append(
        " SPACE:P/P  ENTER:Play  N/P:Prev  +/-:Vol  ←→:Seek  123:EQ  C:Crossf  "
        "W:Wave  A:Q+ D:Q- [ ]:Move  X:ClrQ  /:Search  Q:Quit",
        style=theme.text_secondary,
    )
    return Panel(text, border_style=theme.border, box=box.DOUBLE, style=f"on {theme.bg}")


# ── Main UI Class ───────────────────────────────────────────────────────────


class MusicliUI:
    """Full-screen rich-based retro music player interface."""

    def __init__(self, app_name: str = "musicli") -> None:
        self.app_name = app_name
        self.layout = self._build_layout()
        self._live: Optional[Live] = None

        # Current theme / style
        self.theme: Theme = DEFAULT_THEME
        self.waveform_style: WaveformStyle = WAVEFORM_STYLES[0]

    def _build_layout(self) -> Layout:
        """Construct the 3-panel layout."""
        root = Layout(name="root")
        root.split(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        root["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )
        root["right"].split(
            Layout(name="now_playing", ratio=2),
            Layout(name="visualizer", ratio=1),
        )
        return root

    # ── Lifecycle ──────────────────────────────────────────────────────

    def __enter__(self) -> MusicliUI:
        if sys.platform != "win32":
            from musicli import ui as _ui_mod
            _ui_mod._setup_term()
        return self

    def __exit__(self, *args) -> None:
        if sys.platform != "win32":
            from musicli import ui as _ui_mod
            _ui_mod._restore_term()
        if self._live:
            self._live.stop()

    def start_live(self) -> Live:
        """Begin the Live display context and return it."""
        self._live = Live(
            self.layout,
            refresh_per_second=15,
            screen=True,
        )
        self._live.start()
        return self._live

    def stop_live(self) -> None:
        if self._live:
            self._live.stop()

    # ── Update ─────────────────────────────────────────────────────────

    def update(
        self,
        tracks: List[TrackMeta],
        current_idx: int,
        current_track: Optional[TrackMeta],
        position_sec: float,
        duration: float,
        volume: float,
        muted: bool,
        eq_bass: float,
        eq_mid: float,
        eq_treble: float,
        lyrics: Dict[float, str],
        filter_text: str,
        fft_data: Optional[np.ndarray],
        waveform_peaks: Optional[np.ndarray],
        is_playing: bool,
        repeat_mode: str,
        shuffle: bool,
        queue_indices: Optional[List[int]] = None,
        crossfade: bool = True,
    ) -> None:
        """Push the latest state into the layout panels."""
        self.layout["header"].update(_render_header(self.theme, self.app_name, crossfade))
        self.layout["left"].update(
            _render_library(tracks, current_idx, filter_text, self.theme, is_playing, queue_indices)
        )
        self.layout["now_playing"].update(
            _render_now_playing(
                current_track, position_sec, duration,
                volume, muted, eq_bass, eq_mid, eq_treble,
                lyrics, self.theme, repeat_mode, shuffle,
            )
        )
        self.layout["visualizer"].update(
            _render_visualizer(
                fft_data, waveform_peaks, position_sec, duration,
                self.theme, self.waveform_style,
            )
        )

    def cycle_theme(self) -> None:
        """Rotate to the next theme."""
        from musicli.themes import THEMES
        idx = THEMES.index(self.theme) if self.theme in THEMES else -1
        self.theme = THEMES[(idx + 1) % len(THEMES)]

    def cycle_waveform(self) -> None:
        """Rotate to the next waveform style."""
        idx = WAVEFORM_STYLES.index(self.waveform_style) if self.waveform_style in WAVEFORM_STYLES else -1
        self.waveform_style = WAVEFORM_STYLES[(idx + 1) % len(WAVEFORM_STYLES)]
