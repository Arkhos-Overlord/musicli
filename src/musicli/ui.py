"""Retro CRT/VGA terminal UI using rich: Layout, Live, Panel, box.DOUBLE.

v3 — Fixed help overlay, cover art styles, library scroll + filter indices.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from musicli.state import TrackMeta
from musicli.themes import Theme, WAVEFORM_STYLES, WaveformStyle, DEFAULT_THEME

logger = logging.getLogger(__name__)

# ── Cover art size (in character cells) ─────────────────────────────────────
COVER_WIDTH = 28
COVER_HEIGHT = 14

# Visible library rows (scroll viewport)
LIBRARY_VIEWPORT = 28


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
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return ch
        if select.select([sys.stdin], [], [], 0.01)[0]:
            seq = sys.stdin.read(2)
            if len(seq) >= 2 and seq[0] == "[":
                key_map = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
                return key_map.get(seq[1], "\x1b")
            return "\x1b"
        return "\x1b"


# ── Cover art rendering (ANSI half-blocks) ──────────────────────────────────


def render_cover_art(
    img_bytes: Optional[bytes],
    width: int = COVER_WIDTH,
    height: int = COVER_HEIGHT,
) -> Optional[Text]:
    """Convert album art (JPEG/PNG bytes) to an ANSI half-block Text object."""
    if img_bytes is None or len(img_bytes) < 100:
        return None
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize((width, height * 2), Image.LANCZOS)
        rgb = np.array(img, dtype=np.uint8)
    except Exception:
        return None

    lines: list[Text] = []
    for row in range(height):
        t = Text()
        for col in range(width):
            r_top, g_top, b_top = rgb[row * 2, col]
            r_bot, g_bot, b_bot = rgb[min(row * 2 + 1, height * 2 - 1), col]
            fg = f"#{r_bot:02x}{g_bot:02x}{b_bot:02x}"
            bg = f"#{r_top:02x}{g_top:02x}{b_top:02x}"
            t.append("▄", style=f"{fg} on {bg}")
        lines.append(t)
    return Text("\n").join(lines)


# ── UI Helpers ──────────────────────────────────────────────────────────────


def _progress_bar(
    value: float,
    maximum: float,
    width: int = 30,
    filled_char: str = "\u2588",
    empty_char: str = "\u2591",
) -> str:
    if maximum <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / maximum))
    filled = int(round(ratio * width))
    return filled_char * filled + empty_char * (width - filled)


def _format_time(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        seconds = 0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def _waveform_bar(value: float, style: WaveformStyle) -> str:
    chars = style.chars
    if not chars:
        return " "
    idx = min(max(0, int(value * (len(chars) - 1))), len(chars) - 1)
    return chars[idx]


def filter_tracks(tracks: List[TrackMeta], filter_text: str) -> List[Tuple[int, TrackMeta]]:
    """Return (original_index, track) pairs matching the filter."""
    if not filter_text:
        return [(i, t) for i, t in enumerate(tracks)]
    ft = filter_text.lower()
    return [
        (i, t) for i, t in enumerate(tracks)
        if ft in t.title.lower()
        or ft in t.artist.lower()
        or ft in t.album.lower()
    ]


# ── Help Overlay ────────────────────────────────────────────────────────────

HELP_TEXT = Text.from_markup(
    """[bold #00ff00].──────────────────────────────────────────────────────.
│                   MUSICLI KEYS                   │
├──────────────────────────────────────────────────┤
│  q / Esc      Quit (or clear search)             │
│  ?            Toggle this help overlay            │
│                                                  │
│  k / ↑        Select previous track              │
│  j / ↓        Select next track                  │
│  Enter        Play selected track                 │
│  Space        Play / Pause                       │
│  n / p        Next / Previous track              │
│                                                  │
│  ← / →        Seek -10s / +10s                  │
│  + / -        Volume up / down                   │
│  m            Toggle mute                        │
│                                                  │
│  1 / !        Bass +/-                          │
│  2 / @        Mid +/-                           │
│  3 / #        Treble +/-                        │
│                                                  │
│  /            Start search filter                │
│  a            Add current track to queue          │
│  d            Remove from queue                   │
│  [ / ]        Move queue item                    │
│  x            Clear queue                        │
│                                                  │
│  r            Repeat (off → track → album)       │
│  s            Toggle shuffle                     │
│  c            Toggle crossfade                   │
│  t            Cycle theme                        │
│  w            Cycle waveform style               │
'──────────────────────────────────────────────────'[/]"""
)


# ── Render helpers ──────────────────────────────────────────────────────────


def _render_library(
    tracks: List[TrackMeta],
    current_idx: int,
    filter_text: str,
    theme: Theme,
    is_playing: bool,
    queue_indices: Optional[List[int]] = None,
    viewport: int = LIBRARY_VIEWPORT,
    search_active: bool = False,
) -> Panel:
    """Render the library with scroll viewport and correct global indices."""
    indexed = filter_tracks(tracks, filter_text)
    qset = set(queue_indices or [])

    # Find highlight position within filtered list
    highlight_pos = 0
    for pos, (orig_i, _) in enumerate(indexed):
        if orig_i == current_idx:
            highlight_pos = pos
            break
    else:
        # current_idx not in filter — clamp highlight
        if indexed:
            # Keep current_idx as-is; just scroll to top of filtered
            highlight_pos = 0

    # Scroll window so highlight stays visible
    n = len(indexed)
    if n <= viewport:
        start, end = 0, n
    else:
        half = viewport // 2
        start = max(0, highlight_pos - half)
        end = start + viewport
        if end > n:
            end = n
            start = max(0, end - viewport)

    lines: list[Text] = []
    last_artist = ""
    last_album = ""
    visible = indexed[start:end]

    for orig_i, track in visible:
        if track.artist != last_artist:
            lines.append(Text(f"{track.artist}", style=theme.accent))
            last_artist = track.artist
            last_album = ""
        if track.album != last_album and track.album != "Unknown Album":
            lines.append(Text(f"  {track.album}", style=theme.text_secondary))
            last_album = track.album

        q_mark = "[Q] " if orig_i in qset else ""
        prefix = "▶ " if (orig_i == current_idx and is_playing) else (
            "▸ " if orig_i == current_idx else "  "
        )
        tn = f"{track.track_num:02d}. " if track.track_num else ""
        dur = _format_time(track.duration)

        line = Text()
        if orig_i == current_idx:
            line.append(f"{prefix}{q_mark}{tn}{track.title}", style=theme.accent)
        else:
            line.append(f"{prefix}{q_mark}{tn}{track.title}", style=theme.text_primary)
        line.append(f"  {dur}", style=theme.text_secondary)
        lines.append(line)

    if start > 0:
        lines.insert(0, Text(f"  ↑ {start} more…", style=theme.text_secondary))
    if end < n:
        lines.append(Text(f"  ↓ {n - end} more…", style=theme.text_secondary))

    if not lines:
        content: Text | Group = Text(
            "No tracks found.\nDrop .mp3/.flac/.wav here\nor run from your music folder.",
            style=theme.text_secondary,
        )
    else:
        content = Text("\n").join(lines)

    title = "LIBRARY"
    if search_active or filter_text:
        title = f"LIBRARY [/{filter_text}█]" if search_active else f"LIBRARY [/{filter_text}]"
    if qset:
        title += f"  Q:{len(qset)}"
    if n:
        title += f"  ({n})"

    return Panel(
        content,
        title=title,
        border_style=theme.border,
        box=box.DOUBLE,
        padding=(0, 1),
        style=f"on {theme.bg}",
    )


def _eq_bar(val: float, label: str, width: int = 10) -> str:
    val_c = max(-12.0, min(12.0, val))
    ratio = (val_c + 12) / 24
    filled = int(round(ratio * width))
    bar = "─" * filled + " " * (width - filled)
    return f"{label} [{bar}] {val:+.0f}dB"


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
    cover_art: Optional[Text] = None,
) -> Panel:
    """Render the Now Playing panel with optional cover art (styles preserved)."""
    if track is None:
        return Panel(
            Text("No track loaded", style=theme.text_secondary),
            title="NOW PLAYING",
            border_style=theme.border,
            box=box.DOUBLE,
            padding=(0, 1),
            style=f"on {theme.bg}",
        )

    info_parts: list[Text] = []
    info_parts.append(Text(track.title, style=theme.accent))
    info_parts.append(Text(f"{track.artist}  |  {track.album}", style=theme.text_primary))
    if track.track_num:
        info_parts.append(Text(f"Track {track.track_num}", style=theme.text_secondary))
    info_parts.append(Text(""))

    pos_str = _format_time(position_sec)
    dur_str = _format_time(duration)
    bar_w = 30 if cover_art else 40
    bar = _progress_bar(position_sec, duration, width=bar_w)
    info_parts.append(Text(f"{pos_str} {bar} {dur_str}", style=theme.text_primary))
    info_parts.append(Text(""))

    vol_label = "MUTE" if muted else f"VOL {int(volume * 100):d}%"
    vol_bar = _progress_bar(0.0 if muted else volume, 1.0, width=15 if cover_art else 20)
    info_parts.append(Text(f"{vol_label} {vol_bar}", style=theme.text_primary))

    eq_line = (
        f"{_eq_bar(eq_bass, 'B', 8 if cover_art else 12)}  "
        f"{_eq_bar(eq_mid, 'M', 8 if cover_art else 12)}  "
        f"{_eq_bar(eq_treble, 'T', 8 if cover_art else 12)}"
    )
    info_parts.append(Text(eq_line, style=theme.text_secondary))

    flags = []
    if repeat_mode != "off":
        flags.append(f"RPT:{repeat_mode[:3].upper()}")
    if shuffle:
        flags.append("SHF")
    if flags:
        info_parts.append(Text(" | ".join(flags), style=theme.text_secondary))

    # Lyrics
    if lyrics:
        info_parts.append(Text(""))
        info_parts.append(Text("─" * 36, style=theme.text_secondary))
        sorted_times = sorted(lyrics.keys())
        active_lyric = ""
        next_lyric = ""
        for t in sorted_times:
            if t <= position_sec:
                active_lyric = lyrics[t]
            elif not next_lyric:
                next_lyric = lyrics[t]
                break
        if active_lyric:
            info_parts.append(Text(f"  {active_lyric}", style=theme.accent_dim))
        if next_lyric:
            info_parts.append(Text(f"▌ {next_lyric}", style=theme.accent))

    info_block = Text("\n").join(info_parts)

    if cover_art is not None:
        # Preserve cover art styles via Columns (no str() conversion)
        body: Group | Text | Columns = Columns(
            [cover_art, info_block],
            equal=False,
            expand=True,
            padding=(0, 2),
        )
    else:
        body = info_block

    return Panel(
        body,
        title="NOW PLAYING",
        border_style=theme.border,
        box=box.DOUBLE,
        padding=(0, 1),
        style=f"on {theme.bg}",
    )


def _spectrum_bars(channel: np.ndarray, n_bars: int = 16) -> str:
    fft = np.abs(np.fft.rfft(channel.astype(np.float64)))
    if len(fft) == 0:
        return "▁" * n_bars
    band_size = max(1, len(fft) // n_bars)
    bands = np.zeros(n_bars)
    for i in range(n_bars):
        start = i * band_size
        end = start + band_size
        bands[i] = np.mean(fft[start:end]) if start < len(fft) else 0
    bmax = bands.max()
    if bmax > 0:
        bands /= bmax
    bar_chars = "▁▂▃▄▅▆▇█"
    out = ""
    for b in bands:
        idx = min(int(b * 7), 7)
        out += bar_chars[idx]
    return out


def _render_visualizer(
    fft_data: Optional[np.ndarray],
    waveform_peaks: Optional[np.ndarray],
    position_sec: float,
    duration: float,
    theme: Theme,
    waveform_style: WaveformStyle,
    fft_stereo: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Panel:
    """Render stereo VU bars + waveform + position marker."""
    lines: list[Text] = []

    if fft_stereo is not None:
        try:
            left_ch, right_ch = fft_stereo
            left_bars = _spectrum_bars(left_ch, 16)
            right_bars = _spectrum_bars(right_ch, 16)
            vu = Text()
            vu.append(" L ", style=theme.accent)
            color0 = theme.viz_colors[0] if theme.viz_colors else theme.accent
            color1 = theme.viz_colors[1] if len(theme.viz_colors) > 1 else theme.accent
            vu.append(left_bars, style=color0)
            vu.append("\n")
            vu.append(" R ", style=theme.accent)
            vu.append(right_bars, style=color1)
            lines.append(vu)
            lines.append(Text(""))
        except Exception:
            pass
    elif fft_data is not None and len(fft_data) > 0:
        try:
            viz = _spectrum_bars(fft_data, 8)
            # Widen mono bars
            lines.append(Text("".join(c * 2 for c in viz), style=theme.accent))
            lines.append(Text(""))
        except Exception:
            pass

    if waveform_peaks is not None and len(waveform_peaks) > 0:
        wf_line = "".join(_waveform_bar(float(p), waveform_style) for p in waveform_peaks)
        lines.append(Text(wf_line, style=theme.text_primary))
        if duration > 0:
            ratio = max(0.0, min(1.0, position_sec / duration))
            marker_pos = min(int(ratio * len(waveform_peaks)), len(waveform_peaks) - 1)
            marker_line = " " * marker_pos + "▲"
            lines.append(Text(marker_line, style=theme.accent))

    return Panel(
        Text("\n").join(lines) if lines else Text("  (no signal)", style=theme.text_secondary),
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
        " SPACE:P/P  ENTER:Play  N/P:Skip  +/-:Vol  ←→:Seek"
        "  123:EQ  /:Search  ?:Help  Q:Quit",
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

        self.theme: Theme = DEFAULT_THEME
        self.waveform_style: WaveformStyle = WAVEFORM_STYLES[0]
        self.show_help: bool = False
        self.cover_art: Optional[Text] = None
        self.search_active: bool = False

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

    def _rebuild_body(self) -> None:
        """Restore the multi-panel body after help overlay."""
        body = self.layout["body"]
        # Clear any full-body renderable by rebuilding children
        body.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=2),
        )
        body["right"].split(
            Layout(name="now_playing", ratio=2),
            Layout(name="visualizer", ratio=1),
        )

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

    def set_cover_art(self, img_bytes: Optional[bytes]) -> None:
        """Extract and cache cover art Text from raw image bytes."""
        self.cover_art = render_cover_art(img_bytes) if img_bytes else None

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
        fft_stereo: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        search_active: bool = False,
    ) -> None:
        """Push the latest state into the layout panels."""
        self.search_active = search_active

        # Full-screen help: replace root content so Rich actually paints it
        # (body has children → leaf-only render would ignore body.update)
        if self.show_help:
            help_panel = Panel(
                Align.center(HELP_TEXT, vertical="middle"),
                title="HELP  —  press ? to close",
                border_style="#00ff00",
                box=box.DOUBLE,
                padding=(1, 2),
                style="on #000800",
            )
            # Swap entire layout tree for a simple help layout
            help_layout = Layout(name="root")
            help_layout.split(
                Layout(name="header", size=3),
                Layout(name="help_body"),
            )
            help_layout["header"].update(
                _render_header(self.theme, self.app_name, crossfade)
            )
            help_layout["help_body"].update(help_panel)
            if self._live:
                self._live.update(help_layout)
            self._help_showing = True
            return

        # Leaving help: restore multi-panel layout on the Live display
        if getattr(self, "_help_showing", False):
            self.layout = self._build_layout()
            if self._live:
                self._live.update(self.layout)
            self._help_showing = False

        self.layout["header"].update(
            _render_header(self.theme, self.app_name, crossfade)
        )
        self.layout["left"].update(
            _render_library(
                tracks, current_idx, filter_text, self.theme, is_playing,
                queue_indices, search_active=search_active,
            )
        )
        self.layout["now_playing"].update(
            _render_now_playing(
                current_track, position_sec, duration,
                volume, muted, eq_bass, eq_mid, eq_treble,
                lyrics, self.theme, repeat_mode, shuffle,
                cover_art=self.cover_art,
            )
        )
        self.layout["visualizer"].update(
            _render_visualizer(
                fft_data, waveform_peaks, position_sec, duration,
                self.theme, self.waveform_style,
                fft_stereo=fft_stereo,
            )
        )

    def cycle_theme(self) -> None:
        from musicli.themes import THEMES
        idx = THEMES.index(self.theme) if self.theme in THEMES else -1
        self.theme = THEMES[(idx + 1) % len(THEMES)]

    def cycle_waveform(self) -> None:
        idx = (
            WAVEFORM_STYLES.index(self.waveform_style)
            if self.waveform_style in WAVEFORM_STYLES
            else -1
        )
        self.waveform_style = WAVEFORM_STYLES[(idx + 1) % len(WAVEFORM_STYLES)]
