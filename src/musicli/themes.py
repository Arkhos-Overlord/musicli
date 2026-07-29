"""Retro CRT/VGA themes and waveform style definitions for musicli."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Theme:
    """A retro terminal theme."""

    name: str
    text_primary: str  # main text color
    text_secondary: str  # dim / secondary text
    accent: str  # highlights, now-playing marker
    accent_dim: str  # dimmed accent (past lyrics, etc.)
    border: str  # panel borders
    bg: str  # background
    viz_colors: List[str] = field(default_factory=lambda: ["yellow", "orange", "red"])


# ── Theme Definitions ───────────────────────────────────────────────────────

CRT_AMBER = Theme(
    name="CRT Amber",
    text_primary="bold #ffb000",
    text_secondary="dim #aa7700",
    accent="bold #ffcc00 on #1a1500",
    accent_dim="dim #665500",
    border="#aa7700",
    bg="#0a0800",
    viz_colors=["#ffcc00", "#ff9900", "#ff6600"],
)

VGA_SYNTH = Theme(
    name="VGA Synthwave",
    text_primary="bold #00ffff",
    text_secondary="dim #0088aa",
    accent="bold #ff00ff on #0d0221",
    accent_dim="dim #550055",
    border="#00aaff",
    bg="#050010",
    viz_colors=["#00ffff", "#ff00ff", "#8844ff"],
)

# Default
DEFAULT_THEME = CRT_AMBER

# Theme cycle list
THEMES: List[Theme] = [CRT_AMBER, VGA_SYNTH]


# ── Waveform Styles ─────────────────────────────────────────────────────────

class WaveformStyle:
    """A waveform visualisation style."""

    name: str
    chars: str  # characters from low to high amplitude
    bar_width: int = 1

    def __init__(self, name: str, chars: str, bar_width: int = 1) -> None:
        self.name = name
        self.chars = chars
        self.bar_width = bar_width


WAVEFORM_STYLES: List[WaveformStyle] = [
    WaveformStyle("Blocks", "▁▂▃▄▅▆▇█", bar_width=1),
    WaveformStyle("Dots", "•∙·", bar_width=1),
    WaveformStyle("Solid", "█▓▒░", bar_width=2),
]

DEFAULT_WAVEFORM = WAVEFORM_STYLES[0]
