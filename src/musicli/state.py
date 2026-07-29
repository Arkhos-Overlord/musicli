"""Persistent state management (cache + playback state) as JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List


# ── Paths ───────────────────────────────────────────────────────────────────

CACHE_FILE = Path(".cli_player_cache.json")
STATE_FILE = Path(".cli_player_state.json")


# ── Models ──────────────────────────────────────────────────────────────────

@dataclass
class TrackMeta:
    """Metadata for a single track."""

    path: str
    artist: str = "Unknown Artist"
    album: str = "Unknown Album"
    title: str = "Unknown Title"
    track_num: int = 0
    duration: float = 0.0
    replaygain_track: str | None = None
    replaygain_album: str | None = None


@dataclass
class PlayerState:
    """Playback and UI state persisted across sessions."""

    current_track_path: str = ""
    position_sec: float = 0.0
    volume: float = 0.8
    muted: bool = False
    eq_bass: float = 0.0
    eq_mid: float = 0.0
    eq_treble: float = 0.0
    theme_index: int = 0
    waveform_index: int = 0
    repeat_mode: str = "off"  # off / track / album
    shuffle: bool = False
    queue_paths: list[str] = field(default_factory=list)
    crossfade: bool = True


# ── Persistence ─────────────────────────────────────────────────────────────


def load_cache() -> List[Dict[str, Any]]:
    """Load cached library scan results."""
    if not CACHE_FILE.exists():
        return []
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_cache(tracks: List[Dict[str, Any]]) -> None:
    """Save library scan results to cache."""
    CACHE_FILE.write_text(json.dumps(tracks, indent=2), encoding="utf-8")


def load_state() -> PlayerState:
    """Load saved player state, returning defaults if missing."""
    if not STATE_FILE.exists():
        return PlayerState()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state = PlayerState()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state
    except (json.JSONDecodeError, OSError):
        return PlayerState()


def save_state(state: PlayerState) -> None:
    """Persist player state to disk."""
    STATE_FILE.write_text(
        json.dumps(asdict(state), indent=2), encoding="utf-8"
    )
