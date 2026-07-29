"""Audio file scanner: discovers .mp3/.flac/.wav files and extracts metadata."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import mutagen
from mutagen.flac import FLAC
from mutagen.id3 import ID3
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from musicli.state import TrackMeta, save_cache, load_cache

logger = logging.getLogger(__name__)

# Supported extensions
SUPPORTED_EXTS = {".mp3", ".flac", ".wav"}


# ── Metadata extraction ─────────────────────────────────────────────────────


def _extract_metadata(filepath: Path) -> TrackMeta:
    """Extract Artist, Album, Title, Track#, Duration, ReplayGain from a
    file. Falls back to filename parsing when tags are missing."""
    meta = TrackMeta(path=str(filepath))
    filename = filepath.stem

    try:
        audio = mutagen.File(str(filepath))
    except Exception:
        # Can't read at all — fall back entirely to filename
        return _fallback_filename(meta, filename)

    if audio is None:
        return _fallback_filename(meta, filename)

    # ── Duration ────────────────────────────────────────────────────────
    try:
        meta.duration = audio.info.length
    except Exception:
        meta.duration = 0.0

    # ── Tags ────────────────────────────────────────────────────────────
    artist: str | None = None
    album: str | None = None
    title: str | None = None
    track_num: int = 0
    rg_track: float | None = None
    rg_album: float | None = None

    if isinstance(audio, (MP3, WAVE)):
        # ID3 tags
        tags = audio.tags if audio.tags else {}
        artist = _id3_text(tags, "TPE1")
        album = _id3_text(tags, "TALB")
        title = _id3_text(tags, "TIT2")
        tn = _id3_text(tags, "TRCK")
        if tn:
            track_num = _parse_track_number(tn)

        # ReplayGain in TXXX frames
        rg_track = _id3_txxx(tags, "REPLAYGAIN_TRACK_GAIN")
        rg_album = _id3_txxx(tags, "REPLAYGAIN_ALBUM_GAIN")

    elif isinstance(audio, FLAC):
        # Vorbis comments
        artist = _vc_first(audio, "artist")
        album = _vc_first(audio, "album")
        title = _vc_first(audio, "title")
        tn = _vc_first(audio, "tracknumber")
        if tn:
            track_num = _parse_track_number(tn)

        rg_track = _vc_first(audio, "replaygain_track_gain")
        rg_album = _vc_first(audio, "replaygain_album_gain")

    # Apply or fallback
    meta.artist = artist or "Unknown Artist"
    meta.album = album or "Unknown Album"
    meta.title = title or filename
    meta.track_num = track_num
    meta.replaygain_track = rg_track
    meta.replaygain_album = rg_album

    return meta


def _fallback_filename(meta: TrackMeta, filename: str) -> TrackMeta:
    """Attempt 'Artist - Title' parsing from filename."""
    meta.title = filename
    if " - " in filename:
        parts = filename.split(" - ", 1)
        meta.artist = parts[0].strip()
        meta.title = parts[1].strip()
    return meta


def _id3_text(tags: Dict[str, Any], key: str) -> str | None:
    """Get the first text value of an ID3 frame."""
    frame = tags.get(key)
    if frame and hasattr(frame, "text") and frame.text:
        return str(frame.text[0])
    return None


def _id3_txxx(tags: Dict[str, Any], desc: str) -> str | None:
    """Get a TXXX (user-defined text) frame by description."""
    for frame in tags.values():
        if frame.FrameID == "TXXX" and frame.desc.lower() == desc.lower():
            return str(frame.text[0])
    return None


def _vc_first(audio: FLAC, key: str) -> str | None:
    """Get the first value of a Vorbis comment."""
    vals = audio.get(key)
    if vals:
        return str(vals[0])
    return None


def _parse_track_number(tn: str) -> int:
    """Parse '04' or '4/12' → int."""
    try:
        return int(tn.split("/")[0].strip())
    except (ValueError, AttributeError):
        return 0


# ── Scanning ────────────────────────────────────────────────────────────────


def scan_directory(
    directory: Path | None = None,
    force: bool = False,
) -> List[TrackMeta]:
    """Scan directory for audio files, returning sorted TrackMeta list.

    Uses a JSON cache to avoid re-reading tags every launch unless
    `force=True`. The cache key is filepath → mtime.
    """
    directory = directory or Path.cwd()
    logger.info("Scanning %s for audio files...", directory)

    # Load existing cache
    cache = load_cache()
    cache_map: Dict[str, Dict[str, Any]] = {
        c["path"]: c for c in cache if "path" in c
    }

    tracks: list[TrackMeta] = []
    for filepath in directory.iterdir():
        if filepath.suffix.lower() not in SUPPORTED_EXTS:
            continue

        mtime = filepath.stat().st_mtime

        # Check cache
        cached = cache_map.get(str(filepath))
        if not force and cached and cached.get("_mtime") == mtime:
            # Restore from cache
            meta = TrackMeta(
                path=cached["path"],
                artist=cached.get("artist", "Unknown Artist"),
                album=cached.get("album", "Unknown Album"),
                title=cached.get("title", "Unknown Title"),
                track_num=cached.get("track_num", 0),
                duration=cached.get("duration", 0.0),
                replaygain_track=cached.get("replaygain_track"),
                replaygain_album=cached.get("replaygain_album"),
            )
            tracks.append(meta)
            continue

        # Fresh extraction
        logger.debug("Reading metadata: %s", filepath.name)
        meta = _extract_metadata(filepath)
        tracks.append(meta)

    # ── Sort: Artist → Album → Track# ──────────────────────────────────
    tracks.sort(key=lambda t: (
        t.artist.lower(),
        t.album.lower(),
        t.track_num,
        t.title.lower(),
    ))

    # ── Save cache ──────────────────────────────────────────────────────
    cache_data: list[Dict[str, Any]] = []
    for t in tracks:
        entry: Dict[str, Any] = {
            "path": t.path,
            "artist": t.artist,
            "album": t.album,
            "title": t.title,
            "track_num": t.track_num,
            "duration": t.duration,
            "replaygain_track": t.replaygain_track,
            "replaygain_album": t.replaygain_album,
        }
        try:
            entry["_mtime"] = Path(t.path).stat().st_mtime
        except OSError:
            entry["_mtime"] = 0
        cache_data.append(entry)

    save_cache(cache_data)
    logger.info("Scanned %d tracks.", len(tracks))
    return tracks
