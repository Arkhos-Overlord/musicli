"""Audio file scanner: discovers .mp3/.flac/.wav files and extracts metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import mutagen
from mutagen.flac import FLAC
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from musicli.state import TrackMeta, save_cache, load_cache

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".opus"}

# Skip these directory names while walking
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".venv", "venv", "dist", "build", ".tox",
}


def _extract_metadata(filepath: Path) -> TrackMeta:
    """Extract Artist, Album, Title, Track#, Duration, ReplayGain from a file."""
    meta = TrackMeta(path=str(filepath.resolve()))
    filename = filepath.stem

    try:
        audio = mutagen.File(str(filepath), easy=False)
    except Exception:
        return _fallback_filename(meta, filename)

    if audio is None:
        return _fallback_filename(meta, filename)

    try:
        meta.duration = float(getattr(audio.info, "length", 0.0) or 0.0)
    except Exception:
        meta.duration = 0.0

    artist: str | None = None
    album: str | None = None
    title: str | None = None
    track_num: int = 0
    rg_track: str | None = None
    rg_album: str | None = None

    if isinstance(audio, MP3) or (audio.tags and hasattr(audio.tags, "getall")):
        tags = audio.tags if audio.tags else {}
        artist = _id3_text(tags, "TPE1")
        album = _id3_text(tags, "TALB")
        title = _id3_text(tags, "TIT2")
        tn = _id3_text(tags, "TRCK")
        if tn:
            track_num = _parse_track_number(tn)
        rg_track = _id3_txxx(tags, "REPLAYGAIN_TRACK_GAIN")
        rg_album = _id3_txxx(tags, "REPLAYGAIN_ALBUM_GAIN")

    if isinstance(audio, FLAC) or (hasattr(audio, "get") and not artist):
        try:
            artist = artist or _vc_first(audio, "artist")
            album = album or _vc_first(audio, "album")
            title = title or _vc_first(audio, "title")
            tn = _vc_first(audio, "tracknumber")
            if tn and not track_num:
                track_num = _parse_track_number(tn)
            rg_track = rg_track or _vc_first(audio, "replaygain_track_gain")
            rg_album = rg_album or _vc_first(audio, "replaygain_album_gain")
        except Exception:
            pass

    # MP4 / M4A
    if not artist and hasattr(audio, "tags") and audio.tags:
        try:
            tags = audio.tags
            if "\xa9ART" in tags:
                artist = str(tags["\xa9ART"][0])
            if "\xa9alb" in tags:
                album = str(tags["\xa9alb"][0])
            if "\xa9nam" in tags:
                title = str(tags["\xa9nam"][0])
            if "trkn" in tags:
                tr = tags["trkn"][0]
                track_num = int(tr[0]) if tr else 0
        except Exception:
            pass

    if isinstance(audio, WAVE) and not title:
        # WAVE rarely has tags; filename fallback below
        pass

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


def _id3_text(tags: Any, key: str) -> str | None:
    try:
        frame = tags.get(key) if hasattr(tags, "get") else None
        if frame and hasattr(frame, "text") and frame.text:
            return str(frame.text[0])
    except Exception:
        pass
    return None


def _id3_txxx(tags: Any, desc: str) -> str | None:
    try:
        values = tags.values() if hasattr(tags, "values") else []
        for frame in values:
            if (
                getattr(frame, "FrameID", None) == "TXXX"
                and getattr(frame, "desc", "").lower() == desc.lower()
            ):
                return str(frame.text[0])
    except Exception:
        pass
    return None


def _vc_first(audio: Any, key: str) -> str | None:
    try:
        vals = audio.get(key)
        if vals:
            return str(vals[0])
    except Exception:
        pass
    return None


def _parse_track_number(tn: str) -> int:
    try:
        return int(str(tn).split("/")[0].strip())
    except (ValueError, AttributeError):
        return 0


def _iter_audio_files(directory: Path) -> List[Path]:
    """Recursively find supported audio files under directory."""
    found: list[Path] = []
    try:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            # Skip hidden / excluded dirs in path
            parts_lower = {p.lower() for p in path.parts}
            if parts_lower & _SKIP_DIRS:
                continue
            if path.suffix.lower() in SUPPORTED_EXTS:
                found.append(path)
    except OSError as exc:
        logger.warning("Scan error under %s: %s", directory, exc)
    return found


def scan_directory(
    directory: Path | None = None,
    force: bool = False,
) -> List[TrackMeta]:
    """Scan directory (recursively) for audio files, returning sorted TrackMeta list.

    Uses a JSON cache keyed by filepath → mtime unless `force=True`.
    """
    directory = (directory or Path.cwd()).resolve()
    logger.info("Scanning %s for audio files...", directory)

    cache = load_cache()
    cache_map: Dict[str, Dict[str, Any]] = {
        c["path"]: c for c in cache if "path" in c
    }

    tracks: list[TrackMeta] = []
    for filepath in _iter_audio_files(directory):
        try:
            mtime = filepath.stat().st_mtime
        except OSError:
            continue

        path_str = str(filepath.resolve())
        cached = cache_map.get(path_str)
        if not force and cached and cached.get("_mtime") == mtime:
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

        logger.debug("Reading metadata: %s", filepath.name)
        meta = _extract_metadata(filepath)
        tracks.append(meta)

    tracks.sort(key=lambda t: (
        t.artist.lower(),
        t.album.lower(),
        t.track_num,
        t.title.lower(),
    ))

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
