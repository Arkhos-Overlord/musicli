"""Lyrics fetching (syncedlyrics) and LRC timestamp parsing."""

from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_LRC_LINE_RE = re.compile(r"(\[\d{2}:\d{2}\.\d{2,3}\])+(.*)")


def parse_lrc(lrc_text: str) -> Dict[float, str]:
    """Parse an LRC-format string into {time_seconds: lyric_line}."""
    lyrics: Dict[float, str] = {}

    for line in lrc_text.splitlines():
        line = line.strip()
        match = _LRC_LINE_RE.match(line)
        if not match:
            continue

        text = match.group(2).strip()
        if not text:
            continue

        ts_re = re.compile(r"\[(\d{2}):(\d{2})\.(\d{2,3})\]")
        for ts_match in ts_re.finditer(line):
            minutes = int(ts_match.group(1))
            seconds = int(ts_match.group(2))
            frac = ts_match.group(3)
            frac_seconds = float(f"0.{frac}")
            total_sec = minutes * 60 + seconds + frac_seconds
            lyrics[total_sec] = text

    return lyrics


def fetch_lyrics_async(
    artist: str,
    title: str,
    callback: Callable[[Dict[float, str]], None],
    generation: int = 0,
    generation_check: Optional[Callable[[], int]] = None,
) -> None:
    """Fetch synced lyrics in a background thread.

    If `generation_check` is provided, the callback is only invoked when
    `generation_check() == generation` (drops stale results).
    """

    def _worker() -> None:
        try:
            query = f"{title} {artist}".strip()
            if not query or query == "Unknown Title Unknown Artist":
                _safe_callback({})
                return
            logger.info("Fetching lyrics: %s", query)
            lrc = _safe_search(query)
            if lrc:
                parsed = parse_lrc(lrc)
                logger.info("Got %d lyric lines.", len(parsed))
                _safe_callback(parsed)
            else:
                _safe_callback({})
        except Exception as exc:
            logger.error("Lyrics fetch failed: %s", exc)
            _safe_callback({})

    def _safe_callback(result: Dict[float, str]) -> None:
        if generation_check is not None and generation_check() != generation:
            return
        try:
            callback(result)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


def _safe_search(query: str) -> Optional[str]:
    """Call syncedlyrics.search() with error handling."""
    try:
        import syncedlyrics
        result = syncedlyrics.search(query)
        if result and isinstance(result, str):
            return result
    except Exception:
        logger.debug("syncedlyrics.search failed", exc_info=True)
    return None
