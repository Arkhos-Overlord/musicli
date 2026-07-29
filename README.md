# musicli

> A premium, retro CRT/VGA CLI music player with high-fidelity audio processing.

![Python](https://img.shields.io/badge/python-3.10+-blue)
[![MIT License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Arkhos--Overlord%2Fmusicli-181717?logo=github)](https://github.com/Arkhos-Overlord/musicli)

```text
┌──────────────────────────────────────────────────────────┐
│  musicli │ XF:ON │ SPACE:P/P ... Q:Quit                  │
├────────────────────────┬─────────────────────────────────┤
│                        │ NOW PLAYING                     │
│ ARTIST                 │ Track Title                     │
│   Album                │ Artist  |  Album                │
│    01. Track One       │                                 │
│    02. Track Two     ▶ │ 1:23 ████████░░░░░░░░░░ 3:45   │
│    03. Track Three     │                                 │
│                        │ VOL 80% ████████░░░░░░░░░░      │
│                        │ B [████─] +0dB  M [████─] +0dB │
│                        │                                 │
│                        │ ────────────────────────────────│
│                        │  past lyric text                │
│                        │ ▌ current lyric text            │
├────────────────────────┼─────────────────────────────────┤
│                        │ VISUALIZER [Blocks]             │
│                        │ ████▅▃▁▁▃▅████████▅▃▁▁▃▅████    │
│                        │                  ▲              │
└────────────────────────┴─────────────────────────────────┘
```

## Features

- 🎵 Scans current directory for `.mp3`, `.flac`, `.wav` files
- 📂 Auto-sorts hierarchically: Artist → Album → Track #
- 🎚️ **3-band parametric EQ** (Bass, Mid, Treble) with Biquad filters
- 📊 Real-time FFT visualizer + pre-calculated waveform display
- 🎨 Retro themes: CRT Amber, VGA Synthwave
- 📝 **Auto-fetched synced lyrics** via syncedlyrics (LRC)
- 🔊 ReplayGain normalization + TPDF dithering for clean output
- ⏭️ **Gapless playback** with configurable crossfade
- 💾 Persistent state (resume playback, remember volume/EQ/theme)
- 📋 Playlist queue management (add/reorder/clear)

## Quick Start

```bash
# Install
pip install musicli

# Launch in your music directory
cd ~/Music
musicli

# Force re-scan (ignore cache)
musicli --scan
```

### From Source

```bash
git clone https://github.com/Arkhos-Overlord/musicli.git
cd musicli
pip install -e .
musicli
```

Or run directly:

```bash
cd path/to/your/music
python -m musicli
```

## Controls

| Key | Action |
|-----|--------|
| `↑` `↓` `j` `k` | Navigate library |
| `Enter` | Play selected track |
| `Space` | Play / Pause |
| `n` / `p` | Next / Previous track |
| `←` `→` | Seek -10s / +10s |
| `+` / `-` | Volume up / down |
| `m` | Mute toggle |
| `1` / `2` / `3` | Boost Bass / Mid / Treble (+2 dB) |
| `Shift+1/2/3` | Cut Bass / Mid / Treble (-2 dB) |
| `r` | Cycle Repeat (Off / Track / Album) |
| `s` | Toggle Shuffle |
| `c` | Toggle Crossfade |
| `t` | Cycle Theme |
| `w` | Cycle Waveform style |
| `a` / `d` | Add / Remove track from Queue |
| `[` / `]` | Move queue item earlier / later |
| `x` | Clear entire Queue |
| `/` | Filter library (search artist/album/title) |
| `Esc` | Clear search |
| `q` | Quit (saves state) |

## Audio Pipeline

```
File → miniaudio (decode) → ReplayGain → 3-Band EQ → Volume → TPDF Dither → Speaker
                                                                          ↓
                                                                     FFT → Visualizer
```

- **Decoding**: miniaudio (C library) for fast .mp3/.flac/.wav decoding
- **ReplayGain**: Reads embedded RG tags or computes RMS fallback
- **EQ**: 3-band parametric (low-shelf @ 150 Hz, peaking @ 1 kHz, high-shelf @ 8 kHz) — RBJ Cookbook biquads via SciPy
- **Dithering**: TPDF (triangular probability density function) prevents quantization distortion
- **Sample rate**: 96 kHz internal processing for headroom

## Themes

| Theme | Preview |
|-------|---------|
| **CRT Amber** | Warm amber-on-black, classic terminal glow |
| **VGA Synthwave** | Cyan/magenta/blue synthwave aesthetic |

Press `t` to cycle between themes while playing.

## Project Structure

```
musicli/
├── __init__.py        # Package metadata
├── __main__.py        # `python -m musicli` entry
├── main.py            # Entry point + main loop
├── scanner.py         # Audio file discovery + metadata
├── audio_engine.py    # miniaudio playback + DSP + FFT
├── dsp.py             # Biquad EQ, ReplayGain, TPDF dithering
├── ui.py              # Rich-based retro TUI (Layout, Live, Panel)
├── themes.py          # CRT Amber / VGA Synthwave themes
├── waveform.py        # Pre-calculated waveform generation
├── lyrics.py          # syncedlyrics fetch + LRC parsing
├── state.py           # JSON state & cache persistence
├── pyproject.toml     # Package build config
└── README.md          # You are here
```

## Requirements

- Python 3.10+
- Audio output device
- Tested on Windows, Linux, macOS

## Development

```bash
pip install -e ".[dev]"
```

## License

MIT
