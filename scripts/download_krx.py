#!/usr/bin/env python3
"""Backward-compatible market downloader.

Despite the historical file name, v0.4.2 defaults to KIS because KRX Data
Marketplace now requires login and pykrx access is brittle. Pass
`--provider pykrx` explicitly if you have a working authenticated pykrx setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_market import main


if __name__ == "__main__":
    raise SystemExit(main())
