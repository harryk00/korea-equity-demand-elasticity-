#!/usr/bin/env python3
"""
Patch helper for True DTC collector:
- Prefer pykrxauth (current KRX-compatible fork)
- Fall back to pykrx if unavailable
"""
from pathlib import Path

target = Path("scripts/collect_true_short_balance_pykrx.py")
text = target.read_text(encoding="utf-8")

old = "from pykrx import stock"
new = """try:
    from pykrxauth import stock
    _SHORT_SOURCE = "pykrxauth"
except ImportError:
    from pykrx import stock
    _SHORT_SOURCE = "pykrx"
"""

if old not in text and "from pykrxauth import stock" not in text:
    raise SystemExit("Could not find pykrx import line to patch.")

if old in text:
    text = text.replace(old, new, 1)

target.write_text(text, encoding="utf-8")
print("patched ->", target)
