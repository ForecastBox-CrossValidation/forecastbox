"""Setup datasets for pipeline examples.

Copies macro_brazil.csv from basic_forecasting examples.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent.parent / "basic_forecasting" / "data"
DST_DIR = Path(__file__).parent


def setup():
    """Copy datasets for pipeline examples."""
    for fname in ["macro_brazil.csv"]:
        src = SRC_DIR / fname
        dst = DST_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fname}")
        else:
            print(f"WARNING: {src} not found. Run FASE1 setup first.")


if __name__ == "__main__":
    setup()
