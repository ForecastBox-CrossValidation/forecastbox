"""Setup datasets for evaluation examples.

Copies/links datasets from combination examples or generates fresh copies.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent.parent / "combination" / "data"
DST_DIR = Path(__file__).parent


def setup():
    """Copy datasets from combination to evaluation."""
    for fname in ["inflation_forecasts.csv", "m4_sample.csv"]:
        src = SRC_DIR / fname
        dst = DST_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {fname}")
        else:
            print(f"WARNING: {src} not found. Run FASE3 setup first.")


if __name__ == "__main__":
    setup()
