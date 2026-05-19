"""Setup all datasets for complete workflow examples.

Copies datasets from all previous FASE examples.
"""
from __future__ import annotations

import shutil
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
DST = Path(__file__).parent

DATASETS = {
    "basic_forecasting/data/macro_brazil.csv": "macro_brazil.csv",
    "basic_forecasting/data/macro_us.csv": "macro_us.csv",
    "auto_forecast/data/m3_sample.csv": "m3_sample.csv",
    "auto_forecast/data/airline.csv": "airline.csv",
    "combination/data/inflation_forecasts.csv": "inflation_forecasts.csv",
    "combination/data/m4_sample.csv": "m4_sample.csv",
    "scenarios/data/us_macro_quarterly.csv": "us_macro_quarterly.csv",
    "nowcasting/data/gdp_vintages.csv": "gdp_vintages.csv",
    "nowcasting/data/mixed_freq.csv": "mixed_freq.csv",
}


def setup():
    """Copy all datasets to complete_workflow/data/."""
    for src_rel, dst_name in DATASETS.items():
        src = BASE / src_rel
        dst = DST / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {dst_name}")
        else:
            print(f"WARNING: {src} not found")


if __name__ == "__main__":
    setup()
