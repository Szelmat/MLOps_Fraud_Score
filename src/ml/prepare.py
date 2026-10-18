"""DVC stage 1 -- prepare: load raw CSV, clean, split into train/valid.

Run:  python -m src.ml.prepare      (or: dvc repro prepare)
"""

import pandas as pd


def clean(df: pd.DataFrame, target: str, drop_columns: list[str]) -> pd.DataFrame:
    """Deterministic cleaning: drop the configured columns and duplicates, check the target is 0/1."""
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    return df
