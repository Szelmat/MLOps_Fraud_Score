"""DVC stage 1 -- prepare: load raw CSV, clean, split into train/valid.

Run:  python -m src.ml.prepare      (or: dvc repro prepare)
"""

import pandas as pd

from src.ml import config

log = config.get_logger(__name__)


def clean(df: pd.DataFrame, target: str, drop_columns: list[str]) -> pd.DataFrame:
    """Deterministic cleaning: drop the configured columns and duplicates, check the target is 0/1."""
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    if not set(df[target].unique()) <= {0, 1}:
        raise ValueError(f"Target column '{target}' must hold only 0 and 1")

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != before:
        log.warning("Dropped %d duplicate rows", before - len(df))
    return df
