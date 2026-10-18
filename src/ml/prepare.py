"""DVC stage 1 -- prepare: load raw CSV, clean, split into train/valid.

Run:  python -m src.ml.prepare      (or: dvc repro prepare)
"""

from typing import Any, cast

import pandas as pd
from sklearn.model_selection import train_test_split

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


def main() -> None:
    params: dict[str, Any] = config.load_params()
    prepare_params = params["prepare"]
    target = prepare_params["target"]

    log.info("Reading %s", config.RAW_CSV)
    df = clean(pd.read_csv(config.RAW_CSV), target=target, drop_columns=prepare_params["drop_columns"])
    log.info("Clean shape: %s", df.shape)

    stratify = df[target] if prepare_params["stratify"] else None
    # train_test_split returns a loosely typed list; with DataFrame input both parts are DataFrames.
    train_df, valid_df = cast(
        tuple[pd.DataFrame, pd.DataFrame],
        train_test_split(df, test_size=prepare_params["test_size"], random_state=params["seed"], stratify=stratify),
    )

    config.ensure_dirs(config.PROCESSED_DIR)
    train_df.to_csv(config.TRAIN_CSV, index=False)
    valid_df.to_csv(config.VALID_CSV, index=False)
    log.info("train=%d valid=%d", len(train_df), len(valid_df))
