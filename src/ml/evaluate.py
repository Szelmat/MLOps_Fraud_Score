"""DVC stage 3 -- evaluate: score the new model and promote it only if it is not worse.

This is the automatic guard against model quality dropping. The new version gets the
registry's "champion" alias -- the one the API serves -- only if its F1 on the validation
split clears `min_f1` AND is not below the current champion's F1 on the same split.

Writes metrics/metrics.json, declared as `metrics` in dvc.yaml with `cache: false`,
so it goes into Git and `dvc metrics diff` can compare commits.

Run:  python -m src.ml.evaluate      (or: dvc repro evaluate)
"""


def should_promote(f1: float, champion_f1: float | None, min_f1: float) -> bool:
    """The promotion rule: above the floor, and not worse than the champion (if there is one)."""
    return f1 >= min_f1 and (champion_f1 is None or f1 >= champion_f1)
