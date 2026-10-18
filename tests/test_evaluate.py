import pytest

from src.ml import evaluate


@pytest.mark.parametrize(
    ("f1", "champion", "promote"),
    [(0.5, None, False), (0.7, None, True), (0.7, 0.8, False), (0.8, 0.8, True), (0.9, 0.8, True)],
)
def test_should_promote(f1: float, champion: float | None, promote: bool):
    assert evaluate.should_promote(f1, champion, min_f1=0.6) is promote
