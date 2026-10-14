import os
from dataclasses import dataclass, field

DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read from environment variables."""

    fraud_threshold: float = field(
        default_factory=lambda: float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _split(os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS))
    )
