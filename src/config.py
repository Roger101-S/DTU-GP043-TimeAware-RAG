"""Central configuration. Values come from environment variables (.env) with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Fixed "today" for reproducible experiments. "Current status" queries are answered as of this date.
REFERENCE_DATE = datetime(2026, 1, 1)


@dataclass
class Config:
    embedder: str = "minilm"
    llm_provider: str = "none"
    arbitrator_mode: str = "rules"      # rules | hybrid | llm
    top_k: int = 5
    half_life_days: float = 365.0
    alpha: float = 0.6                  # weight of semantic similarity in the soft time-decay baseline
    date_order: str = "DMY"
    seed: int = 42
    reference_date: datetime = REFERENCE_DATE


def load_config() -> Config:
    """Read configuration from the environment at call time (so tests can override it)."""
    return Config(
        embedder=os.getenv("EMBEDDER", "minilm").strip().lower(),
        llm_provider=os.getenv("LLM_PROVIDER", "none").strip().lower(),
        arbitrator_mode=os.getenv("ARBITRATOR_MODE", "rules").strip().lower(),
        top_k=int(os.getenv("TOP_K", "5")),
        half_life_days=float(os.getenv("DECAY_HALF_LIFE_DAYS", "365")),
        alpha=float(os.getenv("DECAY_ALPHA", "0.6")),
        date_order=os.getenv("DATE_ORDER", "DMY").strip().upper(),
        seed=int(os.getenv("SEED", "42")),
    )
