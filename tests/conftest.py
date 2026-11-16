import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("EMBEDDER", "tfidf")

from src.config import load_config  # noqa: E402
from src.runner import build_system  # noqa: E402
from src.schema import Note  # noqa: E402

NOTES = [
    Note("A-1", "PA", "Clinic visit on 10 March 2020. No history of hypertension. BP 118/76.", "2020-03-12"),
    Note("A-2", "PA", "Visit date: 2022-06-15. Pre-hypertension noted, BP 132/86.", "2022-06-20"),
    Note("A-3", "PA", "Seen on January 20, 2024. Stage 2 hypertension, BP 168/104, started on amlodipine 5 mg.", "2024-01-25"),
    Note("A-4", "PA", "Clinic visit on 5 May 2023. Routine visit; vaccinations up to date.", "2023-05-06"),
    Note("B-1", "PB", "Clinic visit on 2 February 2021. Seen for seasonal cold.", "2021-02-03"),
]


@pytest.fixture(scope="session")
def system():
    cfg = load_config()
    cfg.embedder, cfg.arbitrator_mode, cfg.top_k = "tfidf", "rules", 5
    cfg.reference_date = datetime(2026, 1, 1)
    pipelines, meter, report = build_system(cfg, NOTES, llm=None)
    return pipelines, meter, report, cfg


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Stands in for a LangChain chat model. `reply` is a str or callable(prompt)->str."""
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse(self.reply(prompt) if callable(self.reply) else self.reply)
