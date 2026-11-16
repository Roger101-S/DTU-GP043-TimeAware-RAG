"""LLM access through LangChain chat models, plus token accounting and robust JSON parsing.

Providers: groq | gemini | ollama | none.  `none` means no LLM: the pipeline runs fully offline using
the rule-based reader/arbitrator. LLM code paths are covered by unit tests with a fake LLM, but you must
test them yourself with a real API key / local model.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token). Used when the provider returns no usage data."""
    return max(1, round(len(text) / 4))


@dataclass
class TokenMeter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_role: dict = field(default_factory=dict)     # role -> total tokens

    def add(self, role: str, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        self.by_role[role] = self.by_role.get(role, 0) + prompt + completion

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def make_llm(provider: str):
    """Build a LangChain chat model, or return None for provider 'none'."""
    p = (provider or "none").lower()
    if p in ("none", ""):
        return None
    if p == "groq":
        from langchain_groq import ChatGroq            # pip install langchain-groq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), temperature=0)
    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI   # pip install langchain-google-genai
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), temperature=0)
    if p == "ollama":
        from langchain_ollama import ChatOllama        # pip install langchain-ollama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                          base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0)
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use none | groq | gemini | ollama.")


def call_llm(llm, prompt: str, meter: Optional[TokenMeter], role: str, retries: int = 3) -> str:
    """Invoke the model with simple exponential back-off (free tiers rate-limit)."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", resp)
            if isinstance(text, list):                    # some providers return content blocks
                text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
            text = str(text)
            usage = getattr(resp, "usage_metadata", None) or {}
            p_tok = usage.get("input_tokens") or approx_tokens(prompt)
            c_tok = usage.get("output_tokens") or approx_tokens(text)
            if meter is not None:
                meter.add(role, int(p_tok), int(c_tok))
            return text
        except Exception as exc:                          # network / rate-limit errors
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_exc}")


def extract_json(text: str):
    """Parse the first JSON object in a model reply (tolerates ```json fences and extra prose)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
