"""Central configuration (paths relative to the repository root; overridable via environment)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = ["bugs", "performance", "servers", "balance", "monetization",
              "matchmaking", "ux", "content", "support", "other"]
CATEGORY_LABELS = {
    "bugs": "Bugs & crashes", "performance": "Performance & optimization", "servers": "Servers & connectivity",
    "balance": "Gameplay balance", "monetization": "Monetization & microtransactions", "matchmaking": "Matchmaking",
    "ux": "User experience & UI", "content": "Content & updates", "support": "Customer support", "other": "Other",
}


def _p(env: str, default: Path) -> Path:
    v = os.getenv(env)
    return Path(v) if v else default


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = field(default_factory=lambda: _p("GAMEOPS_RAW_DIR", ROOT / "data" / "raw"))
    processed_dir: Path = field(default_factory=lambda: _p("GAMEOPS_PROCESSED_DIR", ROOT / "data" / "processed"))
    artifacts_dir: Path = field(default_factory=lambda: _p("GAMEOPS_ARTIFACTS_DIR", ROOT / "artifacts"))
    knowledge_dir: Path = ROOT / "docs" / "knowledge"
    dataset: str = field(default_factory=lambda: os.getenv("GAMEOPS_DATASET", "demo"))  # demo | steam
    embedder: str = field(default_factory=lambda: os.getenv("GAMEOPS_EMBEDDER", "auto"))  # auto | sentence-transformers | spacy | tfidf
    st_model: str = field(default_factory=lambda: os.getenv("GAMEOPS_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    ollama_url: str | None = field(default_factory=lambda: os.getenv("OLLAMA_URL") or None)
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
    agent_max_tool_calls: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_TOOL_CALLS", "5")))
    tool_timeout_s: float = field(default_factory=lambda: float(os.getenv("TOOL_TIMEOUT_S", "10")))


def get_settings() -> Settings:
    return Settings()
