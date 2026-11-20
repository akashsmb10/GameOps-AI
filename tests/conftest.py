"""Builds a small, isolated pipeline once per test session (3,000 synthetic reviews in a temp dir)."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session", autouse=True)
def built(tmp_path_factory):
    base = tmp_path_factory.mktemp("gameops")
    for k, sub in {"GAMEOPS_RAW_DIR": "raw", "GAMEOPS_PROCESSED_DIR": "processed", "GAMEOPS_ARTIFACTS_DIR": "artifacts"}.items():
        os.environ[k] = str(base / sub)
    os.environ["GAMEOPS_DATASET"] = "demo"
    os.environ.pop("OLLAMA_URL", None)
    from src.config import get_settings
    from src.ingestion.demo_generator import write
    write(get_settings().raw_dir, n=3000, seed=11)
    import build_pipeline
    manifest = build_pipeline.main()
    return {"base": base, "manifest": manifest}
