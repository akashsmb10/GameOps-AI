"""Lightweight monitoring: request, retrieval and agent telemetry stored in SQLite.

This is OFFLINE DEMONSTRATION TELEMETRY - it records whatever traffic hits this local
instance (manual use, tests, load tests). It is not production monitoring of real users.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import get_settings

_LOCK = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (ts REAL, method TEXT, path TEXT, status INTEGER, latency_ms REAL);
CREATE TABLE IF NOT EXISTS retrievals (ts REAL, latency_ms REAL, best_score REAL, answerable INTEGER);
CREATE TABLE IF NOT EXISTS agent_runs (ts REAL, mode TEXT, refused TEXT, tool_calls INTEGER, tool_errors INTEGER,
                                       citations INTEGER, invalid_citations INTEGER, latency_ms REAL);
"""


def _db() -> Path:
    p = get_settings().artifacts_dir / "logs" / "telemetry.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _exec(sql: str, params: tuple) -> None:
    with _LOCK, sqlite3.connect(_db()) as con:
        con.executescript(SCHEMA)
        con.execute(sql, params)


def record_request(method: str, path: str, status: int, latency_ms: float) -> None:
    _exec("INSERT INTO requests VALUES (?,?,?,?,?)", (time.time(), method, path, status, latency_ms))


def record_retrieval(latency_ms: float, best_score: float, answerable: bool) -> None:
    _exec("INSERT INTO retrievals VALUES (?,?,?,?)", (time.time(), latency_ms, best_score, int(answerable)))


def record_agent(out: dict) -> None:
    calls = out.get("tool_calls", [])
    _exec("INSERT INTO agent_runs VALUES (?,?,?,?,?,?,?,?)",
          (time.time(), out.get("mode"), (out.get("refused") or {}).get("category"), len(calls),
           sum(c["status"] != "ok" for c in calls), len(out.get("citations", [])), len(out.get("invalid_citations", [])),
           out.get("latency_ms", 0.0)))


def _pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 2) if values else None


def _read_json(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def summary(window_hours: float = 24 * 7) -> dict:
    s = get_settings()
    since = time.time() - window_hours * 3600
    with _LOCK, sqlite3.connect(_db()) as con:
        con.executescript(SCHEMA)
        req = con.execute("SELECT path, status, latency_ms FROM requests WHERE ts >= ?", (since,)).fetchall()
        ret = con.execute("SELECT latency_ms, answerable FROM retrievals WHERE ts >= ?", (since,)).fetchall()
        runs = con.execute("SELECT mode, refused, tool_calls, tool_errors, citations, invalid_citations, latency_ms "
                           "FROM agent_runs WHERE ts >= ?", (since,)).fetchall()
    by_path: dict[str, list] = {}
    for path, status, lat in req:
        by_path.setdefault(path, []).append((status, lat))
    endpoints = {p: {"requests": len(v), "error_rate": round(sum(st >= 500 for st, _ in v) / len(v), 4),
                     "client_error_rate": round(sum(400 <= st < 500 for st, _ in v) / len(v), 4),
                     "p50_ms": _pct([lat for _, lat in v], 50), "p95_ms": _pct([lat for _, lat in v], 95)}
                 for p, v in sorted(by_path.items())}
    tool_calls = sum(r[2] for r in runs)
    tool_errors = sum(r[3] for r in runs)
    cites, bad = sum(r[4] for r in runs), sum(r[5] for r in runs)
    manifest = _read_json(s.artifacts_dir / "pipeline_manifest.json") or {}
    idx = _read_json(s.artifacts_dir / "index" / "reviews_manifest.json") or {}
    dq = _read_json(s.processed_dir / "data_quality_report.json") or {}
    built = idx.get("built_at")
    age_h = round((datetime.now(timezone.utc) - datetime.fromisoformat(built)).total_seconds() / 3600, 2) if built else None
    return {
        "label": "offline demonstration telemetry (local traffic only, not production monitoring)",
        "window_hours": window_hours,
        "api": {"requests": len(req), "error_rate": round(sum(st >= 500 for _, st, _ in req) / len(req), 4) if req else None,
                "p50_ms": _pct([x[2] for x in req], 50), "p95_ms": _pct([x[2] for x in req], 95), "endpoints": endpoints},
        "retrieval": {"queries": len(ret), "p50_ms": _pct([x[0] for x in ret], 50), "p95_ms": _pct([x[0] for x in ret], 95),
                      "answerable_rate": round(sum(x[1] for x in ret) / len(ret), 4) if ret else None},
        "agent": {"runs": len(runs), "refused": sum(1 for r in runs if r[1]), "tool_calls": tool_calls,
                  "tool_success_rate": round(1 - tool_errors / tool_calls, 4) if tool_calls else None, "failed_tool_calls": tool_errors,
                  "citation_validity": round(cites / (cites + bad), 4) if (cites + bad) else None,
                  "p50_ms": _pct([r[6] for r in runs], 50), "p95_ms": _pct([r[6] for r in runs], 95)},
        "data": {"dataset": manifest.get("dataset"), "dataset_refreshed_at": dq.get("generated_at"),
                 "included_reviews": manifest.get("included_reviews")},
        "models": {"complaint_model": manifest.get("complaint_model"), "embedder": idx.get("embedder"),
                   "index_built_at": built, "index_age_hours": age_h, "retrieval_mode": manifest.get("retrieval_mode")},
        "evaluations": {"rag": _read_json(s.artifacts_dir / "eval" / "rag_eval.json"),
                        "agent": {k: v for k, v in (_read_json(s.artifacts_dir / "eval" / "agent_eval.json") or {}).items() if k != "failed"}},
    }
