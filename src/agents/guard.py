"""Input validation, prompt-injection detection and scope checking for the agent.

These checks run BEFORE any tool or model call. They are heuristic (pattern based) and are a
first line of defence, not a guarantee; the real protection is architectural: tools are
read-only, parameterised and whitelisted, so even a successful injection cannot run SQL,
code or write data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 1000
INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (your|all|the) (instructions|rules|guidelines)",
    r"(reveal|print|show|leak) (me )?(your|the) (system|hidden|internal) (prompt|instructions)",
    r"you are now\b", r"act as (?!a game)", r"\bjailbreak\b", r"developer mode",
    r"\b(drop|delete|truncate|alter|insert|update)\s+(table|from|into)\b",
    r"\bselect\s+.+\s+from\s+\w+", r";\s*--", r"\bexec(ute)?\s*\(", r"import\s+os|subprocess|__import__",
    r"(api[_ ]?key|password|credentials|secret token)",
]
DOMAIN_WORDS = {
    "review", "reviews", "player", "players", "game", "games", "complaint", "complaints", "sentiment", "negative",
    "positive", "feedback", "update", "patch", "bug", "bugs", "crash", "crashes", "server", "servers", "performance",
    "monetization", "microtransactions", "matchmaking", "balance", "support", "content", "issue", "issues", "trend",
    "trends", "rating", "ratings", "fps", "lag", "metric", "metrics", "dataset",
    "quality", "category", "categories", "investigate", "developers", "unhappy", "complaining", "players'", "version",
    "release", "notes", "summarize", "summarise", "compare", "percentage", "share", "emerging", "excluded", "model",
}


@dataclass
class GuardResult:
    allowed: bool
    reason: str | None = None
    category: str | None = None  # invalid_input | prompt_injection | out_of_scope


def check(question: str, game_names: list[str]) -> GuardResult:
    if not isinstance(question, str) or not question.strip():
        return GuardResult(False, "The question is empty.", "invalid_input")
    if len(question) > MAX_CHARS:
        return GuardResult(False, f"The question is longer than {MAX_CHARS} characters.", "invalid_input")
    q = question.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, q):
            return GuardResult(False, "The request contains instructions that try to change my rules or access data "
                                      "directly. I only answer questions about player feedback using approved read-only tools.",
                               "prompt_injection")
    tokens = set(re.findall(r"[a-z']+", q))
    mentions_game = any(name.lower() in q or any(w in q for w in name.lower().split() if len(w) >= 5) for name in game_names)
    if not (tokens & DOMAIN_WORDS) and not mentions_game:
        return GuardResult(False, "That question is outside what I can answer. I analyse player reviews, complaints, "
                                  "sentiment trends and update notes for the games in this dataset.", "out_of_scope")
    return GuardResult(True)
