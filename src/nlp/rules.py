"""Keyword weak labeler for complaint categories (baseline and weak-supervision source).

Simple negation handling: a keyword is ignored if "no", "not", "never", "without" or
"zero" appears within the two preceding tokens ("no crashes so far").
"""

from __future__ import annotations

import re

LEXICON: dict[str, list[str]] = {
    "bugs": ["bug", "bugs", "bugged", "crash", "crashes", "crashed", "crashing", "glitch", "glitches", "corrupt",
             "corrupted", "freeze", "froze", "broken save", "error", "fails to load", "falling through"],
    "performance": ["fps", "frame rate", "framerate", "stutter", "stuttering", "optimization", "optimised", "optimized",
                    "lag spikes", "loading times", "memory leak", "cpu", "gpu", "slideshow", "choppy", "frames"],
    "servers": ["server", "servers", "disconnect", "disconnected", "ping", "rubber-banding", "connection", "desync",
                "offline", "netcode", "latency", "timed out", "log in"],
    "balance": ["balance", "balanced", "nerf", "nerfed", "buff", "overpowered", "underpowered", "op ", "meta",
                "unfair", "one-shot", "difficulty spikes", "scaling", "tuned"],
    "monetization": ["microtransaction", "microtransactions", "pay to win", "pay-to-win", "loot box", "loot boxes",
                     "battle pass", "premium", "packs", "store", "cash grab", "currency", "real money", "greedy"],
    "matchmaking": ["matchmaking", "queue", "queue times", "matched", "rank", "ranked", "lobby", "sbmm", "opponents"],
    "ux": ["menu", "menus", "interface", "ui", "hud", "controller support", "keybinds", "tutorial", "subtitles",
           "accessibility", "navigate", "layout", "settings screen"],
    "content": ["content", "endgame", "repetitive", "replay value", "roadmap", "nothing new", "maps", "missions",
                "abandoned", "nothing to do"],
    "support": ["support", "ticket", "refund", "customer service", "help desk", "banned", "account recovery",
                "no reply", "responds"],
}
NEGATORS = {"no", "not", "never", "without", "zero", "nothing"}
_TOKEN = re.compile(r"[a-z0-9'-]+")


def _negated(tokens: list[str], start: int) -> bool:
    return any(t in NEGATORS for t in tokens[max(0, start - 2):start])


def label(text: str) -> dict[str, int]:
    t = " " + text.lower() + " "
    tokens = _TOKEN.findall(t)
    out = {}
    for cat, words in LEXICON.items():
        hit = 0
        for w in words:
            w_tokens = _TOKEN.findall(w)
            for i in range(len(tokens) - len(w_tokens) + 1):
                if tokens[i:i + len(w_tokens)] == w_tokens and not _negated(tokens, i):
                    hit = 1
                    break
            if hit:
                break
        out[cat] = hit
    return out


def label_many(texts) -> list[dict[str, int]]:
    return [label(t) for t in texts]
