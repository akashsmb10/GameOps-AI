"""SYNTHETIC demo review generator (clearly labelled; never real player feedback).

Why it exists: this build environment cannot reach Steam, Kaggle or Hugging Face, and Steam
review text is user content that should not be redistributed in a public repository.
The demo set lets the whole platform run offline. Use `src/ingestion/steam.py` to collect
real reviews locally (kept out of git).

Design choices that keep the evaluation honest:
* The five games, their patch notes and every review are fictional.
* Each review records the complaint categories the generator injected
  (`labels_synthetic.parquet`). They are ground truth ONLY for this synthetic set.
* Phrase templates are split into a training pool and a held-out pool; the held-out pool is
  split again into disjoint validation and test phrasings. Model selection therefore sees
  unseen wording, and test scores measure generalisation to wording seen by neither
  training nor selection (still an optimistic proxy for real reviews).
* Negation traps ("no crashes so far") and mixed reviews make keyword rules imperfect.
* Patches change complaint rates in designed ways, so the update-impact analysis has real
  structure to find - the effect exists only because the generator put it there.
* Data-quality defects (duplicates, blank text, bad timestamps, non-English text,
  spam, impossible playtime) are injected so the cleaning pipeline has work to do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORIES, get_settings

GAMES = [
    {"game_id": "G001", "game_name": "Starfall Arena", "genre": "competitive shooter", "weight": 0.30},
    {"game_id": "G002", "game_name": "Kingdoms of Ember", "genre": "fantasy RPG", "weight": 0.22},
    {"game_id": "G003", "game_name": "Turbo Circuit 26", "genre": "racing", "weight": 0.18},
    {"game_id": "G004", "game_name": "Pitch Legends", "genre": "football sim", "weight": 0.20},
    {"game_id": "G005", "game_name": "Hollow Depths", "genre": "survival horror", "weight": 0.10},
]
START, END = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-10-31")

# Fictional patches. "effects" multiply the base rate of a complaint category for "days".
PATCHES = [
    {"game_id": "G001", "version": "2.1", "date": "2026-07-08", "days": 21, "effects": {"servers": 3.0, "matchmaking": 1.8},
     "notes": ["Migrated matchmaking to new regional data centres", "Added ranked season 3", "Reworked anti-cheat client"]},
    {"game_id": "G001", "version": "2.2", "date": "2026-09-15", "days": 14, "effects": {"balance": 2.5},
     "notes": ["Rebalanced assault rifles and sniper damage", "Fixed server desync on Harbor map"]},
    {"game_id": "G002", "version": "1.4", "date": "2026-06-20", "days": 28, "effects": {"monetization": 3.2},
     "notes": ["Introduced the Ember Store with cosmetic bundles", "Added premium battle pass", "New questline: The Ash Crown"]},
    {"game_id": "G002", "version": "1.5", "date": "2026-09-01", "days": 14, "effects": {"bugs": 2.2, "content": 0.6},
     "notes": ["New region: Frostmere", "Save-system rewrite", "Quest log redesign"]},
    {"game_id": "G003", "version": "3.0", "date": "2026-08-05", "days": 21, "effects": {"performance": 2.8},
     "notes": ["New graphics engine with ray-traced reflections", "Added 6 tracks", "Updated minimum PC requirements"]},
    {"game_id": "G004", "version": "26.2", "date": "2026-07-25", "days": 21, "effects": {"monetization": 2.0, "balance": 1.6},
     "notes": ["Ultimate Squad packs refreshed", "Player ratings update", "Passing physics tuned"]},
    {"game_id": "G005", "version": "1.1", "date": "2026-06-10", "days": 14, "effects": {"bugs": 2.0},
     "notes": ["Added co-op mode", "Fixed lighting in Chapter 2"]},
]

BASE_RATE = {"bugs": 0.20, "performance": 0.14, "servers": 0.10, "balance": 0.10, "monetization": 0.10,
             "matchmaking": 0.07, "ux": 0.08, "content": 0.09, "support": 0.04}
GAME_TILT = {"G001": {"servers": 1.6, "matchmaking": 1.8, "balance": 1.4}, "G002": {"bugs": 1.3, "content": 1.2},
             "G003": {"performance": 1.5}, "G004": {"monetization": 1.8, "balance": 1.2}, "G005": {"bugs": 1.4, "performance": 1.3}}

# phrase pools: (train pool, held-out pool). {g} = game name.
COMPLAINTS: dict[str, tuple[list[str], list[str]]] = {
    "bugs": ([
        "The game crashes to desktop every time I open the inventory.",
        "Constant bugs - quests fail to trigger and NPCs get stuck in walls.",
        "My save file got corrupted after two hours of progress.",
        "Crashed three times in one evening, lost progress every time.",
        "There is a game-breaking bug where the final mission never loads.",
        "Textures fail to load and I keep falling through the map.",
        "Achievements are bugged and do not unlock.",
        "Random crashes with an unhandled exception error.",
    ], [
        "It froze completely and I had to hard reset my PC.",
        "Glitches everywhere, the checkpoint system is broken.",
        "Keeps closing itself without any error message.",
        "Progress resets after every restart, clearly a broken save system.",
    ]),
    "performance": ([
        "Terrible optimization, the frame rate drops below 30 fps on a high-end GPU.",
        "Massive stuttering whenever there are more than ten players on screen.",
        "Loading times are painfully long even on an SSD.",
        "The game is unplayable on medium settings, constant fps drops.",
        "CPU usage hits 100 percent and the fans scream.",
        "Shader compilation stutter makes the first hour miserable.",
        "Performance tanked after the patch, frame pacing is awful.",
        "Huge memory leak, RAM usage keeps climbing until it lags.",
    ], [
        "Runs like a slideshow on my laptop despite meeting the specs.",
        "Choppy and sluggish, it barely holds 40 frames.",
        "My GPU is at full load for a menu screen, poorly optimised.",
        "Micro-stutters every few seconds ruin the experience.",
    ]),
    "servers": ([
        "Servers are down every evening during peak hours.",
        "I keep getting disconnected from the server mid-match.",
        "Huge ping spikes and rubber-banding make online play impossible.",
        "Connection lost errors every ten minutes.",
        "Cannot even log in, stuck on connecting to servers.",
        "Desync is so bad that shots do not register.",
        "Server lag makes the multiplayer unplayable.",
        "Online services go offline without any warning.",
    ], [
        "Network issues kicked me out of three games in a row.",
        "Latency is through the roof on the new data centres.",
        "Always online requirement plus unstable servers is a disaster.",
        "Timed out of the lobby again, the netcode is a mess.",
    ]),
    "balance": ([
        "Weapon balance is a joke, one rifle dominates every match.",
        "The latest nerf made my favourite class useless.",
        "Snipers are completely overpowered right now.",
        "Difficulty spikes are unfair and poorly tuned.",
        "Pay attention to balance, some characters are simply broken.",
        "The meta is stale because one strategy beats everything.",
        "Ratings make no sense, low rated players outperform stars.",
        "Enemies one-shot you while taking forever to kill.",
    ], [
        "Unfair matchups because certain loadouts are absurdly strong.",
        "Everything except two builds feels underpowered after the update.",
        "The boss scaling is ridiculous and badly tuned.",
        "Damage numbers feel random and badly balanced.",
    ]),
    "monetization": ([
        "Way too many microtransactions for a full price game.",
        "The new store is pure pay to win.",
        "Loot boxes and packs are predatory.",
        "Everything good is locked behind a premium battle pass.",
        "Cosmetics cost more than the game itself.",
        "Constant pop-ups pushing me to buy currency.",
        "Grind is designed to sell boosters, greedy monetization.",
        "Paying extra for content that should be in the base game is shameful.",
    ], [
        "It feels like a cash grab with a game attached.",
        "Spent money on packs and got nothing, the odds are rigged.",
        "Real money shortcuts everywhere ruin the progression.",
        "The premium currency prices are outrageous.",
    ]),
    "matchmaking": ([
        "Matchmaking takes ten minutes to find a game.",
        "I get matched with players far above my rank every time.",
        "Ranked matchmaking is broken, teams are always unbalanced.",
        "Queue times are insane outside peak hours.",
        "Skill based matchmaking makes every casual game a sweat fest.",
        "Party matchmaking puts me against full premade squads.",
    ], [
        "Waiting forever in the lobby just to be placed against pros.",
        "The rank system pairs bronze players with diamond teams.",
        "Searching for opponents takes longer than the match itself.",
    ]),
    "ux": ([
        "The menus are confusing and cluttered.",
        "The user interface is clunky and hard to navigate.",
        "Controller support is terrible and keybinds cannot be changed.",
        "The tutorial explains nothing and the HUD is unreadable.",
        "Too many clicks to do anything simple in the menu.",
        "Subtitles are tiny and there are no accessibility options.",
    ], [
        "Navigating the settings screen is a nightmare.",
        "Text is impossible to read on a TV, terrible layout.",
        "The inventory screen is a mess with no sorting options.",
    ]),
    "content": ([
        "There is not enough content, I finished everything in eight hours.",
        "The new update added almost nothing to do.",
        "Endgame content is missing entirely.",
        "Same three maps for months, it gets repetitive quickly.",
        "The roadmap promised new modes that never arrived.",
        "Very little replay value once the story ends.",
    ], [
        "Feels empty, barely any missions after the campaign.",
        "Nothing new in months, the developers abandoned it.",
        "Repetitive missions and recycled environments.",
    ]),
    "support": ([
        "Customer support never answered my ticket.",
        "Support closed my refund request without reading it.",
        "I was banned by mistake and support ignores me.",
        "Waited three weeks for a reply from customer service.",
        "The help desk keeps sending copy paste answers.",
    ], [
        "Nobody from the support team responds to emails.",
        "Contacted the developers about my lost purchase and got no help.",
        "Account recovery through support was useless.",
    ]),
}
PRAISE = ([
    "The core gameplay loop is genuinely fun.", "Great art direction and soundtrack.",
    "Playing with friends is a blast.", "The story kept me hooked.", "Controls feel tight and responsive.",
    "Lots of fun once you get into it.", "Beautiful visuals and a great atmosphere.",
    "Worth the price, highly recommended.", "The developers listen to feedback.",
], [
    "An absolute gem, I lost a weekend to it.", "Solid fun with a great community.",
    "Excellent pacing and memorable characters.", "One of the best in the genre.",
])
NEGATION_TRAPS = ["No crashes so far, runs fine.", "Servers have been stable for me.", "No pay to win as far as I can tell.",
                  "Matchmaking is quick in my region.", "Performance is smooth on my machine.", "Did not need support at all."]
GENERIC_NEG = (["Not worth the money.", "I regret buying this.", "Just not fun anymore.", "Disappointing overall."],
               ["Cannot recommend it in this state.", "Refunded after an hour."])
FOREIGN = [("spanish", "El juego se cierra constantemente, muy decepcionante."), ("german", "Die Server sind ständig offline."),
           ("french", "Trop de microtransactions, je ne recommande pas."), ("portuguese", "Jogo muito bom, recomendo.")]


def _typo(text: str, rng: np.random.Generator, p: float = 0.03) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _pool(pools: tuple[list[str], list[str]], split: str) -> list[str]:
    """train -> train phrasings; validation / test -> two DISJOINT halves of the held-out phrasings."""
    if split == "train":
        return pools[0]
    held = pools[1]
    half = len(held) // 2
    return held[:half] if split == "validation" else held[half:]


def _effect(game_id: str, day: pd.Timestamp) -> dict[str, float]:
    mult: dict[str, float] = {}
    for p in PATCHES:
        d = pd.Timestamp(p["date"])
        if p["game_id"] == game_id and d <= day < d + pd.Timedelta(days=p["days"]):
            for k, v in p["effects"].items():
                mult[k] = mult.get(k, 1.0) * v
    return mult


def generate(n: int = 12_000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    games = pd.DataFrame(GAMES)
    days = pd.date_range(START, END, freq="D")
    rows, labels = [], []
    game_idx = rng.choice(len(games), n, p=games["weight"] / games["weight"].sum())
    # review volume rises around patches (people review after updates)
    day_w = np.ones(len(days))
    for k in range(n):
        g = games.iloc[game_idx[k]]
        w = day_w.copy()
        for p in PATCHES:
            if p["game_id"] == g.game_id:
                d0 = (pd.Timestamp(p["date"]) - START).days
                w[max(0, d0):min(len(days), d0 + 10)] *= 2.0
        day = days[rng.choice(len(days), p=w / w.sum())]
        u = rng.random()
        split = "test" if u < 0.2 else ("validation" if u < 0.3 else "train")
        mult = _effect(g.game_id, day)
        tilt = GAME_TILT.get(g.game_id, {})
        cats = [c for c, r in BASE_RATE.items() if rng.random() < min(0.9, r * tilt.get(c, 1.0) * mult.get(c, 1.0))]
        n_cats = len(cats)
        p_negative = min(0.95, 0.12 + 0.30 * n_cats)
        negative = rng.random() < p_negative
        sentences: list[str] = []
        if negative:
            if not cats:
                cats = ["other"]
                sentences.append(rng.choice(_pool(GENERIC_NEG, split)))
            for c in cats:
                if c != "other":
                    sentences.append(rng.choice(_pool(COMPLAINTS[c], split)))
            if rng.random() < 0.25:
                sentences.append(rng.choice(_pool(PRAISE, split)))  # mixed review
            if rng.random() < 0.5:
                extra = rng.choice(_pool(GENERIC_NEG, split))
                if extra not in sentences:
                    sentences.append(extra)
        else:
            sentences.extend(rng.choice(_pool(PRAISE, split), 2 if rng.random() < 0.5 else 1, replace=False).tolist())
            # a recommended review can still mention one minor complaint
            cats = cats[:1] if rng.random() < 0.35 else []
            for c in cats:
                sentences.append("Only issue: " + rng.choice(_pool(COMPLAINTS[c], split)).lower())
            if rng.random() < 0.15:
                sentences.append(rng.choice(NEGATION_TRAPS))
        rng.shuffle(sentences)
        text = " ".join(sentences)
        if rng.random() < 0.3:
            text = _typo(text, rng)
        if rng.random() < 0.05:
            text = text.upper()
        if rng.random() < 0.05:
            text += " " + rng.choice(["10/10", ":(", "!!!", "pls fix", "gg"])
        created = day + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
        updated = created if rng.random() < 0.85 else created + pd.Timedelta(days=int(rng.integers(1, 40)))
        rows.append({
            "review_id": f"DEMO-{k + 1:06d}", "game_id": g.game_id, "game_name": g.game_name,
            "review_text": text, "voted_up": not negative,
            "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"), "updated_at": updated.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": rng.choice(["windows", "steam_deck", "linux", "mac"], p=[0.82, 0.10, 0.04, 0.04]),
            "playtime_hours": round(float(rng.gamma(2.0, 15.0 if not negative else 8.0)), 1),
            "helpful_votes": int(rng.poisson(1.5 if negative else 0.8)), "language": "english",
            "data_source": "synthetic_demo"})
        labels.append({"review_id": rows[-1]["review_id"], "split": split, "intended_negative": negative,
                       **{c: int(c in cats) for c in CATEGORIES}})
    df, lab = pd.DataFrame(rows), pd.DataFrame(labels)

    # ---- inject data-quality defects (documented, detected by the cleaning pipeline)
    defects = []
    dup = df.sample(frac=0.01, random_state=seed)
    defects.append(dup)  # exact duplicate rows (same review_id)
    spam = df.sample(frac=0.004, random_state=seed + 1).assign(
        review_id=lambda d: d["review_id"] + "-SPAM", review_text="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" * 4)
    defects.append(spam)
    foreign = df.sample(n=int(n * 0.03), random_state=seed + 2).copy()
    pick = rng.integers(0, len(FOREIGN), len(foreign))
    foreign["review_id"] = foreign["review_id"] + "-L"
    foreign["language"] = [FOREIGN[i][0] for i in pick]
    foreign["review_text"] = [FOREIGN[i][1] for i in pick]
    defects.append(foreign)
    out = pd.concat([df] + defects, ignore_index=True)
    bad = out.sample(frac=0.004, random_state=seed + 3).index
    out.loc[bad[: len(bad) // 3], "review_text"] = "   "
    out.loc[bad[len(bad) // 3: 2 * len(bad) // 3], "created_at"] = "not-a-date"
    out.loc[bad[2 * len(bad) // 3:], "playtime_hours"] = -5.0
    late = out.sample(frac=0.003, random_state=seed + 4).index
    out.loc[late, "updated_at"] = "2026-01-01T00:00:00Z"  # update before creation -> invalid
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, lab, PATCHES


def write(out_dir: Path, n: int = 12_000, seed: int = 7) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    reviews, labels, patches = generate(n, seed)
    reviews.to_csv(out_dir / "demo_reviews.csv", index=False)
    labels.to_parquet(out_dir / "labels_synthetic.parquet", index=False)
    notes = [{"doc_id": f"PATCH-{p['game_id']}-{p['version']}", "game_id": p["game_id"],
              "game_name": next(g["game_name"] for g in GAMES if g["game_id"] == p["game_id"]),
              "version": p["version"], "release_date": p["date"], "changes": p["notes"],
              "data_source": "synthetic_demo"} for p in patches]
    (out_dir / "demo_patch_notes.json").write_text(json.dumps(notes, indent=2))
    meta = {"data_source": "SYNTHETIC DEMO - fictional games, reviews and patch notes; not real player feedback",
            "generator": "src/ingestion/demo_generator.py", "seed": seed, "requested_reviews": n,
            "raw_rows": int(len(reviews)), "games": [g["game_name"] for g in GAMES],
            "period": [str(START.date()), str(END.date())]}
    (out_dir / "demo_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the SYNTHETIC demo review dataset")
    ap.add_argument("--reviews", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(json.dumps(write(get_settings().raw_dir, a.reviews, a.seed), indent=2))
