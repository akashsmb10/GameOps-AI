# Data dictionary

## Raw review schema (`src/ingestion/loader.RAW_COLUMNS`)
| Field | Type | Demo source | Steam source | Notes |
|---|---|---|---|---|
| review_id | str | `DEMO-000123` | `steam-<recommendationid>` | primary key |
| game_id | str | `G001`..`G005` | `steam:<appid>` | |
| game_name | str | fictional names | passed by the collector | |
| review_text | str | generated | `review` | original text is never modified |
| voted_up | bool | generated | `voted_up` | **source sentiment label** (recommendation) |
| created_at | ISO-8601 UTC | generated | `timestamp_created` | |
| updated_at | ISO-8601 UTC | generated | `timestamp_updated` | set to missing if earlier than created_at |
| platform | str | windows / steam_deck / linux / mac | `steam_deck` or `steam_pc_or_other` | Steam gives only a Deck flag |
| playtime_hours | float | generated | `author.playtime_at_review / 60` | negative values set to missing |
| helpful_votes | int | generated | `votes_up` | |
| language | str | english (+ injected foreign rows) | `language` | |
| data_source | str | `synthetic_demo` | `steam_store_api` | |
| *(dropped)* | | | `author.steamid` etc. | reviewer identifiers are never stored |

## Added by cleaning (`reviews_clean.parquet`)
`text_norm` (lower-case, URLs removed, whitespace collapsed) · `detected_language` · `text_hash` (game + normalised text) ·
`is_text_duplicate` · `char_len`, `word_len` · `include` · `exclusion_reason`.

Exclusion rules (`include = False`): empty text, invalid created_at, missing recommendation, spam text (character runs ≥ 20
or < 5% distinct characters), non-English (source language or detector). Duplicate review_ids are removed (first kept).

## Added by NLP (`review_predictions.parquet`)
`bugs, performance, servers, balance, monetization, matchmaking, ux, content, support` (bool, predicted) ·
`other` (negative recommendation with no predicted category) · `model_p_negative`, `model_sentiment` (text model).

## Synthetic labels (`data/raw/labels_synthetic.parquet`, demo only)
`review_id, split (train/validation/test), intended_negative, <category flags>` - the categories the generator injected.
Used only for training/evaluating models on the demo; never shown as facts about real players.

## Update notes
`doc_id, game_id, game_name, version, release_date, changes[], data_source` (`demo_patch_notes.json` or Steam news JSON).

## Manual labels for real data (recommended)
To evaluate on real reviews, create `data/labels/manual_labels.csv` with `review_id` + the nine category columns (0/1),
labelled by a person following `docs/knowledge/complaint_categories.md`, and evaluate the saved models against it.
