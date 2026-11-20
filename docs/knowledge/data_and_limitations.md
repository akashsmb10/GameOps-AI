# GameOps AI - dataset and limitations

## Demo dataset
The bundled dataset is SYNTHETIC: five fictional games (Starfall Arena, Kingdoms of Ember, Turbo Circuit 26, Pitch Legends, Hollow Depths), generated reviews and fictional patch notes from 2026-05-01 to 2026-10-31.
It is not real player feedback. Any pattern in it exists because the generator created it.

## Real data
Real Steam reviews can be collected locally with the Steam store review connector (src/ingestion/steam.py). Review text belongs to its authors, so collected data stays in a git-ignored folder and must not be redistributed.

## Cleaning rules
Rows are excluded when the text is empty, the creation timestamp is invalid, the text is spam (long repeated characters), or the review is not in English. Duplicate review ids are removed. Negative playtime and update timestamps earlier than creation are set to missing but the review is kept.

## Known limitations
Review sentiment reflects people who chose to write reviews, which is a self-selected sample. Complaint categories are model predictions with measurable errors. Temporal coincidence between an update and a change in complaints is not proof of causation.
