"""RAG evaluation questions. Relevance comes from the SYNTHETIC generator labels (reviews)
or a known document id; none of these questions were used to tune retrieval."""

CATEGORY_PHRASES = {
    "bugs": "crashes, glitches or broken saves",
    "performance": "frame rate and optimization problems",
    "servers": "connection problems and server outages",
    "balance": "unfair weapons or difficulty balance",
    "monetization": "microtransactions and in-game purchases",
    "matchmaking": "waiting times and unfair matchmaking",
    "ux": "the menus, controls and interface",
    "content": "lack of content and repetitiveness",
    "support": "customer support and refunds",
}
# (game_id, category) pairs used for review-retrieval questions
REVIEW_PAIRS = [("G001", "servers"), ("G001", "matchmaking"), ("G001", "balance"), ("G001", "bugs"),
                ("G002", "monetization"), ("G002", "bugs"), ("G002", "content"), ("G002", "performance"),
                ("G003", "performance"), ("G003", "bugs"), ("G003", "ux"), ("G004", "monetization"),
                ("G004", "balance"), ("G004", "servers"), ("G005", "bugs"), ("G005", "performance"),
                ("G004", "support"), ("G001", "support")]
DOC_QUESTIONS = [
    ("What changed in Starfall Arena update 2.1?", "PATCH-G001-2.1"),
    ("Which patch introduced the Ember Store in Kingdoms of Ember?", "PATCH-G002-1.4"),
    ("What did Turbo Circuit 26 version 3.0 change about graphics?", "PATCH-G003-3.0"),
    ("How is the negative review percentage defined?", "DOC-metric_definitions-positive-and-negative-review-percentage"),
    ("What counts as an emerging issue?", "DOC-metric_definitions-emerging-issue"),
    ("Is the bundled dataset real player feedback?", "DOC-data_and_limitations-demo-dataset"),
]
UNANSWERABLE = [
    "What is the capital of France?", "Give me a recipe for paneer butter masala.",
    "What is the current share price of a chip maker?", "How do I file my income tax return?",
    "Who won the cricket world cup in 2011?", "What are the symptoms of dengue fever?",
]

# DEV set (used only to choose the default review retrieval mode; disjoint pairs + wording)
DEV_PHRASES = {
    "bugs": "the game freezing or errors", "performance": "lag spikes and slow loading",
    "servers": "being disconnected online", "balance": "overpowered characters",
    "monetization": "paying real money", "matchmaking": "long queues", "ux": "the user interface",
    "content": "nothing to do after finishing", "support": "the help team not answering",
}
DEV_PAIRS = [("G001", "performance"), ("G002", "servers"), ("G002", "balance"), ("G003", "servers"),
             ("G003", "monetization"), ("G004", "bugs"), ("G004", "content"), ("G005", "servers"),
             ("G005", "ux"), ("G001", "content")]
