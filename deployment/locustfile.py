"""Load test for the API.  Example (headless, 60 s, 20 users):
    locust -f deployment/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 60s --csv artifacts/eval/locust
"""
import random

from locust import HttpUser, between, task

GAMES = ["G001", "G002", "G003", "G004", "G005", None]
QUERIES = ["servers keep disconnecting", "frame rate drops", "too many microtransactions", "matchmaking takes forever",
           "game crashes on startup", "not enough content"]
QUESTIONS = ["What are players complaining about most in Kingdoms of Ember?",
             "Why did negative reviews for Starfall Arena increase after update 2.1?",
             "Summarize the latest negative reviews for Pitch Legends"]


class Analyst(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(4)
    def overview(self):
        g = random.choice(GAMES)
        self.client.get("/api/v1/analytics/overview", params={"game_id": g} if g else None, name="/analytics/overview")

    @task(3)
    def complaints(self):
        self.client.get("/api/v1/analytics/complaints", params={"start": "2026-07-01", "end": "2026-08-31"}, name="/analytics/complaints")

    @task(3)
    def search(self):
        self.client.post("/api/v1/reviews/search", json={"query": random.choice(QUERIES), "top_k": 5}, name="/reviews/search")

    @task(1)
    def agent(self):
        self.client.post("/api/v1/agent/query", json={"question": random.choice(QUESTIONS)}, name="/agent/query")


class Saturation(Analyst):
    """No think time: measures capacity rather than a realistic user mix.
    locust -f deployment/locustfile.py Saturation --host http://localhost:8000 --headless -u 40 -r 20 -t 30s"""
    wait_time = between(0, 0)
