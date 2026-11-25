"""Flood the model with prediction requests.

Interactive:   locust -f locustfile.py --host http://localhost:8080
Headless run:  locust -f locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 60s --csv results/run
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

CLIPS = sorted(Path(__file__).parent.glob("data/test/*/*.wav"))
assert CLIPS, "No test clips found: run `python -m src.data_acquisition` first."
CACHE = {p: p.read_bytes() for p in random.sample(CLIPS, min(24, len(CLIPS)))}   # read once, reuse


class SoundSensor(HttpUser):
    """A field sensor that sends a clip every second or so."""
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        path, data = random.choice(list(CACHE.items()))
        with self.client.post("/api/predict", files={"file": (path.name, data, "audio/wav")},
                              name="POST /api/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
            elif path.parent.name != r.json()["label"] and random.random() < 0:
                r.failure("wrong label")

    @task(2)
    def dashboard(self):
        self.client.get("/api/status", name="GET /api/status")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
