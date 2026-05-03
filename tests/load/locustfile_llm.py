"""Phase 8c LLM-fallback exercise.

Run at low concurrency with `use_llm=true` so the Gemini Flash free-tier
RPM (~15) is exhausted within ~90 s — proving the 2 s timeout falls
back to the hybrid order without a 5xx.

`make loadtest-llm` runs this for 2 minutes at 10 users. The headline
500 RPS run uses `locustfile.py` with LLM off so quota interference
doesn't perturb the p95 budget.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

from tests.load.locustfile import (
    _load_query_pool,
    _load_seeded_user_ids,
    _zipf_choice,
)


class SearchPulseLLMUser(HttpUser):
    """100% `/search?use_llm=true` at low concurrency."""

    wait_time = between(0.5, 1.5)

    user_ids: list[str] = []
    queries: list[str] = []

    def on_start(self) -> None:
        self.user_ids = _load_seeded_user_ids()
        self.queries = _load_query_pool()

    @task
    def search_with_llm(self) -> None:
        uid = random.choice(self.user_ids)
        q = _zipf_choice(self.queries)
        with self.client.get(
            "/search",
            params={"q": q, "user_id": uid, "limit": 10, "use_llm": "true"},
            name="/search?use_llm=true",
            catch_response=True,
        ) as resp:
            # 200 is the only acceptable outcome — even when the LLM
            # times out / hits quota, the route falls back to hybrid.
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
