"""Phase 8c headline load test.

Mix matches plan §8.8c: 80% `/search`, 15% `/events/click`, 5%
`/events/impression`. Users + queries are sampled with Zipf-like
weighting so the top of the distribution drives cache-hit and
candidate-pool warm-up paths the same way real traffic would.

Inputs:
- `artifacts/seeded_users.txt` — written by `make seed`. Override via
  `SEEDED_USERS` env if running against a different fixture.
- `tests/load/queries.txt` — hand-curated 200-query pool. Override via
  `QUERY_POOL` env.

Acceptance criteria (plan §8.8c verify): zero 5xx, p95 < 150 ms on
`/search` (LLM off), `kafka_consumer_lag` never exceeds 1 000.
"""

from __future__ import annotations

import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path

from locust import HttpUser, between, task

_DEFAULT_USERS = ["u_brand_loyalist_0001"]
_DEFAULT_QUERIES = ["wireless headphones"]


def _load_lines(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return fallback
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln] or fallback


def _load_seeded_user_ids() -> list[str]:
    p = Path(os.environ.get("SEEDED_USERS", "artifacts/seeded_users.txt"))
    return _load_lines(p, _DEFAULT_USERS)


def _load_query_pool() -> list[str]:
    p = Path(
        os.environ.get(
            "QUERY_POOL",
            str(Path(__file__).parent / "queries.txt"),
        )
    )
    return _load_lines(p, _DEFAULT_QUERIES)


def _zipf_choice(items: list[str], s: float = 1.2) -> str:
    """Cheap Zipf-weighted pick — top items get most of the traffic.

    Builds the rank-weight curve once per call (small N; ~200 entries).
    Avoids `numpy.random.zipf` so the locustfile has zero numpy dep.
    """
    n = len(items)
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= r:
            return items[i]
    return items[-1]


class SearchPulseUser(HttpUser):
    """Mixed search + event-write load."""

    # Wait between user iterations. With 500 users and ~0.25s mean wait
    # we land at ~2k req/s nominal — comfortably above the 500 RPS target.
    wait_time = between(0.05, 0.5)

    user_ids: list[str] = []
    queries: list[str] = []

    def on_start(self) -> None:
        self.user_ids = _load_seeded_user_ids()
        self.queries = _load_query_pool()

    @task(80)
    def search(self) -> None:
        uid = random.choice(self.user_ids)
        q = _zipf_choice(self.queries)
        with self.client.get(
            "/search",
            params={"q": q, "user_id": uid, "limit": 10},
            name="/search",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")

    @task(15)
    def click(self) -> None:
        # Random item_ids deliberately don't all match real items —
        # surfaces the cache-miss path. The consumer is idempotent on
        # event_id, so flooding bad ids is safe and *intentionally*
        # exercises `cache_unavailable_total` if anything is mis-wired.
        # See phase8plan.md risks.
        uid = random.choice(self.user_ids)
        payload = {
            "event_type": "click",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": uid,
            "session_id": f"s_{uid}_{uuid.uuid4().hex[:8]}",
            "item_id": f"i_load_{random.randint(0, 9999)}",
            "query": _zipf_choice(self.queries),
            "rank_position": random.randint(1, 10),
            "dwell_ms": random.randint(500, 30000),
        }
        self.client.post("/events/click", json=payload, name="/events/click")

    @task(5)
    def impression(self) -> None:
        uid = random.choice(self.user_ids)
        payload = {
            "event_type": "impression",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": uid,
            "session_id": f"s_{uid}_{uuid.uuid4().hex[:8]}",
            "query": _zipf_choice(self.queries),
            "result_ids": [f"i_load_{random.randint(0, 9999)}" for _ in range(10)],
            "page": 1,
        }
        self.client.post(
            "/events/impression", json=payload, name="/events/impression"
        )
