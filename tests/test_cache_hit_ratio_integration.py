"""Phase 8c §3 cache-hit-ratio acceptance check, without Locust.

The 90% hit-ratio claim in `docs/phase8_artifacts.md:37` is a property
of the access distribution, not the concurrency level — a 30 s
single-process driver against a warmed `/search` endpoint produces the
same ratio as a 5 min × 500-user Locust run, for far less ceremony.

Skips cleanly when the API isn't reachable, so unit-test runs aren't
blocked. Reuses the Prometheus parser from
`scripts/check_loadtest_results.py` so any drift in metric naming gets
caught in one place.
"""

from __future__ import annotations

import os
import random
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


_API_BASE = os.environ.get("CLICK_REC_API", "http://localhost:8000")
_DURATION_S = float(os.environ.get("CACHE_RATIO_DURATION_S", "30"))
_WARMUP_REQS = int(os.environ.get("CACHE_RATIO_WARMUP", "50"))
_MIN_RATIO = 0.90
_QUERY_POOL = Path(__file__).parent / "load" / "queries.txt"
_USER_POOL = Path("artifacts/seeded_users.txt")


def _api_reachable(base: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _scrape_metrics(base: str) -> str:
    with urllib.request.urlopen(f"{base}/metrics", timeout=5) as resp:
        return resp.read().decode("utf-8")


def _zipf_choice(items: list[str], rng: random.Random, s: float = 1.2) -> str:
    n = len(items)
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= r:
            return items[i]
    return items[-1]


def _hit_miss_totals(text: str) -> tuple[float, float]:
    """Sum cache_hit_total and cache_miss_total across all label variants.

    Mirrors the parser in scripts/check_loadtest_results.py so this test
    fails for the same reason the loadtest acceptance script would.
    """
    hits = 0.0
    misses = 0.0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name_with_labels, value = parts
        try:
            v = float(value)
        except ValueError:
            continue
        if name_with_labels.startswith("cache_hit_total"):
            hits += v
        elif name_with_labels.startswith("cache_miss_total"):
            misses += v
    return hits, misses


def test_cache_hit_ratio_above_threshold() -> None:
    if not _api_reachable(_API_BASE):
        pytest.skip(
            f"API not reachable at {_API_BASE}; "
            "run `make up && make migrate && make seed && make api` first"
        )

    if not _QUERY_POOL.exists():
        pytest.skip(f"missing query pool {_QUERY_POOL}")
    queries = [q for q in _QUERY_POOL.read_text(encoding="utf-8").splitlines() if q]
    users = (
        [u for u in _USER_POOL.read_text(encoding="utf-8").splitlines() if u]
        if _USER_POOL.exists()
        else ["u_brand_loyalist_0001"]
    )

    rng = random.Random(0xC4CE)

    # Warmup: prime the cache for the top of the Zipf distribution. The
    # 90% claim is about steady state, not first-hit; warmup misses are
    # intentionally excluded from the ratio measurement by snapshotting
    # counters after warmup completes.
    for _ in range(_WARMUP_REQS):
        q = _zipf_choice(queries, rng)
        u = rng.choice(users)
        try:
            urllib.request.urlopen(
                f"{_API_BASE}/search?q={urllib.request.quote(q)}"
                f"&user_id={urllib.request.quote(u)}&limit=10",
                timeout=5,
            ).read()
        except urllib.error.HTTPError as exc:
            pytest.fail(
                f"warmup /search returned {exc.code}; preflight (seed + replay) "
                f"likely missing — see docs/phase8_artifacts.md:10-26"
            )

    import time

    # Snapshot after warmup so only steady-state traffic is measured.
    # Without this, cold-start misses from warmup dilute the ratio even
    # when the cache is functioning correctly.
    before_hits, before_misses = _hit_miss_totals(_scrape_metrics(_API_BASE))

    # Steady-state load: same Zipf mix the locustfile uses, but
    # single-process so we can measure ratio without contention noise.
    deadline = time.monotonic() + _DURATION_S
    sent = 0
    while time.monotonic() < deadline:
        q = _zipf_choice(queries, rng)
        u = rng.choice(users)
        urllib.request.urlopen(
            f"{_API_BASE}/search?q={urllib.request.quote(q)}"
            f"&user_id={urllib.request.quote(u)}&limit=10",
            timeout=5,
        ).read()
        sent += 1

    after_hits, after_misses = _hit_miss_totals(_scrape_metrics(_API_BASE))
    delta_hits = after_hits - before_hits
    delta_misses = after_misses - before_misses
    total = delta_hits + delta_misses

    assert total > 0, (
        f"no cache traffic recorded after {sent} /search calls — "
        f"check that `cache_hit_total` and `cache_miss_total` are exported"
    )

    ratio = delta_hits / total
    assert ratio >= _MIN_RATIO, (
        f"cache hit ratio={ratio:.2%} (hits={delta_hits:.0f}, "
        f"misses={delta_misses:.0f}) < {_MIN_RATIO:.0%} "
        f"after {sent} requests over {_DURATION_S:.0f}s"
    )
