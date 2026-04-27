"""Prometheus metrics for the Phase 5 hot-cache layer.

Counters and one histogram live at module level so every import path
sees the same series — `prometheus_client` de-duplicates by registry,
but a series created inside a function would be re-registered on every
call and blow up with `Duplicated timeseries`.

`mount_metrics(app)` wires `prometheus_client.make_asgi_app()` onto
`/metrics` when `settings.prometheus_enabled` is True. Gated so tests
that build a bare FastAPI app don't accidentally expose global state.
"""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from click_rec.config import get_settings

cache_hit_total = Counter(
    "cache_hit_total",
    # `status` distinguishes a real value hit from a null-sentinel hit
    # (a cached miss for a deleted/unknown key). Both protect Postgres —
    # splitting them lets dashboards answer "what fraction of my 404s
    # bypassed the DB?" separately from the headline hit-ratio.
    "Cache-aside hits. Labelled by key_type (e.g. 'item') and status ('value' | 'null').",
    ["key_type", "status"],
)

cache_miss_total = Counter(
    "cache_miss_total",
    "Cache-aside misses that fell through to the backing store.",
    ["key_type"],
)

cache_lock_wait_seconds = Histogram(
    "cache_lock_wait_seconds",
    "Time spent waiting for another caller to finish a cache fill.",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0),
)

cache_unavailable_total = Counter(
    "cache_unavailable_total",
    "Fail-open events: Redis returned an error and we fell through.",
    ["operation"],
)

# ---------------------------------------------------------------- Phase 6 ranker

ranker_latency_seconds = Histogram(
    "ranker_latency_seconds",
    "Ranker latency by stage.",
    ["stage"],  # candidates | features | score | total
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

ranker_candidates_total = Histogram(
    "ranker_candidates_total",
    "Candidate pool size per request.",
    buckets=(0, 10, 50, 100, 150, 200, 250),
)

ranker_cold_start_total = Counter(
    "ranker_cold_start_total",
    "Requests with no recent-click history.",
)

ranker_empty_result_total = Counter(
    "ranker_empty_result_total",
    "Requests where both BM25 and vector channels returned 0.",
)

# ---------------------------------------------------------------- Phase 7 LLM

llm_request_total = Counter(
    "llm_request_total",
    "LLM requests by use case and outcome.",
    # use_case: query_understanding | re_rank | explain | judge
    # outcome:  success | timeout | parse_error | error | unavailable
    ["use_case", "outcome"],
)

llm_request_latency_seconds = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency by use case.",
    ["use_case"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)

llm_token_usage_total = Counter(
    "llm_token_usage_total",
    "Tokens consumed by use case and kind (prompt | completion).",
    ["use_case", "kind"],
)

llm_timeout_total = Counter(
    "llm_timeout_total",
    "Timeouts triggering fallback, by use case.",
    ["use_case"],
)

llm_fallback_total = Counter(
    "llm_fallback_total",
    "Fallback to deterministic path. reason: timeout|parse_error|unavailable|error.",
    ["use_case", "reason"],
)

llm_cache_hit_total = Counter(
    "llm_cache_hit_total",
    "Redis cache hits for LLM responses.",
    ["use_case"],
)


def mount_metrics(app: FastAPI) -> None:
    """Expose `/metrics` when Prometheus is enabled in settings.

    Uses a plain route instead of `make_asgi_app()` because mounting a
    sub-ASGI app on FastAPI in lifespan mode triggers a second lifespan
    call, which double-starts the Redis + Kafka singletons.
    """
    if not get_settings().prometheus_enabled:
        return

    from starlette.responses import Response

    @app.get("/metrics", include_in_schema=False)
    async def _metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
