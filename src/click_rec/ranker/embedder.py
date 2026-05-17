"""Process-local MiniLM singleton for query encoding.

Lazy-loads on first call (~1s cold-start) and reuses the model
afterwards. Encoding runs on the asyncio default executor so the event
loop isn't blocked for the ~5ms forward pass.

`warm()` is a one-shot pre-load wired into the FastAPI lifespan so the
first user request never pays the cold-load tax. Without it, a `curl`
during verification step 2 reports a misleading ~1s number.

Phase 8 adds a Redis cache-aside in front of the encode call. The cold
encode is ~80–2000 ms depending on whether the asyncio executor is
saturated; warm encodes are ~5 ms. Under Zipf-weighted query traffic
the steady-state hit rate exceeds 95 %, which is what lets `/search`
p95 fit the 150 ms budget. Hits/misses surface on the existing
`cache_hit_total` / `cache_miss_total` counters under
`key_type="query_embedding"` so the same scrape parser
(`scripts/check_loadtest_results.py`) covers both this path and the
item cache.

Matches the catalog seeder's `normalize_embeddings=False` setting at
[seed_catalog.py:190](../../../scripts/seed_catalog.py) — the IVFFLAT
index uses cosine ops, so encoded vectors are compared via the `<=>`
operator rather than dot product, and consistent normalisation between
query and index is what we actually need (both unnormalised).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

import numpy as np
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from click_rec.cache.redis_client import get_redis
from click_rec.config import get_settings
from click_rec.telemetry.metrics import (
    cache_hit_total,
    cache_miss_total,
    cache_unavailable_total,
)

logger = logging.getLogger(__name__)

_model: Any = None
_load_lock: asyncio.Lock | None = None

_REDIS_ERRORS = (RedisConnectionError, RedisTimeoutError, RedisError)
_KEY_TYPE = "query_embedding"
# float32 keeps the cached payload at 1.5 KB for a 384-dim vector — an
# order of magnitude under orjson list encoding and still well within
# MiniLM's effective precision (the model itself runs fp32 inference but
# downstream pgvector ops are cosine-similarity bound, not bit-exact).
_DTYPE = np.float32


def _get_load_lock() -> asyncio.Lock:
    """Lazily create the lock so the module imports cleanly off-loop."""
    global _load_lock
    if _load_lock is None:
        _load_lock = asyncio.Lock()
    return _load_lock


async def _load_model() -> Any:
    """Lazy-load the SentenceTransformer model under a lock."""
    global _model
    lock = _get_load_lock()
    async with lock:
        if _model is not None:
            return _model
        # Imported lazily so `pytest --collect-only` and the `--help`
        # path don't pay the ~1s torch import.
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        logger.info("loading ranker embedding model: %s", settings.embedding_model)
        # The constructor blocks on disk + torch init; offload it.
        loop = asyncio.get_running_loop()
        _model = await loop.run_in_executor(None, SentenceTransformer, settings.embedding_model)
        return _model


def _cache_key(model_name: str, query: str) -> str:
    # Hash the query so arbitrary user input can't blow up the keyspace
    # with control chars or absurd lengths. Model name in the prefix means
    # a model swap invalidates the entire cache without an explicit flush.
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return f"query:embedding:{model_name}:{digest}"


def _encode_payload(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=_DTYPE).tobytes()


def _decode_payload(raw: bytes) -> list[float]:
    out: list[float] = np.frombuffer(raw, dtype=_DTYPE).astype(np.float64).tolist()
    return out


async def _encode_blocking_offloaded(model: Any, query: str) -> list[float]:
    loop = asyncio.get_running_loop()

    def _run() -> list[float]:
        vec = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )[0]
        result: list[float] = vec.tolist()
        return result

    return await loop.run_in_executor(None, _run)


async def encode_query(query: str) -> list[float]:
    """Encode a single query string into a 384-dim vector.

    Cache-aside in Redis: a hit returns in <2 ms, a miss runs the
    SentenceTransformer forward pass on the asyncio executor (~5 ms warm,
    up to ~2 s cold under load) and writes the result back. Any Redis
    error fails open to a direct encode — the cache is a perf optimisation,
    never a correctness boundary.

    Matches catalog encoding: `normalize_embeddings=False`. Cosine
    semantics in the vector channel come from pgvector's `<=>` operator,
    not from L2-normalising both sides ourselves.
    """
    settings = get_settings()
    key = _cache_key(settings.embedding_model, query)

    client = None
    try:
        client = get_redis()
    except RuntimeError:
        # Redis singleton not started (e.g. unit-test harness): just
        # encode directly.
        client = None

    if client is not None:
        try:
            raw = await client.get(key)
        except _REDIS_ERRORS as exc:
            logger.warning(
                "cache_unavailable: query embedding GET failed",
                extra={"error": str(exc)},
            )
            cache_unavailable_total.labels(operation="get").inc()
            raw = None
            client = None  # skip the SET on the way back too
        if raw is not None:
            cache_hit_total.labels(key_type=_KEY_TYPE, status="value").inc()
            return _decode_payload(raw)
        cache_miss_total.labels(key_type=_KEY_TYPE).inc()

    model = await _load_model()
    vec = await _encode_blocking_offloaded(model, query)

    if client is not None:
        try:
            await client.set(key, _encode_payload(vec), ex=settings.query_embedding_ttl_seconds)
        except _REDIS_ERRORS as exc:
            logger.warning(
                "cache_unavailable: query embedding SET failed",
                extra={"error": str(exc)},
            )
            cache_unavailable_total.labels(operation="set").inc()

    return vec


async def warm() -> None:
    """Pre-load the model so the first user request hits steady-state.

    Wired into `api/lifespan.py`. Idempotent — repeated calls only run
    one inference because `_model` is cached.
    """
    await encode_query("warmup")
    logger.info("ranker embedder warmed")


def reset_for_tests() -> None:
    """Clear module-level state. Tests use this to swap fakes in/out."""
    global _model, _load_lock
    _model = None
    _load_lock = None
