"""FastAPI lifespan: ensure topics, then start Redis + Kafka producer.

Topics are created before the producer starts so the first `send_and_wait`
doesn't hang on `UnknownTopicOrPartitionError`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from click_rec.cache.redis_client import start_redis, stop_redis
from click_rec.kafka.admin import ensure_topics
from click_rec.kafka.producer import start_producer, stop_producer
from click_rec.ranker import embedder as ranker_embedder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await ensure_topics()
    await start_redis()
    await start_producer()
    # Pre-load the ranker embedder so the first /search request hits
    # steady-state latency rather than the ~1s torch / model cold-load.
    # Best-effort: a model-load failure shouldn't block API startup —
    # the search route surfaces a 503 if the model can't encode.
    try:
        await ranker_embedder.warm()
    except Exception:
        logger.exception("ranker embedder warm failed; /search will load on demand")
    try:
        yield
    finally:
        await stop_producer()
        await stop_redis()
