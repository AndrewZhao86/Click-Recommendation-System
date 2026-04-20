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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await ensure_topics()
    await start_redis()
    await start_producer()
    try:
        yield
    finally:
        await stop_producer()
        await stop_redis()
