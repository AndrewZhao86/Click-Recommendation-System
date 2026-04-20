"""Process-wide AIOKafkaProducer singleton for the ingestion API.

The producer config mirrors [scripts/replay_clicks.py](../../../scripts/replay_clicks.py)
so both write paths share identical durability guarantees: `acks=all`,
`enable_idempotence=True`, `linger_ms=5`, `lz4` compression, orjson values,
utf-8 keys.

`publish_event` uses `send_and_wait` so HTTP 202 is only returned after the
broker acks — otherwise a silent post-202 flush failure would violate the
durability contract promised to clients.
"""

from __future__ import annotations

import logging
from typing import Any

import orjson
from aiokafka import AIOKafkaProducer

from click_rec.config import get_settings

logger = logging.getLogger(__name__)

_producer: AIOKafkaProducer | None = None


async def start_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is not None:
        return _producer
    settings = get_settings()
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        acks="all",
        enable_idempotence=True,
        linger_ms=5,
        compression_type="lz4",
        value_serializer=orjson.dumps,
        key_serializer=str.encode,
    )
    await producer.start()
    _producer = producer
    logger.info("kafka producer started")
    return producer


async def stop_producer() -> None:
    global _producer
    if _producer is None:
        return
    try:
        await _producer.stop()
    finally:
        _producer = None
        logger.info("kafka producer stopped")


def get_producer() -> AIOKafkaProducer:
    if _producer is None:
        raise RuntimeError("producer not started — ensure FastAPI lifespan is wired")
    return _producer


async def publish_event(topic: str, key: str, event: dict[str, Any]) -> None:
    producer = get_producer()
    await producer.send_and_wait(topic, event, key=key)
