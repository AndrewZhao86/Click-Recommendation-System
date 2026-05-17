from __future__ import annotations

import asyncio
import contextlib
import logging

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError

from click_rec.config import get_settings
from click_rec.kafka.topics import ALL_TOPICS, TopicConfig

logger = logging.getLogger(__name__)


async def ensure_topics(
    topics: list[TopicConfig] | None = None,
    *,
    retries: int = 3,
    backoff_s: float = 2.0,
) -> None:
    """Create every topic in `topics` (default `ALL_TOPICS`) on the broker.

    Auto-create is disabled in docker-compose, so replay / producers MUST call
    this before the first `send_and_wait` — otherwise they hang or raise
    `UnknownTopicOrPartitionError`. Retries cover the window where the admin
    client races broker startup.
    """
    selected = topics if topics is not None else ALL_TOPICS
    bootstrap = get_settings().kafka_bootstrap
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
        started = False
        try:
            await admin.start()
            started = True
            new_topics = [
                NewTopic(name=t.name, num_partitions=t.partitions, replication_factor=1)
                for t in selected
            ]
            try:
                await admin.create_topics(new_topics)
                logger.info("created kafka topics: %s", [t.name for t in selected])
            except TopicAlreadyExistsError:
                logger.info("kafka topics already exist — continuing")
            return
        except Exception as exc:
            last_exc = exc
            logger.warning("ensure_topics attempt %d/%d failed: %s", attempt, retries, exc)
        finally:
            if started:
                with contextlib.suppress(Exception):
                    await admin.close()

        if attempt < retries:
            await asyncio.sleep(backoff_s)

    assert last_exc is not None
    raise last_exc
