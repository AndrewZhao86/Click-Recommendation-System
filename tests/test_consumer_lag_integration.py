"""Integration test for the Phase 8b Kafka consumer-lag poller.

Spins up a real Kafka via testcontainers, publishes events without
consuming them, and asserts `kafka_consumer_lag` is populated by one
tick of `_lag_poller`. The poller is what feeds the Phase 8c
acceptance criterion ("consumer lag never exceeds 1 000") so it gets
its own end-to-end check independent of the full consumer pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytestmark = pytest.mark.integration

try:
    import orjson
    import testcontainers  # noqa: F401
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from testcontainers.kafka import KafkaContainer
except ImportError:
    pytest.skip(
        "testcontainers, aiokafka, or orjson not installed",
        allow_module_level=True,
    )


_TEST_TOPIC = "user.clicks"
_TEST_GROUP = "lag-poller-test"
_NUM_PARTITIONS = 2
_NUM_MESSAGES = 100


@pytest.fixture(scope="module")
def kafka_container() -> Any:
    with KafkaContainer() as kc:
        yield kc


@pytest.fixture
def configured_settings(kafka_container: Any) -> Iterator[str]:
    """Mirror Kafka bootstrap into settings; reset the cached singleton.

    Yields the bootstrap string for direct producer/consumer use.
    """
    mp = pytest.MonkeyPatch()
    bootstrap = kafka_container.get_bootstrap_server()
    mp.setenv("KAFKA_BOOTSTRAP", bootstrap)

    from click_rec.config import get_settings

    get_settings.cache_clear()
    try:
        yield bootstrap
    finally:
        mp.undo()
        get_settings.cache_clear()


async def _create_topic(bootstrap: str) -> None:
    """Create the test topic with multiple partitions. Idempotent."""
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap)
    await admin.start()
    try:
        with contextlib.suppress(Exception):
            # `TopicAlreadyExistsError` (or its variants) — ignore so a
            # rerun against a warm container doesn't fail setup.
            await admin.create_topics(
                [
                    NewTopic(
                        name=_TEST_TOPIC,
                        num_partitions=_NUM_PARTITIONS,
                        replication_factor=1,
                    )
                ]
            )
    finally:
        await admin.close()


async def _publish_messages(bootstrap: str, n: int) -> None:
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap, value_serializer=orjson.dumps)
    await producer.start()
    try:
        for i in range(n):
            # Spread across partitions by varying the key so both ends
            # of the assignment see lag — `consumer.committed()` returns
            # None for any partition that has never committed, so even
            # with one partition we'd see lag, but two partitions
            # exercises the loop body more honestly.
            await producer.send_and_wait(
                _TEST_TOPIC,
                {"event_id": str(i)},
                key=str(i % _NUM_PARTITIONS).encode(),
            )
    finally:
        await producer.stop()


async def _build_subscribed_consumer(
    bootstrap: str,
) -> AsyncIterator[AIOKafkaConsumer]:
    """Subscribe a consumer and wait until partitions are assigned."""
    consumer = AIOKafkaConsumer(
        _TEST_TOPIC,
        bootstrap_servers=bootstrap,
        group_id=_TEST_GROUP,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await consumer.start()
    try:
        # Wait for the rebalance to assign partitions. `getone()` would
        # block on an empty topic, but `assignment()` returns the empty
        # set until the join completes — poll briefly.
        deadline = asyncio.get_running_loop().time() + 10.0
        while not consumer.assignment():
            if asyncio.get_running_loop().time() > deadline:
                raise AssertionError("consumer never received partition assignment")
            await asyncio.sleep(0.1)
        yield consumer
    finally:
        with contextlib.suppress(Exception):
            await consumer.stop()


def _sum_lag_for_group(group: str) -> float:
    """Sum the `kafka_consumer_lag` gauge across labels for `group`."""
    from click_rec.telemetry.metrics import kafka_consumer_lag

    total = 0.0
    for metric in kafka_consumer_lag.collect():
        for sample in metric.samples:
            if sample.name == "kafka_consumer_lag" and sample.labels.get("group") == group:
                total += sample.value
    return total


async def test_lag_poller_emits_gauge_after_replay(
    configured_settings: str,
) -> None:
    """100 unconsumed events → poller populates the gauge with positive lag.

    The lag poller's first iteration runs immediately (the wait is at
    the *end* of the loop body), so the test only needs to wait long
    enough for the consumer to receive its assignment and one tick to
    finish — well under the 5 s poll interval.
    """
    from click_rec.kafka.consumer import _lag_poller

    bootstrap = configured_settings

    await _create_topic(bootstrap)
    await _publish_messages(bootstrap, _NUM_MESSAGES)

    async for consumer in _build_subscribed_consumer(bootstrap):
        stop = asyncio.Event()
        poller_task = asyncio.create_task(
            _lag_poller(consumer, _TEST_GROUP, stop), name="lag-poller-test"
        )
        try:
            # Poll the gauge until the first tick has populated it.
            deadline = asyncio.get_running_loop().time() + 5.0
            total = 0.0
            while asyncio.get_running_loop().time() < deadline:
                total = _sum_lag_for_group(_TEST_GROUP)
                if total > 0:
                    break
                await asyncio.sleep(0.1)
            assert total > 0, (
                f"expected positive lag for group={_TEST_GROUP} after "
                f"publishing {_NUM_MESSAGES} events, gauge sum was {total}"
            )
            # All messages are unread — total lag across partitions
            # should equal the publish count.
            assert total == _NUM_MESSAGES, (
                f"lag should match publish count; got {total}, expected {_NUM_MESSAGES}"
            )
        finally:
            stop.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(poller_task, timeout=2.0)
