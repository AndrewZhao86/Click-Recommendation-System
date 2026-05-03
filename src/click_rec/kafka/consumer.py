"""Kafka enrichment consumer pool.

`run_consumer_pool(workers=N)` launches N asyncio tasks, each owning its own
`AIOKafkaConsumer` in the `click-enricher` group. Kafka's group coordinator
distributes the 6 partitions of `user.clicks` across all members in the
group, so the pool naturally scales horizontally — `make consumer N=3` and
`make consumer N=6` differ only in how partitions are assigned.

Per-message pipeline:

    parse + validate (poison pill → DLQ)
        → check consumer-side dedupe marker (skip if already processed)
        → apply_enrichment (DB transaction + Redis writes + profile fan-out)
        → mark as processed (Phase 4 review C4(a): set marker AFTER success)

On exception: exponential back-off retry up to `consumer_max_retries`, then
publish to `user.clicks.dlq`. Offsets are committed per-tp using the
*highest successfully-handled offset* in the batch (Phase 4 review C2) —
never `messages[-1].offset + 1`. A single batched `consumer.commit({...})`
runs once per poll cycle (Phase 4 review D3).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import time
import traceback
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord, TopicPartition
from aiokafka.errors import KafkaError
from aiokafka.structs import OffsetAndMetadata
from pydantic import ValidationError

from click_rec.cache.redis_client import (
    is_consumer_event_processed,
    mark_consumer_event_processed,
)
from click_rec.config import get_settings
from click_rec.db.base import get_sessionmaker
from click_rec.kafka.admin import ensure_topics
from click_rec.kafka.enrichment import apply_enrichment
from click_rec.kafka.topics import USER_CLICKS, USER_CLICKS_DLQ
from click_rec.models.schemas import ClickEventDTO
from click_rec.telemetry.metrics import kafka_consumer_lag

logger = logging.getLogger(__name__)


_LAG_POLL_INTERVAL_S = 5.0


async def _lag_poller(
    consumer: AIOKafkaConsumer, group: str, stop_event: asyncio.Event
) -> None:
    """Periodically emit `kafka_consumer_lag` per assigned (topic, partition).

    Reads `consumer.committed()` vs `consumer.end_offsets()` every
    `_LAG_POLL_INTERVAL_S` seconds. Avoids requiring a JMX exporter
    sidecar — gives a queryable lag gauge for the load-test acceptance
    criterion ("consumer lag never exceeds 1 000").

    Fail-open: any broker hiccup logs and continues on the next tick;
    losing one sample is preferable to crashing the worker.
    """
    while not stop_event.is_set():
        try:
            assigned = consumer.assignment()
            if assigned:
                end_offsets = await consumer.end_offsets(list(assigned))
                for tp in assigned:
                    committed = await consumer.committed(tp)
                    end = end_offsets.get(tp, 0)
                    committed_int = committed if committed is not None else 0
                    lag = max(0, end - committed_int)
                    kafka_consumer_lag.labels(
                        topic=tp.topic,
                        partition=str(tp.partition),
                        group=group,
                    ).set(lag)
        except Exception:  # noqa: BLE001
            logger.exception("lag poller tick failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_LAG_POLL_INTERVAL_S)
        except TimeoutError:
            continue


class ProcessOutcome(StrEnum):
    """What happened to a single message — drives offset-commit decisions.

    Phase 4 review C2: the worker loop commits the highest *successful*
    offset per topic-partition. `OK` and `DLQ_OK` both advance the offset
    (the message is durable somewhere — main topic done, or safely on
    the DLQ). `DLQ_FAILED` does NOT advance: the message is in flight
    nowhere durable, so we must re-fetch it on the next poll.
    """

    OK = "ok"
    DLQ_OK = "dlq_ok"
    DLQ_FAILED = "dlq_failed"


async def _send_to_dlq(
    producer: AIOKafkaProducer,
    record: ConsumerRecord[bytes, bytes],
    exc: BaseException | None,
    reason: str,
) -> bool:
    """Publish a failed record to the DLQ with full context. Returns True on success.

    Phase 4 review C3: format the traceback explicitly from the exception
    object. `traceback.format_exc()` reads from the *currently active*
    exception, which has unwound by the time the retry loop calls us
    from a `last_exc = exc; ...; await _send_to_dlq(record, last_exc)`
    branch — leaving `"NoneType: None\\n"` in the DLQ.
    """
    if exc is not None:
        stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_class = type(exc).__name__
        error_message = str(exc)
    else:
        stack = ""
        error_class = "UnknownError"
        error_message = reason

    raw = record.value if record.value is not None else b""
    try:
        original_payload: Any = orjson.loads(raw)
    except orjson.JSONDecodeError:
        # The raw bytes are the only forensic signal for a poison pill —
        # base64 would obscure it for an operator reading the DLQ.
        original_payload = raw.decode("utf-8", errors="replace")

    dlq_message = {
        "original_topic": record.topic,
        "original_partition": record.partition,
        "original_offset": record.offset,
        "original_key": record.key.decode("utf-8", errors="replace") if record.key else None,
        "original_payload": original_payload,
        "reason": reason,
        "error_class": error_class,
        "error_message": error_message,
        "stack_trace": stack,
        "failed_at_ms": int(time.time() * 1000),
    }
    try:
        await producer.send_and_wait(USER_CLICKS_DLQ.name, dlq_message)
        return True
    except KafkaError:
        # Phase 4 review C2: do NOT swallow this. The caller must see we
        # failed so the offset for this message is held back and the
        # broker re-delivers it next poll.
        logger.critical(
            "DLQ publish failed — message will be retried on next poll",
            exc_info=True,
            extra={
                "topic": record.topic,
                "partition": record.partition,
                "offset": record.offset,
            },
        )
        return False


async def _process_one(
    record: ConsumerRecord[bytes, bytes],
    producer: AIOKafkaProducer,
    worker_id: int,
) -> ProcessOutcome:
    """Run the full pipeline for one record, with retry + DLQ on failure."""
    settings = get_settings()
    started_ms = time.monotonic() * 1000.0

    # Step 1: parse + validate. A failure here is a poison pill — no
    # number of retries will help, so it goes straight to the DLQ.
    raw = record.value if record.value is not None else b""
    try:
        decoded = orjson.loads(raw)
        event = ClickEventDTO.model_validate(decoded)
    except (orjson.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "poison pill: parse/validate failed",
            extra={
                "worker_id": worker_id,
                "partition": record.partition,
                "offset": record.offset,
                "error": str(exc),
            },
        )
        sent = await _send_to_dlq(producer, record, exc, reason="parse_or_validation_error")
        return ProcessOutcome.DLQ_OK if sent else ProcessOutcome.DLQ_FAILED

    event_id = event.event_id

    # Step 2: idempotency short-circuit. The marker is set in step 4
    # *after* every side-effect succeeds, so a replayed message that
    # crashed mid-pipeline last time will see no marker here and re-run
    # the (idempotent) side-effects.
    if await is_consumer_event_processed(event_id):
        duration_ms = (time.monotonic() * 1000.0) - started_ms
        logger.info(
            "event already processed — skipping",
            extra={
                "worker_id": worker_id,
                "event_id": str(event_id),
                "duration_ms": round(duration_ms, 2),
                "status": "skip",
            },
        )
        return ProcessOutcome.OK

    # Step 3: apply side-effects with bounded retry + back-off.
    sessionmaker = get_sessionmaker()
    last_exc: BaseException | None = None
    for attempt in range(1, settings.consumer_max_retries + 1):
        try:
            async with sessionmaker() as session:
                await apply_enrichment(
                    event=event,
                    session=session,
                    redis_client=_get_redis_for_consumer(),
                    producer=producer,
                )
            # Step 4: mark as processed AFTER success (review C4(a)).
            await mark_consumer_event_processed(event_id)
            duration_ms = (time.monotonic() * 1000.0) - started_ms
            logger.info(
                "event enriched",
                extra={
                    "worker_id": worker_id,
                    "event_id": str(event_id),
                    "user_id": event.user_id,
                    "item_id": event.item_id,
                    "attempt": attempt,
                    "duration_ms": round(duration_ms, 2),
                    "status": "ok",
                },
            )
            return ProcessOutcome.OK
        except Exception as exc:
            last_exc = exc
            if attempt < settings.consumer_max_retries:
                backoff = settings.consumer_backoff_base_s * (2 ** (attempt - 1))
                logger.warning(
                    "enrichment failed, retrying",
                    extra={
                        "worker_id": worker_id,
                        "event_id": str(event_id),
                        "attempt": attempt,
                        "backoff_s": backoff,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(backoff)

    # Step 5: retries exhausted → DLQ.
    duration_ms = (time.monotonic() * 1000.0) - started_ms
    logger.error(
        "enrichment exhausted retries, sending to DLQ",
        extra={
            "worker_id": worker_id,
            "event_id": str(event_id),
            "duration_ms": round(duration_ms, 2),
            "status": "dlq",
        },
        exc_info=last_exc,
    )
    sent = await _send_to_dlq(
        producer, record, last_exc, reason="retries_exhausted"
    )
    return ProcessOutcome.DLQ_OK if sent else ProcessOutcome.DLQ_FAILED


# Redis helper kept module-level so tests can monkeypatch it without
# touching the lifespan-managed singleton.
def _get_redis_for_consumer() -> Any:
    from click_rec.cache.redis_client import get_redis

    return get_redis()


async def _worker_loop(
    worker_id: int,
    bootstrap: str,
    group_id: str,
    producer: AIOKafkaProducer,
    stop_event: asyncio.Event,
) -> None:
    """One asyncio task per call: own a consumer, poll, dispatch, commit.

    Note on rebalances: there is intentionally no `ConsumerRebalanceListener`.
    A no-arg `consumer.commit()` from a listener commits the *fetched*
    position, not the *processed* position — that would silently advance
    past unprocessed messages in the C2 scenario (a `DLQ_FAILED` outcome
    holding back the offset, then a rebalance fires before the next poll).
    Without a listener, rebalance falls back to the at-least-once default:
    the new owner re-fetches from the last committed offset and re-runs
    any side-effects already performed. The dedupe marker
    (`mark_consumer_event_processed`) short-circuits the common path,
    and remaining side-effects are idempotent or accumulator-bounded —
    see the enrichment module docstring.
    """
    settings = get_settings()
    consumer = AIOKafkaConsumer(
        USER_CLICKS.name,
        bootstrap_servers=bootstrap,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=settings.consumer_max_records,
    )
    await consumer.start()
    logger.info("worker %d: started, joined group %s", worker_id, group_id)
    lag_task = asyncio.create_task(
        _lag_poller(consumer, group_id, stop_event),
        name=f"consumer-lag-{worker_id}",
    )
    try:
        while not stop_event.is_set():
            batch = await consumer.getmany(
                timeout_ms=settings.consumer_poll_timeout_ms,
                max_records=settings.consumer_max_records,
            )
            if not batch:
                continue

            # Phase 4 review C2 + D3: for each tp, track the highest
            # offset that was *successfully* handled (OK or DLQ_OK).
            # Build one dict and commit once per poll cycle.
            commits: dict[TopicPartition, OffsetAndMetadata] = {}
            for tp, records in batch.items():
                last_committable_offset: int | None = None
                for record in records:
                    outcome = await _process_one(record, producer, worker_id)
                    if outcome in (ProcessOutcome.OK, ProcessOutcome.DLQ_OK):
                        last_committable_offset = record.offset
                    else:
                        # First failure on this tp: stop, leave the rest
                        # uncommitted, broker re-delivers next poll.
                        break
                if last_committable_offset is not None:
                    commits[tp] = OffsetAndMetadata(last_committable_offset + 1, "")

            if commits:
                try:
                    await consumer.commit(commits)
                except KafkaError:
                    # A broker hiccup during commit must not crash the
                    # worker; offsets simply stay where they were and
                    # the next successful poll will commit again.
                    logger.exception(
                        "offset commit failed — will retry next cycle",
                        extra={"worker_id": worker_id},
                    )
    finally:
        lag_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await lag_task
        with contextlib.suppress(Exception):
            await consumer.stop()
        logger.info("worker %d: stopped", worker_id)


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    """SIGINT/SIGTERM → set stop_event so workers drain cleanly.

    Windows asyncio doesn't implement `add_signal_handler`. There Ctrl+C
    raises KeyboardInterrupt at the top of `asyncio.run`; the consumers
    still drain via the `finally` block in `_worker_loop`. SIGTERM works
    in the docker-compose containers and in CI (Linux).
    """
    if sys.platform == "win32":
        return

    def _handler() -> None:
        logger.info("signal received — initiating graceful shutdown")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _handler)


async def run_consumer_pool(workers: int = 1) -> int:
    """Start `workers` consumer tasks, run until SIGINT/SIGTERM, drain cleanly.

    Logging configuration is the entrypoint's responsibility (see
    `cli.main`); library code calling this function should configure
    its own root logger.
    """
    settings = get_settings()
    logger.info(
        "starting %d consumer worker(s) on %s (group=%s, pid=%d)",
        workers,
        settings.kafka_bootstrap,
        settings.consumer_group,
        os.getpid(),
    )

    # Topics + a single shared producer for DLQ + profile-update writes.
    # One producer per process is plenty (aiokafka multiplexes sends).
    await ensure_topics()

    from click_rec.cache.redis_client import start_redis, stop_redis

    await start_redis()

    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        acks="all",
        enable_idempotence=True,
        linger_ms=5,
        compression_type="gzip",
        value_serializer=orjson.dumps,
        key_serializer=lambda s: s.encode("utf-8") if s is not None else None,
    )
    await producer.start()

    stop_event = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), stop_event)

    tasks = [
        asyncio.create_task(
            _worker_loop(
                worker_id=i,
                bootstrap=settings.kafka_bootstrap,
                group_id=settings.consumer_group,
                producer=producer,
                stop_event=stop_event,
            ),
            name=f"consumer-worker-{i}",
        )
        for i in range(workers)
    ]

    # Phase 5 step 2: one popularity refresher per pool (not per worker).
    # It scans `items:top:*`, trims to cache_top_max_members, and decays
    # scores every cache_refresh_interval_seconds. Listed in the gather
    # so shutdown joins it cleanly via `stop_event`.
    from click_rec.cache.popularity_refresher import run_popularity_refresher

    tasks.append(
        asyncio.create_task(
            run_popularity_refresher(stop_event),
            name="popularity-refresher",
        )
    )

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await producer.stop()
        with contextlib.suppress(Exception):
            await stop_redis()
        from click_rec.db.base import dispose_engine

        with contextlib.suppress(Exception):
            await dispose_engine()
        logger.info("consumer pool stopped")
    return 0


__all__: Sequence[str] = (
    "ProcessOutcome",
    "apply_enrichment",
    "run_consumer_pool",
)
