"""Phase 4b — replay `user.clicks.dlq` back to `user.clicks`.

Usage: `make replay-dlq` (CLI wires through to `run()`).

The DLQ exists primarily for **poison pills** (Phase 4 review D4): payloads
that the consumer couldn't even parse. Re-publishing a poison pill back to
the main topic just sends it on a round-trip to the DLQ again, so this
script must:

1. Read a snapshot of the DLQ (from current offset to end-of-log at start).
2. Try to JSON-parse each `original_payload`.
   - Success → re-publish to `user.clicks` keyed on `user_id`.
   - Failure → write to a quarantine file for manual review and skip.
3. Report `(replayed, skipped)` counts.

The dedicated consumer group `dlq-replayer` keeps offset bookkeeping
isolated from the main `click-enricher` group.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import OffsetAndMetadata

from click_rec.config import get_settings
from click_rec.kafka.admin import ensure_topics
from click_rec.kafka.topics import USER_CLICKS, USER_CLICKS_DLQ

logger = logging.getLogger(__name__)

QUARANTINE_PATH = Path("./tmp/dlq-quarantine.jsonl")
DLQ_REPLAYER_GROUP = "dlq-replayer"
EMPTY_BATCH_LIMIT = 3  # consecutive empty polls before we declare "done"


def _quarantine(record: dict[str, Any]) -> None:
    QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUARANTINE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def replay_dlq(
    *,
    bootstrap: str,
    poll_timeout_ms: int = 1000,
) -> tuple[int, int]:
    """Drain the DLQ once and re-publish parseable rows. Returns (replayed, skipped).

    Phase 4 review D4: `orjson.loads(original_payload)` must be guarded —
    otherwise the very poison pills that put records on the DLQ in the
    first place crash this tool on its first record.
    """
    consumer = AIOKafkaConsumer(
        USER_CLICKS_DLQ.name,
        bootstrap_servers=bootstrap,
        group_id=DLQ_REPLAYER_GROUP,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=bootstrap,
        acks="all",
        enable_idempotence=True,
        linger_ms=5,
        compression_type="gzip",
        value_serializer=orjson.dumps,
        key_serializer=lambda s: s.encode("utf-8") if s is not None else None,
    )
    await consumer.start()
    await producer.start()

    replayed = 0
    skipped = 0
    consecutive_empty = 0

    try:
        while consecutive_empty < EMPTY_BATCH_LIMIT:
            batch = await consumer.getmany(timeout_ms=poll_timeout_ms)
            if not batch:
                consecutive_empty += 1
                continue
            consecutive_empty = 0

            for tp, records in batch.items():
                last_committable: int | None = None
                for record in records:
                    handled = await _handle_dlq_record(record, producer)
                    if handled == "replayed":
                        replayed += 1
                    else:
                        skipped += 1
                    last_committable = record.offset
                if last_committable is not None:
                    await consumer.commit({tp: OffsetAndMetadata(last_committable + 1, "")})
    finally:
        await producer.stop()
        await consumer.stop()

    logger.info(
        "DLQ replay complete: replayed=%d skipped=%d (quarantine=%s)",
        replayed,
        skipped,
        QUARANTINE_PATH if skipped else "n/a",
    )
    return replayed, skipped


async def _handle_dlq_record(record: Any, producer: AIOKafkaProducer) -> str:
    raw = record.value if record.value is not None else b""
    try:
        dlq_msg = orjson.loads(raw)
    except orjson.JSONDecodeError as exc:
        logger.warning(
            "DLQ envelope itself unparseable, quarantining",
            extra={"partition": record.partition, "offset": record.offset, "error": str(exc)},
        )
        _quarantine(
            {
                "stage": "envelope_parse_failed",
                "partition": record.partition,
                "offset": record.offset,
                "raw": raw.decode("utf-8", errors="replace"),
                "ts_ms": int(time.time() * 1000),
            }
        )
        return "skipped"

    original_payload = dlq_msg.get("original_payload")
    if isinstance(original_payload, str):
        # Poison pill — the consumer stored the raw bytes as a string
        # because they weren't JSON. There is nothing to re-publish.
        logger.info(
            "DLQ row holds raw (unparseable) payload, quarantining",
            extra={"reason": dlq_msg.get("reason")},
        )
        _quarantine(dlq_msg)
        return "skipped"

    if not isinstance(original_payload, dict):
        logger.info(
            "DLQ row payload is not a dict, quarantining",
            extra={"type": type(original_payload).__name__},
        )
        _quarantine(dlq_msg)
        return "skipped"

    user_id = original_payload.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        logger.info("DLQ row missing user_id, quarantining")
        _quarantine(dlq_msg)
        return "skipped"

    await producer.send_and_wait(USER_CLICKS.name, original_payload, key=user_id)
    return "replayed"


async def run() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    await ensure_topics()
    replayed, skipped = await replay_dlq(bootstrap=settings.kafka_bootstrap)
    logger.info("done: replayed=%d skipped=%d", replayed, skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
