"""Phase 2 replay — synthetic search/impression/click stream → Kafka.

Produces events shaped like real user behaviour (Zipf query distribution,
power-law rank positions, per-segment click bias) so Phase 4a consumers and
Phase 6 rankers see meaningful signal without a UI.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import orjson
from aiokafka import AIOKafkaProducer
from sqlalchemy import select, text

from click_rec.config import get_settings
from click_rec.db.base import dispose_engine, get_sessionmaker
from click_rec.kafka.admin import ensure_topics
from click_rec.kafka.topics import USER_CLICKS, USER_IMPRESSIONS, USER_SEARCHES
from click_rec.models import UserAccount
from scripts.seed_catalog import ADJECTIVES, CATEGORIES

logger = logging.getLogger(__name__)

RANDOM_SEED = 99
CANDIDATES_PER_QUERY = 60  # wider pool so weighted sampling has variation
IMPRESSION_TOP_K = 10
ZIPF_SHAPE = 1.3
RANK_GEOMETRIC_P = 0.4  # smaller → clicks spread further down the list
ITEM_POPULARITY_ALPHA = 1.2  # power-law decay over candidate-pool position
SEND_FLUSH_EVERY = 1000  # backpressure flush so producer buffer can't grow unbounded


def _stable_adjective(path: str) -> str:
    # Python's built-in hash() is salted per-process — md5 keeps the corpus
    # identical across runs so replay output is reproducible.
    digest = hashlib.md5(path.encode()).digest()
    idx = int.from_bytes(digest[:4], "big") % len(ADJECTIVES)
    return ADJECTIVES[idx].lower()


def _build_query_corpus() -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for path, noun, brands, _price in CATEGORIES:
        noun_lower = noun.lower()
        candidates = [
            noun_lower,
            f"{_stable_adjective(path)} {noun_lower}",
            f"{brands[0].lower()} {noun_lower}",
            f"cheap {noun_lower}",
            f"best {noun_lower}",
        ]
        for q in candidates[:4]:
            if q not in seen:
                seen.add(q)
                queries.append(q)
    return queries


def _sample_query_index(rng: np.random.Generator, corpus_size: int) -> int:
    # zipf returns >= 1 with a long tail; clip into [0, corpus_size - 1].
    raw = int(rng.zipf(ZIPF_SHAPE))
    return min(raw - 1, corpus_size - 1)


def _sample_rank(rng: random.Random, top_k: int) -> int:
    # Geometric-ish: most clicks land on ranks 0-2, tail stretches to top_k.
    rank = int(rng.random() ** (1.0 / RANK_GEOMETRIC_P) * top_k)
    return max(0, min(rank, top_k - 1))


def _sample_dwell_ms(rng: random.Random) -> int:
    return max(250, int(np.exp(rng.gauss(8.0, 0.5))))


def _weighted_top_k(
    cands: list[dict[str, Any]], np_rng: np.random.Generator, k: int
) -> list[dict[str, Any]]:
    # Power-law sample over the candidate pool so the same query produces
    # different impressions across sessions while still favouring "popular"
    # items (front of the candidate list).
    n = min(k, len(cands))
    if n == 0:
        return []
    weights = np.fromiter(
        (1.0 / ((i + 1) ** ITEM_POPULARITY_ALPHA) for i in range(len(cands))),
        dtype=np.float64,
        count=len(cands),
    )
    weights /= weights.sum()
    indices = np_rng.choice(len(cands), size=n, replace=False, p=weights)
    return [cands[i] for i in indices]


def _pick_clicks_for_segment(
    segment: str,
    impression: list[dict[str, Any]],
    pinned_brand: str | None,
    num_clicks: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not impression or num_clicks == 0:
        return []

    pool = impression
    if segment == "bargain_hunter":
        ranked = sorted(pool, key=lambda c: c["price"])
    elif segment == "premium":
        ranked = sorted(pool, key=lambda c: c["price"], reverse=True)
    elif segment == "new_arrivals":
        ranked = sorted(pool, key=lambda c: c["created_at"], reverse=True)
    elif segment == "brand_loyalist":
        preferred = [c for c in pool if pinned_brand and c["brand"] == pinned_brand]
        ranked = preferred + [c for c in pool if c not in preferred] if preferred else pool
    else:  # category_explorer — rotate across categories
        by_cat: dict[str, list[dict[str, Any]]] = {}
        for c in pool:
            by_cat.setdefault(c["category"], []).append(c)
        rotated: list[dict[str, Any]] = []
        cats = list(by_cat.values())
        rng.shuffle(cats)
        for bucket in cats:
            rotated.extend(bucket)
        ranked = rotated

    take = min(num_clicks, len(ranked))
    return ranked[:take]


def _encode(value: dict[str, Any]) -> bytes:
    return orjson.dumps(value)


async def _load_catalogue() -> tuple[list[tuple[str, str]], dict[str, list[dict[str, Any]]]]:
    sessionmaker = get_sessionmaker()
    queries = _build_query_corpus()
    candidates: dict[str, list[dict[str, Any]]] = {}

    async with sessionmaker() as session:
        users = (
            await session.execute(
                select(UserAccount.id, UserAccount.segment).order_by(UserAccount.id)
            )
        ).all()
        if not users:
            raise RuntimeError("no users found — run `make seed` first")

        for q in queries:
            result = await session.execute(
                text(
                    "SELECT id, category, brand, price, created_at FROM item "
                    "WHERE tsv @@ plainto_tsquery('english', :q) "
                    "ORDER BY popularity_score DESC, id LIMIT :limit"
                ),
                {"q": q, "limit": CANDIDATES_PER_QUERY},
            )
            rows = [dict(row) for row in result.mappings().all()]
            if not rows:
                # Fallback: any items from any category, so producer still emits.
                fallback = await session.execute(
                    text(
                        "SELECT id, category, brand, price, created_at FROM item "
                        "ORDER BY random() LIMIT :limit"
                    ),
                    {"limit": CANDIDATES_PER_QUERY},
                )
                rows = [dict(row) for row in fallback.mappings().all()]
            candidates[q] = rows

    logger.info(
        "loaded %d users, %d queries (avg %.1f candidates/query)",
        len(users),
        len(queries),
        float(np.mean([len(v) for v in candidates.values()])),
    )
    return [(u.id, u.segment) for u in users], candidates


async def run(num_events: int, capture_eval_log: str | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    # Phase 6: tee click events to a JSONL file so `make eval-offline`
    # can build the replay-gold pass without re-running Kafka. No
    # behaviour change when the flag is absent — the writer is None and
    # the per-click branch becomes a no-op.
    eval_writer = None
    if capture_eval_log:
        eval_path = capture_eval_log
        # Open in append mode so a partial run + resume doesn't lose data.
        import os as _os
        _os.makedirs(_os.path.dirname(eval_path) or ".", exist_ok=True)
        eval_writer = open(eval_path, "a", encoding="utf-8")
        logger.info("eval log capture enabled: %s", eval_path)

    logger.info("ensuring kafka topics on %s...", settings.kafka_bootstrap)
    await ensure_topics()

    users, candidates = await _load_catalogue()
    query_list = list(candidates.keys())

    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    # Scope each loyalist to one category, then pin to a brand from that
    # category's brand list. Pinning across all brands made the pin invisible
    # for users whose Zipf-sampled queries didn't surface the chosen brand.
    category_brands = [(path, brands) for path, _, brands, _ in CATEGORIES]
    pinned_brand: dict[str, str] = {}
    for uid, segment in users:
        if segment != "brand_loyalist":
            continue
        _path, brands = rng.choice(category_brands)
        pinned_brand[uid] = rng.choice(brands)

    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap,
        acks="all",
        enable_idempotence=True,
        linger_ms=5,
        compression_type="lz4",
        value_serializer=_encode,
        key_serializer=lambda s: s.encode("utf-8"),
    )
    await producer.start()

    emitted = 0
    last_log = 0
    try:
        while emitted < num_events:
            user_id, segment = rng.choice(users)
            session_id = f"s_{uuid.uuid4().hex[:12]}"
            queries_in_session = rng.randint(1, 3)
            for _ in range(queries_in_session):
                if emitted >= num_events:
                    break
                q_idx = _sample_query_index(np_rng, len(query_list))
                query = query_list[q_idx]
                cands = candidates[query]
                if not cands:
                    continue

                now = datetime.now(tz=UTC)

                search_event = {
                    "event_id": str(uuid.uuid4()),
                    "event_type": "search",
                    "event_version": 1,
                    "timestamp": now.isoformat(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "query": query,
                    "filters": {},
                }
                await producer.send(USER_SEARCHES.name, search_event, key=user_id)
                emitted += 1
                if emitted >= num_events:
                    break

                top_k = _weighted_top_k(cands, np_rng, IMPRESSION_TOP_K)
                impression_event = {
                    "event_id": str(uuid.uuid4()),
                    "event_type": "impression",
                    "event_version": 1,
                    "timestamp": now.isoformat(),
                    "user_id": user_id,
                    "session_id": session_id,
                    "query": query,
                    "result_ids": [c["id"] for c in top_k],
                    "page": 1,
                }
                await producer.send(
                    USER_IMPRESSIONS.name, impression_event, key=user_id
                )
                emitted += 1

                num_clicks = rng.choices([0, 1, 2, 3], weights=[1, 3, 2, 1], k=1)[0]
                clicks = _pick_clicks_for_segment(
                    segment, top_k, pinned_brand.get(user_id), num_clicks, rng
                )
                for clicked in clicks:
                    if emitted >= num_events:
                        break
                    # rank_position must reflect where the item was actually
                    # shown, not an independent sample — Phase 6 uses this as a
                    # CTR feature and a lying field would train on noise.
                    rank_position = next(
                        (i for i, c in enumerate(top_k) if c["id"] == clicked["id"]),
                        -1,
                    )
                    if rank_position < 0:
                        rank_position = _sample_rank(rng, len(top_k))
                    click_event = {
                        "event_id": str(uuid.uuid4()),
                        "event_type": "click",
                        "event_version": 1,
                        "timestamp": now.isoformat(),
                        "user_id": user_id,
                        "session_id": session_id,
                        "query": query,
                        "item_id": clicked["id"],
                        "rank_position": rank_position,
                        "dwell_ms": _sample_dwell_ms(rng),
                    }
                    await producer.send(
                        USER_CLICKS.name, click_event, key=user_id
                    )
                    if eval_writer is not None:
                        eval_writer.write(orjson.dumps(click_event).decode() + "\n")
                    emitted += 1

            if emitted - last_log >= SEND_FLUSH_EVERY:
                # Bound producer buffer growth and surface broker errors early.
                await producer.flush()
                # Flush the eval log on the same cadence so a mid-run
                # crash doesn't truncate the JSONL — the eval harness
                # silently runs against a partial set otherwise.
                if eval_writer is not None:
                    eval_writer.flush()
                logger.info("produced %d / %d events", emitted, num_events)
                last_log = emitted
    finally:
        await producer.flush()
        await producer.stop()
        await dispose_engine()
        if eval_writer is not None:
            eval_writer.close()

    logger.info("replay complete: %d events emitted", emitted)
    return 0


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    raise SystemExit(asyncio.run(run(count)))
