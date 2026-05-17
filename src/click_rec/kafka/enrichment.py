"""Side-effects applied to a single click event.

Pure transforms over (event, db_session, redis, producer). Order is
chosen to *minimise drift* on partial-failure retries — the most common
failure mode (Postgres commit blowing up: deadlock, dropped connection,
constraint violation) must not have already executed any non-idempotent
side-effect.

Execution order:

1. Resolve item metadata (category, price) — read-only.
2. Read the user's prior session items from Redis — read-only.
3. Bulk-upsert co-click pairs over the last `session_window` items —
   *including the current click* (Phase 4 review C1) — and **commit the
   DB transaction**. This is the failure-prone step; if it raises,
   nothing else has run yet, so a retry incurs zero drift.
4. Update `recent_clicks` (Redis ZADD — naturally idempotent on
   (member, score), no drift on retry).
5. Increment per-category popularity counter (Redis INCR — NOT
   idempotent; drift bounded to the rare DB-commit-OK + Redis-fails path).
6. Fan out a `user.profile.updates` event for downstream consumers
   (Kafka — drift on retry is one extra message, downstream is idempotent).

Idempotency per side-effect — what survives a replay and what doesn't:

- ZADD `recent_clicks` with `score=ts` is truly idempotent; a replay of
  the same (member, score) is a no-op.
- co_click `count + excluded.count` and popularity INCR are
  **accumulators**, not idempotent. A replay that re-runs this function
  inflates both by 1 per replay. The consumer-side dedupe marker in
  `_process_one` short-circuits the common replay path (same message
  redelivered before commit), so drift only leaks when steps 4–6 fail
  *after* the DB commit at step 3 succeeds (Redis network blip, Kafka
  broker blip). That drift is bounded by `consumer_max_retries` ×
  rebalance rate, and Phase 5's counter decay compresses whatever
  leaks through. The `+ excluded.count` accumulator in
  `upsert_co_clicks` is deliberate — see its docstring.

  **TODO (Phase 8c):** quantify drift under load. If a publish-failure
  retry visibly inflates co_click / popularity counters, split the
  Kafka publish into its own retry-bounded block so a Kafka-only failure
  doesn't redo steps 3–5. Premature today — `enable_idempotence=True`
  + healthy broker makes publish failures essentially zero in steady
  state.

Per Phase 4 review C4(a), the consumer "processed" marker is set in
`consumer._process_one` *after* this function returns successfully.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import redis.asyncio as redis
from aiokafka import AIOKafkaProducer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from click_rec.config import get_settings
from click_rec.kafka.topics import USER_PROFILE_UPDATES
from click_rec.models.co_click import CoClick
from click_rec.models.item import Item
from click_rec.models.schemas import ClickEventDTO

logger = logging.getLogger(__name__)


# Process-local item metadata cache: item_id -> (category, price).
# Tests must clear this between cases — see the autouse fixture in
# tests/test_consumer_unit.py (Phase 4 review M1).
_item_cache: dict[str, tuple[str, float]] = {}

# Companion negative cache: item_ids known to be missing from the DB.
# Without this, a flood of clicks on a deleted/unknown item_id (producer
# bug, stale catalog, A/B test) hits Postgres on every event because the
# positive cache stays empty. A separate set is needed because
# `dict.get(k)` cannot distinguish "missing key" from "key with value
# None". Capped + cleared by the same M1 autouse fixture.
_item_negative_cache: set[str] = set()


def _evict_if_full(cache: dict[str, tuple[str, float]], cap: int) -> None:
    if len(cache) >= cap:
        # Cheap FIFO eviction — drop ~10% to amortise the work. We don't
        # need true LRU here; the cache is a hot-path optimisation, not a
        # correctness boundary.
        for key in list(cache.keys())[: max(1, cap // 10)]:
            cache.pop(key, None)


def _evict_negative_if_full(cache: set[str], cap: int) -> None:
    if len(cache) >= cap:
        # Same FIFO-ish strategy as the positive cache. `set` iteration
        # order is insertion order in CPython 3.7+ for non-rehash cases;
        # we don't need true FIFO here, just bounded memory.
        for key in list(cache)[: max(1, cap // 10)]:
            cache.discard(key)


async def resolve_item_meta(session: AsyncSession, item_id: str) -> tuple[str, float] | None:
    """Return (category, price) for `item_id`, cached process-locally.

    Negative lookups (item not found) are also cached so a flood of bad
    item_ids doesn't pound Postgres. Negative entries clear on process
    restart, which is the right TTL for a portfolio system — if an item
    is later inserted, a deploy or restart picks it up.
    """
    if item_id in _item_negative_cache:
        return None
    cached = _item_cache.get(item_id)
    if cached is not None:
        return cached

    settings = get_settings()
    row = (
        await session.execute(select(Item.category, Item.price).where(Item.id == item_id))
    ).first()
    if row is None:
        _evict_negative_if_full(_item_negative_cache, settings.item_cache_max_size)
        _item_negative_cache.add(item_id)
        return None
    category, price = row.category, float(row.price)
    _evict_if_full(_item_cache, settings.item_cache_max_size)
    _item_cache[item_id] = (category, price)
    return category, price


async def get_last_session_items(redis_client: redis.Redis, user_id: str, n: int) -> list[str]:
    """Return up to `n` most-recent item_ids for a user (newest first)."""
    raw = await redis_client.zrevrange(f"user:{user_id}:recent_clicks", 0, n - 1)
    return [b.decode() if isinstance(b, bytes) else b for b in raw]


async def update_recent_clicks(
    redis_client: redis.Redis, user_id: str, item_id: str, ts_ms: float
) -> None:
    """ZADD + ZREMRANGEBYRANK to keep only the last `recent_clicks_cap` items.

    ZADD is naturally idempotent on (member, score) replays — the same
    `event_id` arriving twice updates the score in place rather than
    inserting a duplicate. Refreshing the TTL on every update keeps the
    set warm for active users.
    """
    settings = get_settings()
    key = f"user:{user_id}:recent_clicks"
    pipe = redis_client.pipeline(transaction=False)
    pipe.zadd(key, {item_id: ts_ms})
    # Drop everything except the top N (highest score = most recent).
    pipe.zremrangebyrank(key, 0, -(settings.recent_clicks_cap + 1))
    pipe.expire(key, settings.recent_clicks_ttl_seconds)
    await pipe.execute()


async def increment_popularity(redis_client: redis.Redis, category: str, item_id: str) -> None:
    """Bump the per-category counter, per-category sorted set, and the
    global sorted set.

    Three writes, one pipeline:

    - `popularity:{category}` — scalar counter, TTL-bounded so memory
      doesn't grow forever. Used by ranking for hot-category signals.
    - `items:top:{category}` — Phase 5 sorted set of item_ids scored by
      decayed click count. The refresher task trims + decays it every
      `cache_refresh_interval_seconds`; we deliberately don't set a TTL
      here because every click would reset it — the refresher owns it.
    - `items:top:_global` — Phase 8a global sorted set used as the
      hard-cold-start candidate pool by `/recommendations`. The
      refresher task (which scans `items:top:*`) trims and decays this
      key on the same cadence as the per-category ones.
    """
    settings = get_settings()
    counter_key = f"popularity:{category}"
    top_key = f"items:top:{category}"
    global_top_key = "items:top:_global"
    pipe = redis_client.pipeline(transaction=False)
    pipe.incr(counter_key)
    pipe.expire(counter_key, settings.popularity_ttl_seconds)
    pipe.zincrby(top_key, 1, item_id)
    pipe.zincrby(global_top_key, 1, item_id)
    await pipe.execute()


async def upsert_co_clicks(session: AsyncSession, item_ids: list[str]) -> None:
    """Bulk upsert all unordered pairs from `item_ids` into `co_click`.

    Phase 4 review D2: a 5-item session yields C(5,2)=10 pairs. The prior
    plan executed one INSERT per pair (10 round-trips per click event). At
    500 RPS that's 5,000 extra statements/sec for co-click maintenance
    alone. Single batched insert + ON CONFLICT collapses it to one
    statement per click. `dict.fromkeys` deduplicates the input first to
    avoid (X, X) self-pairs when the same item appears twice in a session.

    Note on replay: the ON CONFLICT clause is `count = count + 1`, i.e.
    an accumulator — not `count = GREATEST(count, 1)` or a no-op. A
    replay of the same event re-increments every pair. This is
    deliberate; the consumer dedupe marker handles the happy-path
    replay, and the residual drift is bounded and decayed in Phase 5.
    See the module docstring for the full idempotency model.
    """
    unique = list(dict.fromkeys(item_ids))
    if len(unique) < 2:
        return
    pairs = sorted({(min(a, b), max(a, b)) for a, b in itertools.combinations(unique, 2)})
    if not pairs:
        return
    stmt = pg_insert(CoClick).values([{"item_a": a, "item_b": b, "count": 1} for a, b in pairs])
    stmt = stmt.on_conflict_do_update(
        index_elements=["item_a", "item_b"],
        set_={"count": CoClick.count + stmt.excluded.count},
    )
    await session.execute(stmt)


async def publish_profile_update(
    producer: AIOKafkaProducer,
    user_id: str,
    event_id: str,
    item_id: str,
    category: str,
) -> None:
    """Fan out a `user.profile.updates` event for downstream consumers.

    Plan §6 calls for this topic to feed A/B harnesses + the future
    profile-builder. Phase 4a only emits; consumption is Phase 6.
    """
    payload: dict[str, Any] = {
        "event_id": event_id,
        "user_id": user_id,
        "item_id": item_id,
        "category": category,
        "source": "click",
    }
    await producer.send_and_wait(USER_PROFILE_UPDATES.name, payload, key=user_id)


async def apply_enrichment(
    *,
    event: ClickEventDTO,
    session: AsyncSession,
    redis_client: redis.Redis,
    producer: AIOKafkaProducer,
) -> None:
    """Run all side-effects for one click event.

    Side-effect order is intentional — see the module docstring. The
    short version: do the failure-prone work first (DB commit) so retries
    on the most common failure mode incur zero drift.

    Phase 4 review fixes folded in:
    - C1: the current click is prepended to the session window so a
      6th click never gets sliced out of co-click pairs.
    - D1: no `user:{id}:profile` hash write — that's Phase 6 territory.
    - D2: bulk co-click upsert (one statement, not C(N,2)).
    """
    settings = get_settings()
    meta = await resolve_item_meta(session, event.item_id)
    if meta is None:
        logger.warning(
            "enrichment skipped: item not found",
            extra={"event_id": str(event.event_id), "item_id": event.item_id},
        )
        return
    category, _price = meta
    ts_ms = event.timestamp.timestamp() * 1000.0

    prior_items = await get_last_session_items(redis_client, event.user_id, settings.session_window)
    # C1: current click first, then prior items, then truncate. Without
    # this, prior_items already at length N pushes the current click out
    # of the slice.
    session_items = ([event.item_id] + prior_items)[: settings.session_window]

    # 1. Postgres first — the failure-prone step. If it raises, nothing
    #    else has run yet, so the retry replays cleanly with zero drift.
    await upsert_co_clicks(session, session_items)
    await session.commit()

    # 2. Redis next — ZADD is naturally idempotent; INCR drift is now
    #    only possible on the rare "DB committed but Redis failed" path.
    await update_recent_clicks(redis_client, event.user_id, event.item_id, ts_ms)
    await increment_popularity(redis_client, category, event.item_id)

    # 3. Kafka fan-out last — downstream is idempotent on event_id, so a
    #    retry-induced duplicate message is harmless.
    await publish_profile_update(
        producer,
        user_id=event.user_id,
        event_id=str(event.event_id),
        item_id=event.item_id,
        category=category,
    )
