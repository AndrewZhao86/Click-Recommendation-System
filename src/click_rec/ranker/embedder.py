"""Process-local MiniLM singleton for query encoding.

Lazy-loads on first call (~1s cold-start) and reuses the model
afterwards. Encoding runs on the asyncio default executor so the event
loop isn't blocked for the ~5ms forward pass.

`warm()` is a one-shot pre-load wired into the FastAPI lifespan so the
first user request never pays the cold-load tax. Without it, a `curl`
during verification step 2 reports a misleading ~1s number.

Matches the catalog seeder's `normalize_embeddings=False` setting at
[seed_catalog.py:190](../../../scripts/seed_catalog.py) — the IVFFLAT
index uses cosine ops, so encoded vectors are compared via the `<=>`
operator rather than dot product, and consistent normalisation between
query and index is what we actually need (both unnormalised).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from click_rec.config import get_settings

logger = logging.getLogger(__name__)

_model: Any = None
_load_lock: asyncio.Lock | None = None


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
        _model = await loop.run_in_executor(
            None, SentenceTransformer, settings.embedding_model
        )
        return _model


async def encode_query(query: str) -> list[float]:
    """Encode a single query string into a 384-dim vector.

    Matches catalog encoding: `normalize_embeddings=False`. Cosine
    semantics in the vector channel come from pgvector's `<=>` operator,
    not from L2-normalising both sides ourselves.
    """
    model = await _load_model()
    loop = asyncio.get_running_loop()

    def _encode_blocking() -> list[float]:
        vec = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )[0]
        result: list[float] = vec.tolist()
        return result

    out: list[float] = await loop.run_in_executor(None, _encode_blocking)
    return out


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
