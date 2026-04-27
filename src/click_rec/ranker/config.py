"""Tunable knobs for the hybrid ranker.

Resolution order in `load_ranker_config()`:

1. Explicit `path=` argument.
2. `RANKER_CONFIG` env var.
3. `<repo>/ranker.yaml` (or `settings.ranker_config_path`).
4. Dataclass defaults.

YAML parse failures log a warning and fall back to defaults; the loader
never raises at request path. Mirrors the singleton + `lru_cache` pattern
in `click_rec.config.get_settings`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from click_rec.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RankerConfig:
    # Candidate generation knobs.
    bm25_k: int = 100
    vector_k: int = 100
    candidate_cap: int = 200

    # Feature parameters.
    recency_half_life_days: float = 30.0
    personal_recent_clicks_n: int = 20
    price_fit_scale: float = 200.0  # dollars

    # Per-feature weights (linear scorer).
    w_bm25: float = 1.0
    w_vector: float = 1.0
    w_popularity: float = 0.6
    w_recency: float = 0.4
    w_personal: float = 1.2
    w_co_click: float = 0.8
    w_price_fit: float = 0.3


def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten the YAML schema (`weights.bm25` etc.) into RankerConfig fields."""
    flat: dict[str, Any] = {}
    candidates = raw.get("candidates") or {}
    features = raw.get("features") or {}
    weights = raw.get("weights") or {}

    if "bm25_k" in candidates:
        flat["bm25_k"] = int(candidates["bm25_k"])
    if "vector_k" in candidates:
        flat["vector_k"] = int(candidates["vector_k"])
    if "candidate_cap" in candidates:
        flat["candidate_cap"] = int(candidates["candidate_cap"])

    if "recency_half_life_days" in features:
        flat["recency_half_life_days"] = float(features["recency_half_life_days"])
    if "personal_recent_clicks_n" in features:
        flat["personal_recent_clicks_n"] = int(features["personal_recent_clicks_n"])
    if "price_fit_scale" in features:
        flat["price_fit_scale"] = float(features["price_fit_scale"])

    weight_keys = (
        "bm25",
        "vector",
        "popularity",
        "recency",
        "personal",
        "co_click",
        "price_fit",
    )
    for k in weight_keys:
        if k in weights:
            flat[f"w_{k}"] = float(weights[k])

    return flat


def _resolve_path(path: str | None) -> Path | None:
    if path is not None:
        return Path(path)
    env_path = os.environ.get("RANKER_CONFIG")
    if env_path:
        return Path(env_path)
    settings_path = Path(get_settings().ranker_config_path)
    return settings_path


@lru_cache(maxsize=4)
def load_ranker_config(path: str | None = None) -> RankerConfig:
    """Load `RankerConfig` from YAML, falling back to defaults on any error."""
    resolved = _resolve_path(path)
    if resolved is None or not resolved.exists():
        if resolved is not None:
            logger.info(
                "ranker config file not found, using defaults",
                extra={"path": str(resolved)},
            )
        return RankerConfig()

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError("ranker.yaml must be a mapping at the top level")
        flat = _flatten_yaml(raw)
        # Drop unknown keys defensively rather than raising — any rename
        # in the dataclass should never 5xx the request path.
        valid = {f.name for f in fields(RankerConfig)}
        clean = {k: v for k, v in flat.items() if k in valid}
        return RankerConfig(**clean)
    except Exception as exc:  # noqa: BLE001 — fall back, never raise here.
        logger.warning(
            "ranker config parse failed, using defaults",
            extra={"path": str(resolved), "error": str(exc)},
        )
        return RankerConfig()
