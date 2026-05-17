"""Tunables for the Phase 7 LLM layer.

Mirrors `click_rec.ranker.config.load_ranker_config`'s pattern: a frozen
dataclass plus an `lru_cache`-d loader that flattens YAML into fields.
Sharing `ranker.yaml` (rather than introducing `llm.yaml`) keeps a single
source of truth for runtime knobs and avoids two file-watch surfaces.
Defaults match the dataclass — a missing `llm:` section silently uses
them, never raises at request path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from click_rec.config import get_settings
from click_rec.ranker.config import _resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMConfig:
    query_understanding_timeout: float = 1.5
    query_understanding_cache_ttl: int = 300
    re_rank_timeout: float = 2.0
    re_rank_top_k: int = 10
    re_rank_input_k: int = 20
    explain_timeout: float = 3.0
    explain_cache_ttl: int = 3600
    judge_timeout: float = 10.0
    judge_sample_default: int = 10
    # Free-tier `gemini-2.5-pro` is ~5 RPM; 13s between calls keeps us
    # under the limit with a small safety margin. Set to 0 in tests /
    # paid-tier deployments where bursting is fine.
    judge_pacing_seconds: float = 13.0


def _flatten_yaml(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten the nested `llm:` YAML section into LLMConfig fields."""
    llm = raw.get("llm") or {}
    flat: dict[str, Any] = {}

    qu = llm.get("query_understanding") or {}
    if "timeout_seconds" in qu:
        flat["query_understanding_timeout"] = float(qu["timeout_seconds"])
    if "cache_ttl_seconds" in qu:
        flat["query_understanding_cache_ttl"] = int(qu["cache_ttl_seconds"])

    rr = llm.get("re_rank") or {}
    if "timeout_seconds" in rr:
        flat["re_rank_timeout"] = float(rr["timeout_seconds"])
    if "top_k" in rr:
        flat["re_rank_top_k"] = int(rr["top_k"])
    if "input_k" in rr:
        flat["re_rank_input_k"] = int(rr["input_k"])

    ex = llm.get("explain") or {}
    if "timeout_seconds" in ex:
        flat["explain_timeout"] = float(ex["timeout_seconds"])
    if "cache_ttl_seconds" in ex:
        flat["explain_cache_ttl"] = int(ex["cache_ttl_seconds"])

    jd = llm.get("judge") or {}
    if "timeout_seconds" in jd:
        flat["judge_timeout"] = float(jd["timeout_seconds"])
    if "sample_default" in jd:
        flat["judge_sample_default"] = int(jd["sample_default"])
    if "pacing_seconds" in jd:
        flat["judge_pacing_seconds"] = float(jd["pacing_seconds"])

    return flat


@lru_cache(maxsize=4)
def load_llm_config(path: str | None = None) -> LLMConfig:
    """Load `LLMConfig` from `ranker.yaml`'s `llm:` section.

    Falls back to dataclass defaults on any error. Mirrors
    `load_ranker_config`'s never-raise contract.
    """
    resolved: Path | None = _resolve_path(path)
    if resolved is None or not resolved.exists():
        return LLMConfig()
    try:
        with resolved.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError("ranker.yaml must be a mapping at the top level")
        flat = _flatten_yaml(raw)
        valid = {f.name for f in fields(LLMConfig)}
        clean = {k: v for k, v in flat.items() if k in valid}
        return LLMConfig(**clean)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm config parse failed, using defaults",
            extra={"path": str(resolved), "error": str(exc)},
        )
        return LLMConfig()


def runtime_model() -> str:
    return get_settings().gemini_runtime_model


def judge_model() -> str:
    return get_settings().gemini_judge_model
