"""Unit tests for `RankerConfig` + `load_ranker_config`.

The loader is `lru_cache`-d, so each test must clear the cache to see
its monkeypatched env / fixture file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from click_rec.config import get_settings
from click_rec.ranker.config import RankerConfig, load_ranker_config


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    load_ranker_config.cache_clear()
    get_settings.cache_clear()


def test_defaults_when_no_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANKER_CONFIG", raising=False)
    monkeypatch.setenv("RANKER_CONFIG_PATH", str(tmp_path / "missing.yaml"))
    cfg = load_ranker_config()
    assert cfg == RankerConfig()


def test_loads_from_explicit_path(tmp_path: Path) -> None:
    yaml_path = tmp_path / "ranker.yaml"
    yaml_path.write_text(
        """
candidates:
  bm25_k: 50
  vector_k: 75
weights:
  bm25: 2.5
  personal: 0.0
"""
    )
    cfg = load_ranker_config(str(yaml_path))
    assert cfg.bm25_k == 50
    assert cfg.vector_k == 75
    assert cfg.w_bm25 == 2.5
    assert cfg.w_personal == 0.0
    # Other fields fall back to dataclass defaults.
    assert cfg.w_vector == RankerConfig().w_vector


def test_invalid_yaml_falls_back_to_defaults(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("this is: : :\n  - not - valid")
    cfg = load_ranker_config(str(yaml_path))
    assert cfg == RankerConfig()
    assert any("ranker config parse failed" in rec.message for rec in caplog.records)


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    yaml_path = tmp_path / "ranker.yaml"
    yaml_path.write_text(
        """
weights:
  bm25: 1.5
  unknown_weight: 99.0  # should be silently dropped
"""
    )
    cfg = load_ranker_config(str(yaml_path))
    assert cfg.w_bm25 == 1.5
    # No exception, no AttributeError surface.
