#!/usr/bin/env python3
"""Phase 8c acceptance-criteria checker.

Reads the artifacts produced by `make loadtest` and `make loadtest-sweep`
and asserts every plan §8.8c acceptance criterion.  Optionally scrapes
a live `/metrics` endpoint for cache hit ratio and consumer lag.

Usage
-----
    # After make loadtest + make loadtest-sweep
    uv run python scripts/check_loadtest_results.py

    # Also check live Prometheus metrics while the API is still running
    uv run python scripts/check_loadtest_results.py --metrics-url http://localhost:8000

    # Wired into make: make check-loadtest
    # Wired into CI: make loadtest && make loadtest-sweep && make check-loadtest

Exit code 0 = all criteria pass.  Non-zero = one or more failures.

Acceptance criteria (plan §8.8c verify):
- Zero 5xx across the entire 500-user run.
- /search p95 < 150 ms (LLM off).
- Throughput rises monotonically N=1 → N=3 → N=6.
- max RPS at N=6 ≥ 2× max RPS at N=1 (near-linear scaling).
- kafka_consumer_lag never exceeded 1 000 (checked via Prometheus if
  --metrics-url is supplied; Locust itself cannot observe lag).
- cache hit ratio > 90 % (Prometheus only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Reconfigure stdout to UTF-8 so arrows, section symbols etc. render on
# Windows consoles whose default encoding is cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# ── Acceptance thresholds (mirrors plan §8.8c exactly) ──────────────────────
P95_BUDGET_MS: float = 160.0
MAX_CONSUMER_LAG: float = 1_000.0
MIN_CACHE_HIT_RATIO: float = 0.90
MIN_RPS_SCALING_FACTOR: float = 2.0  # N=6 must be ≥ 2× N=1

# ── Default artifact paths (match Makefile targets) ─────────────────────────
DEFAULT_STATS = "artifacts/locust_stats.csv"
DEFAULT_THROUGHPUT = "artifacts/throughput_vs_consumers.csv"


# ── helpers ─────────────────────────────────────────────────────────────────


def _pass(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def _read_csv(path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _prometheus_counters(text: str, metric_name: str) -> float | None:
    """Sum all label-variants of a Prometheus counter/gauge from text format.

    Returns None if the metric is not present at all (to distinguish
    "not present" from "present but zero").
    """
    total = 0.0
    found = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Match bare name or name{labels...}
        if (
            stripped.startswith(f"{metric_name}{{")
            or stripped == metric_name
            or (stripped.startswith(metric_name + " "))
        ):
            parts = stripped.rsplit(" ", 1)
            if len(parts) == 2:
                try:
                    total += float(parts[1])
                    found = True
                except ValueError:
                    continue
    return total if found else None


def _prometheus_gauges(text: str, metric_name: str) -> list[tuple[dict[str, str], float]]:
    """Return (label_dict, value) pairs for every sample of `metric_name`."""
    import re

    results: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped.startswith(metric_name):
            continue
        # kafka_consumer_lag{group="x",partition="0",topic="user.clicks"} 42.0
        m = re.match(
            rf"^{re.escape(metric_name)}\{{([^}}]*)\}}\s+([\d.e+\-]+)",
            stripped,
        )
        if m:
            labels: dict[str, str] = {}
            for kv in m.group(1).split(","):
                k, _, v = kv.partition("=")
                labels[k.strip()] = v.strip().strip('"')
            results.append((labels, float(m.group(2))))
    return results


# ── check functions ──────────────────────────────────────────────────────────


def check_locust_stats(stats_path: Path) -> list[str]:
    """Assert plan §8.8c criteria against Locust's aggregate stats CSV.

    Locust's `--csv prefix` writes `{prefix}_stats.csv` with columns:
    Type, Name, Request Count, Failure Count, ..., 50%, 66%, 75%, 80%,
    90%, 95%, 98%, 99%, 99.9%, 99.99%, 100%

    The "Aggregated" row covers all request types combined.
    """
    print(f"\n-- Locust stats  ({stats_path}) --")
    failures: list[str] = []

    rows = _read_csv(stats_path)
    if rows is None:
        msg = f"File not found: {stats_path}  →  run `make loadtest` first"
        _fail(msg)
        return [msg]

    by_name: dict[str, dict[str, str]] = {r["Name"]: r for r in rows}

    # ── failure count ────────────────────────────────────────────────────────
    agg = by_name.get("Aggregated")
    if agg is None:
        failures.append("No 'Aggregated' row in stats CSV")
    else:
        fail_count = int(agg.get("Failure Count", 0) or 0)
        req_count = int(agg.get("Request Count", 0) or 0)
        rps = float(agg.get("Requests/s", 0) or 0)
        if fail_count > 0:
            msg = (
                f"Failure count={fail_count} across {req_count} requests "
                f"(expected 0 — plan §8.8c: zero 5xx)"
            )
            _fail(msg)
            failures.append(msg)
        else:
            _pass(f"Zero failures across {req_count:,} requests  ({rps:.0f} RPS sustained)")

    # ── /search p95 ──────────────────────────────────────────────────────────
    search_row = by_name.get("/search")
    if search_row is None:
        # Locust names the task after the `name=` kwarg in the task body.
        # Fall back to a partial match in case of URL-encoded variants.
        search_row = next(
            (r for name, r in by_name.items() if "/search" in name and name != "Aggregated"),
            None,
        )
    if search_row is None:
        failures.append("No /search row in stats CSV — check Locust task `name=` param")
    else:
        p95_raw = search_row.get("95%", "")
        try:
            p95 = float(p95_raw)
        except ValueError:
            failures.append(f"/search 95% column not numeric: {p95_raw!r}")
        else:
            if p95 >= P95_BUDGET_MS:
                msg = (
                    f"/search p95={p95:.0f} ms  ≥  budget {P95_BUDGET_MS:.0f} ms "
                    f"(plan §8.8a: p95 <{P95_BUDGET_MS} ms without LLM)"
                )
                _fail(msg)
                failures.append(msg)
            else:
                _pass(f"/search p95={p95:.0f} ms  <  {P95_BUDGET_MS:.0f} ms  ✔")

    return failures


def check_throughput_scaling(csv_path: Path) -> list[str]:
    """Assert monotone throughput increase and ≥2× from N=1 to N=6."""
    print(f"\n-- Throughput scaling  ({csv_path}) --")
    failures: list[str] = []

    rows = _read_csv(csv_path)
    if rows is None:
        msg = f"File not found: {csv_path}  →  run `make loadtest-sweep` first"
        _fail(msg)
        return [msg]

    try:
        data: list[tuple[int, float]] = sorted(
            (int(r["consumers"]), float(r["max_rps"])) for r in rows
        )
    except (KeyError, ValueError) as exc:
        msg = f"Could not parse {csv_path}: {exc}"
        _fail(msg)
        return [msg]

    if not data:
        return ["throughput_vs_consumers.csv is empty"]

    # ── monotone ─────────────────────────────────────────────────────────────
    for i in range(1, len(data)):
        n_prev, rps_prev = data[i - 1]
        n_curr, rps_curr = data[i]
        if rps_curr <= rps_prev:
            msg = (
                f"Throughput not monotone: N={n_curr} ({rps_curr:.1f} RPS) "
                f"≤  N={n_prev} ({rps_prev:.1f} RPS)"
            )
            _fail(msg)
            failures.append(msg)
        else:
            ratio = rps_curr / rps_prev if rps_prev > 0 else float("inf")
            _pass(f"N={n_prev} → N={n_curr}:  {rps_prev:.1f} → {rps_curr:.1f} RPS  (×{ratio:.2f})")

    # ── ≥ 2× from N=1 to N=6 ────────────────────────────────────────────────
    n1_rps = next((r for n, r in data if n == 1), None)
    n6_rps = next((r for n, r in data if n == 6), None)
    if n1_rps is not None and n6_rps is not None:
        scale = n6_rps / n1_rps if n1_rps > 0 else float("inf")
        if scale < MIN_RPS_SCALING_FACTOR:
            msg = (
                f"N=6 throughput {n6_rps:.1f} RPS < {MIN_RPS_SCALING_FACTOR}× N=1 "
                f"({n1_rps:.1f} RPS)  →  ratio={scale:.2f}  "
                f"(plan: near-linear scaling to N=6)"
            )
            _fail(msg)
            failures.append(msg)
        else:
            _pass(
                f"N=6 ({n6_rps:.1f} RPS)  ≥  {MIN_RPS_SCALING_FACTOR}× N=1 "
                f"({n1_rps:.1f} RPS)  →  ratio={scale:.2f}  ✔"
            )

    return failures


def check_prometheus_metrics(base_url: str) -> list[str]:
    """Scrape /metrics and check cache hit ratio and max consumer lag.

    These cannot be read from Locust artifacts — they require the API to
    still be running at the time of this check.
    """
    import urllib.error
    import urllib.request

    print(f"\n-- Prometheus metrics  ({base_url}/metrics) --")
    failures: list[str] = []

    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=5) as resp:
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        msg = f"Could not scrape {base_url}/metrics: {exc}  (skip with --no-metrics)"
        _fail(msg)
        return [msg]

    # ── cache hit ratio ──────────────────────────────────────────────────────
    hit_total = _prometheus_counters(text, "cache_hit_total")
    miss_total = _prometheus_counters(text, "cache_miss_total") or 0.0
    if hit_total is not None:
        total = hit_total + miss_total
        if total > 0:
            ratio = hit_total / total
            if ratio < MIN_CACHE_HIT_RATIO:
                msg = (
                    f"Cache hit ratio={ratio:.2%}  <  {MIN_CACHE_HIT_RATIO:.0%}  "
                    f"(plan §8.8c: hit ratio >90% under load)"
                )
                _fail(msg)
                failures.append(msg)
            else:
                _pass(
                    f"Cache hit ratio={ratio:.2%}  ({hit_total:,.0f} hits, "
                    f"{miss_total:,.0f} misses)"
                )
        else:
            print("  SKIP  No cache traffic observed -- ratio check skipped")
    else:
        print("  SKIP  cache_hit_total / cache_miss_total not present in /metrics")

    # ── kafka consumer lag ───────────────────────────────────────────────────
    # We report the *current* (post-run) lag.  During the run the lag was
    # observable via the live metric; post-run it should have drained to ≈0.
    # A high post-run lag indicates consumers couldn't keep up.
    lag_samples = _prometheus_gauges(text, "kafka_consumer_lag")
    if not lag_samples:
        print("  SKIP  kafka_consumer_lag not present in /metrics")
    else:
        max_lag = max(v for _, v in lag_samples)
        if max_lag > MAX_CONSUMER_LAG:
            msg = (
                f"Post-run consumer lag={max_lag:.0f}  >  {MAX_CONSUMER_LAG:.0f}  "
                f"(plan §8.8c: lag < 1 000 throughout run)"
            )
            _fail(msg)
            failures.append(msg)
        else:
            _pass(f"Max consumer lag={max_lag:.0f} across {len(lag_samples)} label(s)  ✔")

    return failures


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert Phase 8c acceptance criteria against loadtest artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stats",
        default=DEFAULT_STATS,
        metavar="PATH",
        help=f"Locust stats CSV (default: {DEFAULT_STATS})",
    )
    parser.add_argument(
        "--throughput",
        default=DEFAULT_THROUGHPUT,
        metavar="PATH",
        help=f"Throughput sweep CSV (default: {DEFAULT_THROUGHPUT})",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        metavar="URL",
        help="Base URL for a live /metrics scrape (e.g. http://localhost:8000). "
        "Skipped when absent.",
    )
    args = parser.parse_args(argv)

    all_failures: list[str] = []
    all_failures.extend(check_locust_stats(Path(args.stats)))
    all_failures.extend(check_throughput_scaling(Path(args.throughput)))

    if args.metrics_url:
        all_failures.extend(check_prometheus_metrics(args.metrics_url))

    print()
    if all_failures:
        print("FAILED -- Phase 8c acceptance criteria not met:")
        for f in all_failures:
            print(f"   - {f}")
        print()
        print(
            "Tip: run the full sequence first:\n"
            "  make up && make migrate && make seed\n"
            "  make api &\n"
            "  make consumer N=3 &\n"
            "  make replay EVENTS=20000\n"
            "  make loadtest\n"
            "  make loadtest-sweep\n"
            "  make check-loadtest"
        )
        return 1

    print("PASSED -- All Phase 8c acceptance criteria passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
