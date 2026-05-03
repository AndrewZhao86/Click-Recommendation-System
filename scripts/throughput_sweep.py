#!/usr/bin/env python3
"""Phase 8c — scalability proof: max RPS at N=1, 3, 6 consumers.

Cross-platform port of `throughput_sweep.sh`. The original used `pkill
-f` which doesn't exist on Windows, so the sweep could never run on
the user's host. This version manages the consumer pool via
`subprocess.Popen` and tears it down by PID, so it works on any host
that can run the rest of the toolchain.

Outputs `artifacts/throughput_vs_consumers.csv` with one row per N.
The file format matches what `scripts/check_loadtest_results.py`
already expects.
"""

from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
OUT = ARTIFACTS / "throughput_vs_consumers.csv"
SWEEP_NS: list[int] = [1, 3, 6]
WARMUP_S = 8.0  # let workers join the group + warm caches
LOCUST_USERS = 200
LOCUST_SPAWN = 40
LOCUST_RUN = "5m"


def _max_rps_from_stats(stats_csv: Path) -> float:
    """Take the max `Requests/s` across all rows of a Locust stats CSV.

    Locust writes one row per request type plus an `Aggregated` row;
    the steady-state max RPS is captured by the Aggregated value, but
    we take the overall max so the answer is robust to Locust column
    ordering / header capitalisation drift between versions.
    """
    if not stats_csv.exists():
        return 0.0
    with stats_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    best = 0.0
    for r in rows:
        for k, v in r.items():
            if k and k.strip().lower() == "requests/s":
                try:
                    best = max(best, float(v))
                except (TypeError, ValueError):
                    pass
    return best


def _spawn_consumers(n: int, log_path: Path) -> subprocess.Popen[bytes]:
    log_fh = log_path.open("wb")
    return subprocess.Popen(
        ["uv", "run", "python", "-m", "click_rec.cli", "consumer", "--workers", str(n)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    """Best-effort kill that works on POSIX and Windows.

    On POSIX we send SIGTERM then escalate; on Windows the same
    `terminate()` call sends CTRL-BREAK semantics, which is the
    closest stdlib gives us. We can't rely on `pkill -f` here.
    """
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def _run_locust(n: int) -> Path:
    csv_prefix = ARTIFACTS / f"locust_n{n}"
    log_path = ARTIFACTS / f"locust_n{n}.log"
    log_fh = log_path.open("wb")
    # `--csv prefix` makes Locust write {prefix}_stats.csv; that's the
    # file the checker reads, and the file we parse for max RPS.
    cmd = [
        "uv",
        "run",
        "locust",
        "-f",
        "tests/load/locustfile.py",
        "--host",
        "http://localhost:8000",
        "--users",
        str(LOCUST_USERS),
        "--spawn-rate",
        str(LOCUST_SPAWN),
        "--run-time",
        LOCUST_RUN,
        "--headless",
        "--csv",
        str(csv_prefix),
    ]
    proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, check=False)
    log_fh.close()
    if proc.returncode != 0:
        print(f"  WARN  locust exited non-zero ({proc.returncode}); see {log_path}")
    return Path(f"{csv_prefix}_stats.csv")


def main() -> int:
    if shutil.which("uv") is None:
        print("FAIL: `uv` not on PATH; install it or use `python -m` directly")
        return 2

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["consumers", "max_rps"])

        for n in SWEEP_NS:
            print(f"=== sweep N={n} ===")
            consumer_log = ARTIFACTS / f"consumer_n{n}.log"
            consumer_proc = _spawn_consumers(n, consumer_log)
            try:
                # Wait for the rebalance to settle before driving load,
                # otherwise N=1 looks artificially slow because half the
                # 5 min window is spent in `assignment()`-empty state.
                time.sleep(WARMUP_S)
                stats_csv = _run_locust(n)
                max_rps = _max_rps_from_stats(stats_csv)
            finally:
                _terminate(consumer_proc)
            writer.writerow([n, f"{max_rps:.1f}"])
            fh.flush()
            print(f"    N={n} → max_rps={max_rps:.1f}")

    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    # Honour Ctrl-C cleanly on Windows too.
    if hasattr(signal, "SIGBREAK"):  # pragma: no cover - Windows-only
        signal.signal(signal.SIGBREAK, lambda *_: sys.exit(130))  # type: ignore[attr-defined]
    sys.exit(main())
