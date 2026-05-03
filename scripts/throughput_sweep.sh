#!/usr/bin/env bash
# Phase 8c — scalability proof: max sustained RPS at N=1, 3, 6 consumers.
#
# Outputs `artifacts/throughput_vs_consumers.csv` with one row per N.
# Restarts the consumer pool between runs so the worker count truly
# changes (not just a noisy assignment shuffle).

set -euo pipefail

ART_DIR="${ARTIFACTS_DIR:-artifacts}"
mkdir -p "$ART_DIR"

OUT="$ART_DIR/throughput_vs_consumers.csv"
echo "consumers,max_rps" > "$OUT"

for N in 1 3 6; do
  echo "=== sweep N=$N ==="

  # Tear down any running consumer pool from a prior iteration.
  pkill -f "click_rec.cli consumer" 2>/dev/null || true
  sleep 2

  # Spawn the new pool. `make consumer N=$N` honours the pool-size knob.
  make consumer N="$N" >"$ART_DIR/consumer_n${N}.log" 2>&1 &
  CONSUMER_PID=$!
  # Give workers a few seconds to join the group + warm caches.
  sleep 8

  uv run locust \
    -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --users 200 \
    --spawn-rate 40 \
    --run-time 5m \
    --headless \
    --csv "$ART_DIR/locust_n${N}" \
    >"$ART_DIR/locust_n${N}.log" 2>&1 || true

  # Locust emits a `_stats.csv` with a "Requests/s" column. Take its
  # max across the ramp+steady-state window.
  STATS="$ART_DIR/locust_n${N}_stats.csv"
  if [[ -f "$STATS" ]]; then
    MAX_RPS=$(awk -F, '
      NR == 1 {
        for (i = 1; i <= NF; i++) {
          col = tolower($i); gsub(/"/, "", col)
          if (col == "requests/s") c = i
        }
      }
      NR > 1 && c {
        v = $c; gsub(/"/, "", v); v += 0
        if (v > m) m = v
      }
      END { print m + 0 }
    ' "$STATS")
  else
    MAX_RPS=0
  fi
  echo "$N,$MAX_RPS" >> "$OUT"

  # Stop the consumer pool before the next iteration.
  kill "$CONSUMER_PID" 2>/dev/null || true
  wait "$CONSUMER_PID" 2>/dev/null || true
done

echo "wrote $OUT"
cat "$OUT"
