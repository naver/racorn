#!/usr/bin/env bash
# Top-level NUMA-pinned benchmark sweep — single entrypoint that delegates
# to focused sub-scripts. Each sub-script is fully self-contained (its own
# NUMA / THREADS / DATASETS) so it can also be run independently.
#
# All sub-scripts pin worker threads + memory to NUMA node 0 (override via
# NUMA_NODE / NUMACTL env) to eliminate cross-socket DRAM access (the source
# of run-to-run variance and ~15% latency inflation observed without it).
# THREADS=32 (physical cores on node 0) and nq=992 are the default
# measurement setup throughout.
#
# Sweep coverage:
#   1. run_bench_default.sh  — bench, default params + per-dataset bbr sweep
#   2. run_qps_default.sh    — qps,   default params, single per-dataset bbr
#   3. run_bench_alt.sh      — bench, alt params (m=32/efc=500/efs=100,400/2hop)
#   4. run_bench_3hop.sh     — bench, 3-hop expansion (needs bench_acorn_3hop)
#   5. run_correlation.sh    — cluster correlation default (pos/neg × K=10/100)
#   6. run_correlation_bbr.sh — cluster correlation × bbr={0.25, 0.5}
#   7. run_racorn1_ablation.sh — modulo RACORN-1 ablation (stride / bbr=0)
#
# Output: all sub-scripts write into the same OUT_DIR, but their result
# filename prefixes (`result_acorn_numa_*` / `qps_acorn_numa_*` /
# `gt_corr_numa_*` / `qps_corr_numa_*` / `result_racorn1_ablation_*`)
# are disjoint, so combined output is unambiguous.
#
# Usage:
#   ./run_final_numa.sh [OUT_DIR] [--no-numa]
#     OUT_DIR   : optional output directory (default: current directory).
#     --no-numa : disable NUMA pinning (default: NUMA pinning ON).
#
# Examples:
#   ./run_final_numa.sh results/                       # NUMA on (default)
#   ./run_final_numa.sh results/ --no-numa             # NUMA off
#   THREADS=16 ./run_final_numa.sh results/ --no-numa  # NUMA off, 16 threads
#
# Env overrides (forwarded to every sub-script):
#   BENCH       bench_acorn binary path (default: ../build/bench/bench_acorn)
#   BENCH_3HOP  bench_acorn_3hop binary path (default: ../build/bench/bench_acorn_3hop)
#   NUMA_NODE   numactl node (default: 0)
#   NUMACTL     full numactl prefix (overrides --no-numa if set explicitly)
#   THREADS     parallel search threads (default: 32)
set -euo pipefail
cd "$(dirname "$0")"

# ── Argument parsing ─────────────────────────────────────────────────────────
NUMA_ON=1
OUT_DIR="."
for arg in "$@"; do
    case "$arg" in
        --no-numa) NUMA_ON=0 ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *)  OUT_DIR="$arg" ;;
    esac
done

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M)
export TS

# ── NUMA setup ────────────────────────────────────────────────────────────────
NUMA_NODE="${NUMA_NODE:-0}"
if [ -z "${NUMACTL+set}" ]; then
    if [ "$NUMA_ON" -eq 1 ]; then
        NUMACTL="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
    else
        NUMACTL=""
    fi
fi
export NUMACTL
echo "NUMA: $([ -n "$NUMACTL" ] && echo "on (node ${NUMA_NODE})" || echo "off")"

echo "=== [$(date +%H:%M:%S)] 1/7 bench default + bbr sweep ==="
./run_bench_default.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] 2/7 qps default + bbr sweep ==="
./run_qps_default.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] 3/7 bench alternate params ==="
./run_bench_alt.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] 4/7 bench 3-hop expansion sweep ==="
./run_bench_3hop.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] 5/7 cluster correlation (default) ==="
./run_correlation.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] 6/7 cluster correlation × bbr sweep ==="
./run_correlation_bbr.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] 7/7 RACORN-1 ablation (modulo selectivity) ==="
./run_racorn1_ablation.sh "$OUT_DIR"

echo "=== [$(date +%H:%M:%S)] All benchmarks complete. (timestamp: ${TS}) ==="
