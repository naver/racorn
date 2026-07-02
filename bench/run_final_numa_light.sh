#!/usr/bin/env bash
# Smoke validation of run_final_numa.sh — runs every sub-script end-to-end
# on the smallest possible workload so the entire pipeline finishes in a
# few minutes.
#
# What this verifies (not perf-grade numbers):
#   - All 7 sub-scripts called by run_final_numa.sh launch without error
#   - bench_acorn / bench_acorn_3hop binaries exist and accept the
#     option combinations each sub-script uses
#   - Result-file prefixes are disjoint across stages and land in OUT_DIR
#   - RACORN-1+ exact-fallback triggers at expected selectivities (cell
#     log shows `FBExact=YES`)
#   - Cluster correlation pipeline runs end-to-end (GT build + bench +
#     QPS) — assumes cluster files already exist under ./data/clusters
#
# Differences from run_final_numa.sh:
#   - Single dataset (sift_1m by default) — text2image_40m is excluded
#     to keep wall time short.
#   - NQ=100, RUNS=1 per dataset row.
#   - Sub-scripts inherit these via env overrides (DATASETS).
#
# Usage:
#   ./run_final_numa_light.sh [OUT_DIR] [--no-numa]
#     OUT_DIR   : optional output directory (default: current directory).
#     --no-numa : disable NUMA pinning (default: NUMA pinning ON).
#
# Examples:
#   ./run_final_numa_light.sh results/                       # NUMA on (default)
#   ./run_final_numa_light.sh results/ --no-numa             # NUMA off
#   THREADS=8 ./run_final_numa_light.sh results/ --no-numa   # NUMA off, 8 threads
#
# Env var overrides:
#   BENCH        bench_acorn binary (default: ../build/bench/bench_acorn)
#   BENCH_3HOP   bench_acorn_3hop binary (default: ../build/bench/bench_acorn_3hop)
#   DATASET      shortcut: sift | gist | text2image_1m | text2image_40m
#                (auto-expands to the matching 7-col DATASETS row; ignored
#                if DATASETS is also set so power users can still override)
#   DATASETS     7-col dataset row (default: sift_1m, nq=100, runs=1)
#                takes precedence over DATASET when both are set
#   STAGES       which stages to run (default: "1 2 3 4 5 6 7")
#   THREADS      parallel search threads (default: 16)
#   NUMACTL      full numactl prefix (overrides --no-numa if set explicitly)
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

# ── Default thread count for light mode ──────────────────────────────────────
THREADS="${THREADS:-16}"
export THREADS

echo "NUMA   : $([ -n "$NUMACTL" ] && echo "on (node ${NUMA_NODE})" || echo "off")"
echo "THREADS: $THREADS"

# Optional DATASET= shortcut: maps a short dataset name to the matching
# 7-col DATASETS row used by Option A's run_light.sh, so Option B's manual
# flow can specify the dataset without typing the full row format.
# Only applied when DATASETS is unset (so DATASETS still wins for power users).
if [ -z "${DATASETS:-}" ] && [ -n "${DATASET:-}" ]; then
    case "$DATASET" in
        sift)
            DATASETS='"sift 1000000 sift_1m 0.003 0.0015 0.006 100 1"' ;;
        gist)
            DATASETS='"gist 1000000 gist_1m 0.003 0.0015 0.006 100 1"' ;;
        text2image_1m|t2i_1m)
            DATASETS='"text2image 1000000 text2image_1m 0.003 0.0015 0.006 100 1"' ;;
        text2image_40m|t2i_40m)
            DATASETS='"text2image 40000000 text2image_40m 0.0003 0.00015 0.0006 100 1"' ;;
        *)
            echo "ERROR: unknown DATASET=$DATASET" >&2
            echo "  valid: sift | gist | text2image_1m | text2image_40m" >&2
            exit 1 ;;
    esac
fi

# Default smoke dataset row.
# 7-col row (used by run_bench_default / qps_default / bench_alt /
# bench_3hop / run_correlation / run_correlation_bbr):
#   DATASET  N  NAME  FB  FB_EFS100  FB_EFS400  NQ  RUNS
DS_7COL="${DATASETS:-\"sift 1000000 sift_1m 0.003 0.0015 0.006 100 1\"}"
# 5-col row (used by run_racorn1_ablation): derived from DS_7COL by
# dropping the three FB-threshold columns. Previously hardcoded to a
# sift_1m row, which caused stage 7 to run on sift_1m (with a missing
# volume) whenever DATASET=gist / text2image_1m was selected.
DS_5COL=$(echo "$DS_7COL" | awk '{
    # Strip leading/trailing quotes to expose tokens, then re-quote
    gsub(/^"|"$/, "", $0)
    nf = split($0, t, /[ \t]+/)
    printf("\"%s %s %s %s %s\"", t[1], t[2], t[3], t[nf-1], t[nf])
}')

STAGES="${STAGES:-1 2 3 4 5 6 7}"

# Sub-script ↔ DATASETS column shape mapping.
declare -A STAGE_SCRIPT=(
    [1]=run_bench_default.sh
    [2]=run_qps_default.sh
    [3]=run_bench_alt.sh
    [4]=run_bench_3hop.sh
    [5]=run_correlation.sh
    [6]=run_correlation_bbr.sh
    [7]=run_racorn1_ablation.sh
)
declare -A STAGE_DS=(
    [1]="$DS_7COL"   [2]="$DS_7COL"   [3]="$DS_7COL"   [4]="$DS_7COL"
    [5]="$DS_7COL"   [6]="$DS_7COL"   [7]="$DS_5COL"
)

run_stage() {
    local n="$1" script="$2" ds="$3"
    local t0
    t0=$(date +%s)
    echo "=== [$(date +%H:%M:%S)] $n/7 $script ==="
    if ! DATASETS="$ds" ./"$script" "$OUT_DIR" > "/tmp/_light_${script}.stdout" 2>&1; then
        local rc=$?
        echo "  *** $script failed (exit $rc, $(($(date +%s)-t0))s) — tail of stdout: ***"
        tail -5 "/tmp/_light_${script}.stdout" | sed 's/^/    /'
        return $rc
    fi
    echo "  ok ($(($(date +%s)-t0))s)"
}

START=$(date +%s)
for n in $STAGES; do
    run_stage "$n" "${STAGE_SCRIPT[$n]}" "${STAGE_DS[$n]}"
done
echo "=== [$(date +%H:%M:%S)] All light stages complete (total $(($(date +%s)-START))s, timestamp: ${TS}) ==="
