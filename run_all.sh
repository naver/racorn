#!/usr/bin/env bash
# run_all.sh — build + setup + benchmark sweep for the RACORN-1 / ACORN-1 artifact.
#
# Assumes dataset files are already present in DATA_DIR.
# Run download_datasets.sh first to fetch them.
#
# Steps:
#   1. Build binaries (bench_acorn + bench_acorn_3hop)
#   2. Check dataset files are present (exit with instructions if not)
#   3. Build HNSW index volumes  (skipped per-file if already present)
#   4. Build ground-truth files  (skipped per-file if already present)
#   5. Build cluster files       (skipped if present; requires Python + sklearn)
#   6. Run run_final_numa_light.sh (sift_1m smoke sweep, stages 1-4, 7)
#      Stages 5-6 (cluster correlation) added automatically if cluster files exist.
#      Cluster GT files are NOT pre-built here — run_correlation.sh (stage 6)
#      generates them inline via --build-gt --cluster-correlation per
#      (dataset, corr, K) tuple. They land in GT_DIR with the suffix
#      _cluster_<pos|neg>_kK so they never collide with modulo GT files.
#
# Usage (run from repository root):
#   ./run_all.sh [OUT_DIR]
#     OUT_DIR : output directory for result logs (default: results_YYYYMMDD_HHMM)
#
# Typical workflow:
#   ./download_datasets.sh [sift] [gist] [text2image]   # step 0: get raw data
#   ./run_all.sh [OUT_DIR]                               # steps 1-6
#
# Env overrides:
#   DATASETS   space-separated list of datasets to prepare and benchmark.
#              Choices: sift  gist  text2image  text2image_40m
#              Default: "sift"  (sift_1m only — suitable for a smoke run)
#              Full paper sweep: DATASETS="sift gist text2image text2image_40m"
#   DATA_DIR   raw dataset directory  (default: bench/data/raw)
#   VOLUME_DIR index volume directory (default: bench/data/volumes)
#   GT_DIR     ground-truth directory (default: bench/data/gt)
#   CLUSTER_DIR cluster file directory (default: bench/data/clusters)
#   NUMACTL    numactl prefix (default: ""; use "numactl --cpunodebind=0 --membind=0" on NUMA)
#   THREADS    parallel search threads (default: 32).
#              On NUMA machines, set to the physical core count of one NUMA node
#              (e.g. 32 for a dual-socket 64-core server) and pair with NUMACTL.
#              Lower to available core count on non-server hardware.
#   JOBS       cmake build parallelism (default: nproc)

set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"

# ── Output directory ─────────────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M)
OUT_DIR="${1:-${REPO_ROOT}/results_${TS}}"
mkdir -p "$OUT_DIR"
# Normalize to an absolute path. Several sub-steps cd into other directories
# (bench/, bench/scripts/) and then redirect logs to $OUT_DIR; a relative
# OUT_DIR would be misinterpreted as relative to the cwd of the sub-step.
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
LOG="$OUT_DIR/run_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "======================================================"
echo " RACORN-1 / ACORN-1 Artifact — Reproduction Run"
echo " $(date)"
echo " OUT_DIR: $OUT_DIR"
echo "======================================================"

# ── Configurable paths ───────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/bench/data/raw}"
VOLUME_DIR="${VOLUME_DIR:-${REPO_ROOT}/bench/data/volumes}"
GT_DIR="${GT_DIR:-${REPO_ROOT}/bench/data/gt}"
CLUSTER_DIR="${CLUSTER_DIR:-${REPO_ROOT}/bench/data/clusters}"
# RUN_DATASETS: dataset names for setup steps (sift, gist, text2image, text2image_40m)
# Stored in RUN_DATASETS to avoid colliding with the full-row DATASETS format
# expected by run_final_numa_light.sh sub-scripts.
RUN_DATASETS="${DATASETS:-sift}"
unset DATASETS   # clear so sub-scripts use their own default row format
export DATA_DIR VOLUME_DIR GT_DIR CLUSTER_DIR

JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

echo "  Datasets : $RUN_DATASETS"
echo "  DATA_DIR : $DATA_DIR"
echo ""

# ── Step 1: Build ────────────────────────────────────────────────────────────
echo "=== [1/5] Build ==="
if [ -f build/bench/bench_acorn ] && [ -f build/bench/bench_acorn_3hop ]; then
    echo "  Binaries already present — skipping. (Delete build/ to force.)"
else
    if ! command -v cmake &>/dev/null; then
        echo "ERROR: cmake not found. Install cmake 3.11+ and retry."
        exit 1
    fi
    echo "  cmake configure..."
    cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -Wno-dev > "$OUT_DIR/cmake.log" 2>&1
    echo "  cmake build (j=${JOBS})..."
    cmake --build build -j"$JOBS" >> "$OUT_DIR/cmake.log" 2>&1
    echo "  Done: $(ls -lh build/bench/bench_acorn build/bench/bench_acorn_3hop | awk '{print $NF,$5}')"
fi

BENCH="${REPO_ROOT}/build/bench/bench_acorn"
BENCH_3HOP="${REPO_ROOT}/build/bench/bench_acorn_3hop"
export BENCH BENCH_3HOP

# ── Step 2: Dataset check ─────────────────────────────────────────────────────
echo ""
echo "=== [2/5] Dataset check ==="
MISSING=0
for DS in $RUN_DATASETS; do
  case "$DS" in
    sift)
      for f in "$DATA_DIR/sift_base.fvecs" "$DATA_DIR/sift_query.fvecs"; do
          [ -f "$f" ] && echo "  OK : $f" || { echo "  MISSING: $f"; MISSING=1; }
      done ;;
    gist)
      for f in "$DATA_DIR/gist_base.fvecs" "$DATA_DIR/gist_query.fvecs"; do
          [ -f "$f" ] && echo "  OK : $f" || { echo "  MISSING: $f"; MISSING=1; }
      done ;;
    text2image|text2image_40m)
      for f in "$DATA_DIR/text2image-40M.fbin" "$DATA_DIR/text2image-query-100K.fbin"; do
          [ -f "$f" ] && echo "  OK : $f" || { echo "  MISSING: $f"; MISSING=1; }
      done ;;
  esac
done

if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "ERROR: one or more dataset files are missing."
    echo "  Run first:  ./download_datasets.sh $RUN_DATASETS"
    echo "  Then retry: ./run_all.sh $OUT_DIR"
    exit 1
fi

# ── Step 3: Build HNSW index volumes ─────────────────────────────────────────
echo ""
echo "=== [3/5] HNSW index volumes ==="
mkdir -p "$VOLUME_DIR"

build_volume() {
    local ds="$1" n="$2" m="$3" efc="$4" label="$5" out="$6"
    if [ -f "$out" ]; then
        echo "  [$label] Already present ($(du -sh "$out" | cut -f1)) — skip."
    else
        echo "  [$label] Building..."
        "$BENCH" --dataset "$ds" --n "$n" --m "$m" --ef-construction "$efc" \
            --build-index \
            --data-dir "$DATA_DIR" --volume-dir "$VOLUME_DIR" \
            >> "$OUT_DIR/build_volumes.log" 2>&1
        echo "  [$label] Done: $(du -sh "$out" | cut -f1)"
    fi
}

for DS in $RUN_DATASETS; do
  case "$DS" in
    sift)
      build_volume sift 1000000 16 100 "sift m16/efc100" \
          "$VOLUME_DIR/sift_1000000_128_m16_ef100.usearch"
      build_volume sift 1000000 32 100 "sift m32/efc100" \
          "$VOLUME_DIR/sift_1000000_128_m32_ef100.usearch"
      ;;
    gist)
      build_volume gist 1000000 16 100 "gist m16/efc100" \
          "$VOLUME_DIR/gist_1000000_960_m16_ef100.usearch"
      build_volume gist 1000000 32 100 "gist m32/efc100" \
          "$VOLUME_DIR/gist_1000000_960_m32_ef100.usearch"
      ;;
    text2image)
      build_volume text2image 1000000  16 100 "t2i_1m  m16/efc100" \
          "$VOLUME_DIR/text2image_1000000_200_m16_ef100.usearch"
      build_volume text2image 1000000  32 100 "t2i_1m  m32/efc100" \
          "$VOLUME_DIR/text2image_1000000_200_m32_ef100.usearch"
      ;;
    text2image_40m)
      build_volume text2image 40000000 16 100 "t2i_40m m16/efc100" \
          "$VOLUME_DIR/text2image_40000000_200_m16_ef100.usearch"
      ;;
  esac
done

# ── Step 4: Ground truth ──────────────────────────────────────────────────────
echo ""
echo "=== [4/5] Ground truth ==="
mkdir -p "$GT_DIR"

build_gt() {
    local ds="$1" n="$2" m="$3" efc="$4" nq="$5" label="$6" gt="$7" vol="$8"
    if [ -f "$gt" ]; then
        echo "  [$label] Already present — skip."
    elif [ ! -f "$vol" ]; then
        echo "  [$label] Volume not found — skip."
    else
        echo "  [$label] Computing exact GT (nq=$nq)..."
        "$BENCH" --dataset "$ds" --n "$n" --m "$m" --ef-construction "$efc" \
            --build-gt --build-gt-k 100 --nq "$nq" \
            --data-dir "$DATA_DIR" --volume-dir "$VOLUME_DIR" --gt-dir "$GT_DIR" \
            >> "$OUT_DIR/build_gt.log" 2>&1
        echo "  [$label] Done."
    fi
}

for DS in $RUN_DATASETS; do
  case "$DS" in
    sift)
      build_gt sift 1000000 16 100 1000 "sift m16" \
          "$GT_DIR/sift_1000000_128_m16_ef100_gt_k100_nq1000.bin" \
          "$VOLUME_DIR/sift_1000000_128_m16_ef100.usearch"
      ;;
    gist)
      build_gt gist 1000000 16 100 1000 "gist m16" \
          "$GT_DIR/gist_1000000_960_m16_ef100_gt_k100_nq1000.bin" \
          "$VOLUME_DIR/gist_1000000_960_m16_ef100.usearch"
      ;;
    text2image)
      build_gt text2image 1000000  16 100 1000 "t2i_1m m16" \
          "$GT_DIR/text2image_1000000_200_m16_ef100_gt_k100_nq1000.bin" \
          "$VOLUME_DIR/text2image_1000000_200_m16_ef100.usearch"
      ;;
    text2image_40m)
      build_gt text2image 40000000 16 100 1000 "t2i_40m m16" \
          "$GT_DIR/text2image_40000000_200_m16_ef100_gt_k100_nq1000.bin" \
          "$VOLUME_DIR/text2image_40000000_200_m16_ef100.usearch"
      ;;
  esac
done

# ── Step 5: Cluster files + benchmark ────────────────────────────────────────
echo ""
echo "=== [5/5] Cluster files + benchmark ==="
mkdir -p "$CLUSTER_DIR"
HAS_CLUSTERS=0

if python3 -c "import sklearn" 2>/dev/null; then
    NEED_BUILD=0
    for DS in $RUN_DATASETS; do
      DS_NAME=$(echo "$DS" | sed 's/_40m//')
      for K in 10 100; do
          [ ! -f "$CLUSTER_DIR/cluster_${DS_NAME}_n1000000_k${K}.bin" ] && NEED_BUILD=1
      done
    done

    if [ "$NEED_BUILD" -eq 0 ]; then
        echo "  Cluster files already present — skipping."
        HAS_CLUSTERS=1
    else
        echo "  Building cluster files..."
        (
            cd "${REPO_ROOT}/bench/scripts"
            for DS in $RUN_DATASETS; do
              case "$DS" in
                sift)            DS_ARG="sift 1000000" ;;
                gist)            DS_ARG="gist 1000000" ;;
                text2image)      DS_ARG="text2image 1000000" ;;
                text2image_40m)  DS_ARG="text2image 40000000" ;;
                *) continue ;;
              esac
              # DATASETS env var is eval'd as a bash-array literal by
              # build_all_clusters.sh (consistent with run_correlation.sh),
              # so the row must be quoted to survive eval as a single entry.
              DATA_DIR="$DATA_DIR" OUT_DIR="$CLUSTER_DIR" \
                  DATASETS="\"$DS_ARG\"" K_LIST="10 100" \
                  bash build_all_clusters.sh >> "$OUT_DIR/build_clusters.log" 2>&1
            done
        )
        echo "  Done."
        HAS_CLUSTERS=1
    fi
else
    echo ""
    echo "  ┌──────────────────────────────────────────────────────────────┐"
    echo "  │  WARNING: scikit-learn not found.                            │"
    echo "  │  Cluster files cannot be built.                              │"
    echo "  │  Stages 5-6 (cluster correlation) will be SKIPPED.          │"
    echo "  │                                                              │"
    echo "  │  To enable: pip install --user numpy scikit-learn scipy      │"
    echo "  │  Then re-run this script.                                    │"
    echo "  └──────────────────────────────────────────────────────────────┘"
    echo ""
fi

# benchmark
NUMACTL="${NUMACTL:-}"
THREADS="${THREADS:-32}"
export NUMACTL THREADS

if [ "$HAS_CLUSTERS" -eq 1 ]; then
    STAGES="1 2 3 4 5 6 7"
    echo "  Stages: 1-7 (cluster files found)"
else
    STAGES="1 2 3 4 7"
    echo "  Stages: 1-4, 7 (skipping cluster correlation stages 5-6)"
fi
export STAGES

echo "  NUMACTL : '${NUMACTL}'"
echo "  THREADS : $THREADS"
echo "  Results → $OUT_DIR/bench/"
echo ""
mkdir -p "$OUT_DIR/bench"
(
    cd "${REPO_ROOT}/bench"
    # LIGHT_DS_ROW (set by run_light.sh when a non-default dataset is
    # selected) is forwarded as DATASETS so run_final_numa_light.sh uses
    # the right row instead of its built-in sift default.
    BENCH="$BENCH" BENCH_3HOP="$BENCH_3HOP" \
    NUMACTL="$NUMACTL" THREADS="$THREADS" \
    STAGES="$STAGES" \
    DATASETS="${LIGHT_DS_ROW:-}" \
    bash run_final_numa_light.sh "$OUT_DIR/bench"
)

echo ""
echo "======================================================"
echo " All done."
echo " Results : $OUT_DIR/bench/"
echo " Log     : $LOG"
echo " $(date)"
echo "======================================================"
