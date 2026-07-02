#!/usr/bin/env bash
# download_datasets.sh — download raw dataset files for bench_acorn.
#
# Downloads SIFT-1M and GIST-1M automatically (FTP / HTTP mirror).
# Text-to-Image 40M cannot be automated (URL changes frequently) —
# manual download instructions are printed instead.
#
# Each file is skipped silently if it already exists in DATA_DIR.
#
# Usage (run from repository root):
#   ./download_datasets.sh [DATASETS...]
#     DATASETS : space-separated list (default: "sift")
#                Choices: sift  gist  text2image
#
# Examples:
#   ./download_datasets.sh                       # sift only
#   ./download_datasets.sh sift gist             # sift + gist
#   ./download_datasets.sh sift gist text2image  # all (text2image: manual)
#
# Env overrides:
#   DATA_DIR   destination directory (default: bench/data/raw)
#
# Requirements:
#   wget  OR  curl

set -euo pipefail
cd "$(dirname "$0")"

DATA_DIR="${DATA_DIR:-bench/data/raw}"
DATASETS="${*:-sift}"
mkdir -p "$DATA_DIR"

echo "DATA_DIR : $DATA_DIR"
echo "Datasets : $DATASETS"
echo ""

# ── Download helper ──────────────────────────────────────────────────────────
fetch() {
    local url="$1" dest="$2"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || { rm -f "$dest"; return 1; }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url" || { rm -f "$dest"; return 1; }
    else
        echo "ERROR: wget or curl is required."
        exit 1
    fi
    if [ ! -s "$dest" ]; then
        rm -f "$dest"
        return 1
    fi
}

# ── SIFT-1M ──────────────────────────────────────────────────────────────────
download_sift() {
    local base="$DATA_DIR/sift_base.fvecs"
    local query="$DATA_DIR/sift_query.fvecs"
    if [ -f "$base" ] && [ -f "$query" ]; then
        echo "[sift] Already present ($(du -sh "$base" | cut -f1)) — skip."
        return
    fi
    echo "[sift] Downloading (~560 MB compressed)..."
    local tar="$DATA_DIR/sift.tar.gz"
    if [ ! -f "$tar" ]; then
        fetch "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz" "$tar" || \
        fetch "http://corpus-texmex.irisa.fr/sift.tar.gz" "$tar"
    fi
    echo "[sift] Extracting..."
    tar -xzf "$tar" -C "$DATA_DIR"
    for d in "$DATA_DIR/sift" "$DATA_DIR/ANN_SIFT1M"; do
        [ -d "$d" ] && mv "$d"/sift_base.fvecs "$d"/sift_query.fvecs "$DATA_DIR/" && rm -rf "$d"
    done
    rm -f "$tar"
    local dim
    dim=$(od -A n -t u4 -N 4 "$base" | tr -d ' ')
    [ "$dim" = "128" ] || { echo "ERROR: sift dim mismatch (got $dim, expected 128)"; exit 1; }
    echo "[sift] Done — sift_base.fvecs $(du -sh "$base" | cut -f1), dim=128 ✓"
}

# ── GIST-1M ──────────────────────────────────────────────────────────────────
download_gist() {
    local base="$DATA_DIR/gist_base.fvecs"
    local query="$DATA_DIR/gist_query.fvecs"
    if [ -f "$base" ] && [ -f "$query" ]; then
        echo "[gist] Already present ($(du -sh "$base" | cut -f1)) — skip."
        return
    fi
    echo "[gist] Downloading (~3.8 GB uncompressed — may take several minutes)..."
    local tar="$DATA_DIR/gist.tar.gz"
    if [ ! -f "$tar" ]; then
        fetch "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz" "$tar" || \
        fetch "http://corpus-texmex.irisa.fr/gist.tar.gz" "$tar"
    fi
    echo "[gist] Extracting..."
    tar -xzf "$tar" -C "$DATA_DIR"
    for d in "$DATA_DIR/gist" "$DATA_DIR/ANN_GIST1M"; do
        [ -d "$d" ] && mv "$d"/gist_base.fvecs "$d"/gist_query.fvecs "$DATA_DIR/" && rm -rf "$d"
    done
    rm -f "$tar"
    local dim
    dim=$(od -A n -t u4 -N 4 "$base" | tr -d ' ')
    [ "$dim" = "960" ] || { echo "ERROR: gist dim mismatch (got $dim, expected 960)"; exit 1; }
    echo "[gist] Done — gist_base.fvecs $(du -sh "$base" | cut -f1), dim=960 ✓"
}

# ── Text-to-Image 40M ────────────────────────────────────────────────────────
download_text2image() {
    local base="$DATA_DIR/text2image-40M.fbin"
    local query="$DATA_DIR/text2image-query-100K.fbin"
    if [ -f "$base" ] && [ -f "$query" ]; then
        echo "[text2image] Already present ($(du -sh "$base" | cut -f1)) — skip."
        return
    fi
    cat <<'MSG'
[text2image] Manual download required.

  No upstream `base.40M.fbin` exists — the 40M subset is derived by
  truncating the larger Yandex T2I file. See bench/datasets.md for the
  full recipe (Text2Image section: which base file to fetch, the
  truncate-in-place command, expected sizes per target run). The short
  story:

    - For `text2image_1m` (light validation): download
        base.10M.fbin (~8 GB) → save as text2image-40M.fbin
        (bench_acorn --n 1000000 only reads the first 1M).
    - For `text2image_40m` (paper-grade): download
        base.1B.fbin (~800 GB transient), then truncate the first
        40M vectors in place and rename to text2image-40M.fbin.
    - Query file (always required):
        query.public.100K.fbin → save as text2image-query-100K.fbin.

  Current download URLs change periodically. Locate them at:

    https://big-ann-benchmarks.com/  → Datasets → Yandex T2I
    or: https://github.com/harsha-simhadri/big-ann-benchmarks
        (file: benchmark/datasets.py — look up the `Text2Image1B` class's
         `base_url` and `query_url`).

  Then download:
MSG
    echo "    wget -O ${base} <base_1B_or_10M_url>"
    echo "    wget -O ${query} <query_100K_url>"
    cat <<'MSG'

  Expected sanity check after download:
    xxd -l 8 text2image-40M.fbin
    # Expect: "005a 6202 c800 0000"  →  n=40,000,000  dim=200
    xxd -l 8 text2image-query-100K.fbin
    # Expect: "a086 0100 c800 0000"  →  n=100,000  dim=200

  See bench/datasets.md §3 for full details.
MSG
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
for DS in $DATASETS; do
    case "$DS" in
        sift)               download_sift ;;
        gist)               download_gist ;;
        text2image*)        download_text2image ;;
        *)
            echo "WARNING: unknown dataset '$DS' — skipping."
            echo "  Valid choices: sift  gist  text2image"
            ;;
    esac
    echo ""
done

echo "All requested datasets processed."
echo "Files are in: $DATA_DIR"
