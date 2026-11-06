#!/usr/bin/env bash
# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
#
# Downloads the datasets of the paper into the working root (README.md, Datasets) and converts
# them to the files the drivers read. Run it from the working root; each name is one dataset:
#   sift        SIFT 1M from corpus-texmex (FTP)                                -> data/raw
#   gist        GIST 1M from corpus-texmex (FTP)                                -> data/raw
#   sift100m    the first 100M vectors of BIGANN, as float fvecs                -> data/raw-sift100m
#   text2image  the first 40M vectors of Yandex Text-to-Image                   -> data/raw
#   yfcc        the YFCC 10M files of the Big ANN filtered track; YFCC 1M is their first million
#               base vectors                                                    -> data/raw-yfcc10m
#   laion1m     LAION-400M CLIP image embeddings, the first 1M rows as the base and the next 1,000
#   laion25m    as the queries (laion_to_fvecs.py), 1M and 25M                  -> data/raw-laion1m, data/raw-laion25m
#   laion-meta  the LAION metadata shards 0-25 (Sections VI-C and VI-D)         -> data/laion_meta
#   all         every name above, in this order
# The script stops at the first failed step.
#
# Usage:  bash /path/to/this/repository/tools/get_datasets.sh sift gist yfcc laion1m laion-meta
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BIGANN=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
T2I=https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I
LAION_META=https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata

texmex() { # sift | gist: the corpus-texmex archive, served over FTP only
    mkdir -p data/raw
    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/$1.tar.gz
    tar -xzf $1.tar.gz && mv $1/${1}_base.fvecs $1/${1}_query.fvecs data/raw/
}
sift100m() {
    mkdir -p data/raw-sift100m
    curl -sf -r 0-$((8 + 100000000 * 128 - 1)) $BIGANN/bigann/base.1B.u8bin \
      | python3 "$HERE/prefix_to_fvecs.py" --type u8 --n 100000000 --d 128 --to fvecs > data/raw-sift100m/sift_base.fvecs
    curl -sf $BIGANN/bigann/query.public.10K.u8bin \
      | python3 "$HERE/prefix_to_fvecs.py" --type u8 --n 10000 --d 128 --to fvecs > data/raw-sift100m/sift_query.fvecs
}
text2image() {
    mkdir -p data/raw
    curl -sf -r 0-$((8 + 40000000 * 200 * 4 - 1)) $T2I/base.1B.fbin \
      | python3 "$HERE/prefix_to_fvecs.py" --type f32 --n 40000000 --d 200 --to fbin > data/raw/text2image-40M.fbin
    curl -sf -o data/raw/text2image-query-100K.fbin $T2I/query.public.100K.fbin
}
yfcc() {
    mkdir -p data/raw-yfcc10m
    for f in base.10M.u8bin query.public.100K.u8bin base.metadata.10M.spmat query.metadata.public.100K.spmat GT.public.ibin; do
        curl -fL -C - -o data/raw-yfcc10m/$f $BIGANN/yfcc100M/$f
    done
}
laion() { # laion1m | laion25m
    local n=${1#laion}; n=${n%m}   # 1 or 25
    python3 "$HERE/laion_to_fvecs.py" --out-dir data/raw-$1 --tmp data/raw-$1/npy --n ${n}000000 --nq 1000
}
laion_meta() {
    mkdir -p data/laion_meta
    for i in $(seq 0 25); do
        curl -fL -C - -o data/laion_meta/metadata_$i.parquet $LAION_META/metadata_$i.parquet
    done
}

[ $# -gt 0 ] || { echo "usage: $0 <sift|gist|sift100m|text2image|yfcc|laion1m|laion25m|laion-meta|all>..." >&2; exit 2; }
[ "$1" = all ] && set -- sift gist sift100m text2image yfcc laion1m laion25m laion-meta
for d in "$@"; do
    echo "== $d"
    case $d in
        sift|gist) texmex $d ;;
        sift100m) sift100m ;;
        text2image) text2image ;;
        yfcc) yfcc ;;
        laion1m|laion25m) laion $d ;;
        laion-meta) laion_meta ;;
        *) echo "unknown dataset: $d" >&2; exit 2 ;;
    esac
done
