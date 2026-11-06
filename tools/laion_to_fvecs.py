#!/usr/bin/env python3
# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
r"""LAION-400M CLIP image embeddings (512 dimensions, fp16 npy) to fvecs.

Downloads the public embedding shards (`img_emb_<i>.npy`, about one million rows each, no
token required) and converts them to float32 fvecs. The first `--n` rows become the base
file, the next `--nq` rows the query file. The CLIP embeddings are L2-normalized before they
are written, so that cosine and Euclidean distance rank the vectors the same way. Only numpy
is needed. Shards are taken in numerical order. A shard that cannot be downloaded aborts
the run: skipping one would shift every later row and silently produce a different dataset
on another machine. The metadata shards (README.md, Datasets) have the same shard and row
order. This is what lets Section VI-C attach a column to key i and Section VI-D a caption.
The environment variable LAION_EMB_URL, when set, replaces the URL prefix of the shards.

Usage:
  python3 laion_to_fvecs.py --out-dir $ROOT/data/raw-laion25m --tmp $ROOT/data/raw-laion25m/npy \
      --n 25000000 --nq 1000
  python3 laion_to_fvecs.py --out-dir $ROOT/data/raw-laion1m --tmp $ROOT/data/raw-laion1m/npy \
      --n 1000000 --nq 1000
"""
import argparse
import os
import struct
import time
import urllib.request

import numpy as np

DEPLOY = os.environ.get("LAION_EMB_URL",
                        "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="where laion_base.fvecs and laion_query.fvecs are written")
    ap.add_argument("--tmp", required=True, help="where the downloaded npy shards are kept until converted")
    ap.add_argument("--n", type=int, default=10_000_000, help="base vectors")
    ap.add_argument("--nq", type=int, default=1000, help="query vectors, taken right after the base rows")
    ap.add_argument("--max-shards", type=int, default=420)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    need = args.n + args.nq
    base_path = os.path.join(args.out_dir, "laion_base.fvecs")
    query_path = os.path.join(args.out_dir, "laion_query.fvecs")

    written = 0
    qbuf = []
    fb = open(base_path, "wb")
    for i in range(args.max_shards):
        if written >= need:
            break
        local = os.path.join(args.tmp, f"img_emb_{i}.npy")
        if not os.path.exists(local) or os.path.getsize(local) == 0:
            url = f"{DEPLOY}/img_emb_{i}.npy"
            for attempt in range(1, args.retries + 1):
                try:
                    with urllib.request.urlopen(url, timeout=180) as r, open(local + ".part", "wb") as out:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            out.write(chunk)
                    os.replace(local + ".part", local)
                    break
                except Exception as e:
                    print(f"  shard {i} attempt {attempt}/{args.retries} failed ({e})", flush=True)
                    time.sleep(10 * attempt)
            else:
                raise SystemExit(f"shard {i} could not be downloaded; aborting to keep the dataset deterministic")
        mat = np.load(local).astype("float32")  # (~1M, 512) fp16 -> f32
        nrm = np.linalg.norm(mat, axis=1, keepdims=True)
        nrm[nrm == 0] = 1
        mat = mat / nrm
        d = mat.shape[1]
        for row in mat:
            if written < args.n:
                fb.write(struct.pack("<i", d))
                fb.write(row.tobytes())
            elif len(qbuf) < args.nq:
                qbuf.append(row.copy())
            written += 1
            if written >= need:
                break
        print(f"  img_emb_{i}: total {written}/{need}", flush=True)
        os.remove(local)
    fb.close()
    with open(query_path, "wb") as fq:
        for row in np.asarray(qbuf, dtype="float32"):
            fq.write(struct.pack("<i", row.shape[0]))
            fq.write(row.tobytes())
    print(f"base {min(written, args.n)}, query {len(qbuf)} -> {base_path}, {query_path}")


if __name__ == "__main__":
    main()
