#!/usr/bin/env python3
# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
r"""LAION metadata column workloads for Section VI-C (Natural Filters).

Reads the columns LAION-400M ships in `metadata_<i>.parquet` (the same shards and row order
as `img_emb_<i>.npy`). Writes the column file (SFCL) and the query file (SFCQ) that
`bench_synthfilter --filter column` takes. The row convention is that of `laion_to_fvecs.py`:
shards in numerical order, the first N rows are the base (keys 0..N-1), the next nq rows the
queries. The predicates do not use the metadata of the query rows. A categorical group applies
the same predicate to every query; a range group draws a window at a seeded random position.

Three columns, nine groups:

| column  | value                                                         | groups                       | selectivity |
|---------|---------------------------------------------------------------|------------------------------|-------------|
| nsfw    | grade of the distributed classifier: UNLIKELY 0, UNSURE 1, NSFW 2, other 3 | `NSFW=UNSURE`, `NSFW=NSFW` | set by the data |
| license | 1 when LICENSE is a creativecommons.org URL, else 0          | `LICENSE=CC`                 | set by the data |
| pixels  | original_width x original_height (the original resolution)   | `pixels <s>`, six groups     | target s in {10, 5, 3, 1, 0.5, 0.1}%, a bounded window at a random position per query |

A range window takes a rank interval [u, u+m) of the sorted value array V and uses its end
values as [lo, hi] (m = round(N*s)). Ties make the realized selectivity larger than the
target. A window that lands on a large run of equal values can be several times too large.
So up to `--tries` windows are drawn, and the first within `--max-overshoot` is kept (the
smallest if none is). The realized selectivity goes to the side-car JSON. The driver measures
it again per group, into `groups[]` of its output.

File formats (matching the driver's `column_t`):
  SFCL: "SFCL", u32 version=1, u64 N, u32 reserved, N x u32
  SFCQ: "SFCQ", u32 version=1, u32 ngroups, u32 nq, per group u32 label_len, label,
        f64 target (NaN = natural), u8 kind (0 = equality, 1 = range), nq x (u32 lo, u32 hi)

Usage:
  python3 laion_column_workload.py --n 1000000 --nq 992 \
      --meta-dir $ROOT/data/laion_meta --out-dir $ROOT/data/raw-laion1m/meta
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

CC = "creativecommons.org"
NSFW_CODES = {"UNLIKELY": 0, "UNSURE": 1, "NSFW": 2}
NSFW_NAMES = {0: "UNLIKELY", 1: "UNSURE", 2: "NSFW", 3: "other"}


def shards_for(meta_dir, need):
    """Shards in numerical order until their rows cover `need`. A missing shard would shift
    every later row, so it aborts instead of skipping."""
    import pyarrow.parquet as pq
    out, rows, i = [], 0, 0
    while rows < need:
        p = os.path.join(meta_dir, f"metadata_{i}.parquet")
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise SystemExit(f"missing {p}; download the shards in order (README.md, Datasets); aborting to keep the row order")
        out.append(p)
        rows += pq.read_metadata(p).num_rows
        i += 1
    return out


def read_columns(shards, n):
    import pyarrow.parquet as pq
    nsfw = np.empty(n, np.uint32)
    lic = np.empty(n, np.uint32)
    px = np.empty(n, np.uint32)
    row = 0
    t0 = time.time()
    for p in shards:
        if row >= n:
            break
        t = pq.read_table(p, columns=["NSFW", "LICENSE", "original_width", "original_height"])
        take = min(t.num_rows, n - row)
        ns = t.column("NSFW").to_pylist()[:take]
        li = t.column("LICENSE").to_pylist()[:take]
        w = t.column("original_width").to_numpy(zero_copy_only=False)[:take].astype(np.float64)
        h = t.column("original_height").to_numpy(zero_copy_only=False)[:take].astype(np.float64)
        nsfw[row:row + take] = np.fromiter((NSFW_CODES.get(x, 3) for x in ns), dtype=np.uint32, count=take)
        lic[row:row + take] = np.fromiter((1 if (x and CC in x) else 0 for x in li), dtype=np.uint32, count=take)
        pxv = np.nan_to_num(w * h, nan=0.0, posinf=0.0, neginf=0.0)
        px[row:row + take] = np.clip(pxv, 0, 2 ** 32 - 1).astype(np.uint32)
        row += take
        print(f"  {os.path.basename(p)}: rows {row}/{n} ({time.time() - t0:.0f}s)", flush=True)
    if row != n:
        raise SystemExit(f"only {row} rows available for N={n}")
    return nsfw, lic, px


def write_sfcl(path, vals):
    with open(path + ".part", "wb") as f:
        f.write(b"SFCL")
        f.write(struct.pack("<IQI", 1, len(vals), 0))
        f.write(np.ascontiguousarray(vals, dtype="<u4").tobytes())
    os.replace(path + ".part", path)


def write_sfcq(path, groups, nq):
    with open(path + ".part", "wb") as f:
        f.write(b"SFCQ")
        f.write(struct.pack("<III", 1, len(groups), nq))
        for g in groups:
            lab = g["label"].encode()
            f.write(struct.pack("<I", len(lab)))
            f.write(lab)
            f.write(struct.pack("<dB", g["target"], g["kind"]))
            lohi = np.empty((nq, 2), dtype="<u4")
            lohi[:, 0] = g["lo"]
            lohi[:, 1] = g["hi"]
            f.write(lohi.tobytes())
    os.replace(path + ".part", path)


def frac_stats(counts, n):
    a = np.sort(np.asarray(counts, dtype=np.float64)) / n
    return {"median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9)),
            "mean": float(a.mean()), "min": float(a[0]), "max": float(a[-1])}


def range_groups(px, sels, nq, seed, tries, max_overshoot):
    """Rank windows on the sorted value array, as [lo, hi]. Each group has its own seed stream
    (seed, s)."""
    n = len(px)
    V = np.sort(px)
    groups, report = [], []
    for s in sels:
        m = int(round(n * s))
        if m < 1:
            raise SystemExit(f"selectivity {s} gives m={m} < 1")
        rng = np.random.default_rng([seed, int(round(s * 1e9))])
        lo = np.empty(nq, np.uint32)
        hi = np.empty(nq, np.uint32)
        realized = np.empty(nq, np.int64)
        attempts = np.empty(nq, np.int64)
        for q in range(nq):
            best = None
            for t in range(1, tries + 1):
                u = int(rng.integers(0, n - m + 1))
                a, b = int(V[u]), int(V[u + m - 1])
                # Probe with the array's own dtype. A Python int would make numpy promote the whole
                # 25M-element array to int64 on every call (seconds per query instead of microseconds).
                ab = np.array([a, b], dtype=V.dtype)
                cnt = int(np.searchsorted(V, ab[1], side="right") - np.searchsorted(V, ab[0], side="left"))
                if best is None or cnt < best[0]:
                    best = (cnt, a, b, t)
                if cnt <= max_overshoot * m:
                    break
            realized[q], lo[q], hi[q], attempts[q] = best[0], best[1], best[2], best[3]
        label = f"pixels {s * 100:g}%"
        groups.append({"label": label, "target": float(s), "kind": 1, "lo": lo, "hi": hi})
        report.append({"label": label, "kind": "range", "column": "pixels", "target": float(s), "m": m,
                       "realized": frac_stats(realized, n),
                       "overshoot_median": float(np.median(realized) / m),
                       "attempts_mean": float(attempts.mean()), "attempts_max": int(attempts.max()),
                       "queries_over_cap": int((realized > max_overshoot * m).sum())})
        print(f"  {label}: m={m} realized median {np.median(realized) / n * 100:.4g}% "
              f"(overshoot x{np.median(realized) / m:.3f}, attempts mean {attempts.mean():.2f})", flush=True)
    return groups, report, V


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--nq", type=int, default=992)
    ap.add_argument("--meta-dir", required=True, help="directory with metadata_<i>.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sels", default="10%,5%,3%,1%,0.5%,0.1%")
    ap.add_argument("--tries", type=int, default=64)
    ap.add_argument("--max-overshoot", type=float, default=1.3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    n, nq = args.n, args.nq
    sels = [float(x.rstrip("%")) / 100 if x.endswith("%") else float(x) for x in args.sels.split(",")]

    shards = shards_for(args.meta_dir, n + nq)
    print(f"reading {len(shards)} shards for N={n} (+{nq} query rows unused)", flush=True)
    nsfw, lic, px = read_columns(shards, n)

    report = {"n": n, "nq": nq, "seed": args.seed, "shards": [os.path.basename(s) for s in shards],
              "max_overshoot": args.max_overshoot, "tries": args.tries, "columns": {}, "groups": []}

    # categorical columns: the predicate is the same for every query, the data sets the selectivity
    cat_specs = [("nsfw", nsfw, [("NSFW=UNSURE", 1), ("NSFW=NSFW", 2)]),
                 ("license", lic, [("LICENSE=CC", 1)])]
    for name, col, preds in cat_specs:
        counts = np.bincount(col, minlength=4)
        report["columns"][name] = {"value_counts": {str(int(v)): int(c) for v, c in enumerate(counts) if c}}
        if name == "nsfw":
            report["columns"][name]["value_names"] = NSFW_NAMES
        groups = []
        for label, code in preds:
            cnt = int(counts[code]) if code < len(counts) else 0
            groups.append({"label": label, "target": float("nan"), "kind": 0,
                           "lo": np.full(nq, code, np.uint32), "hi": np.full(nq, code, np.uint32)})
            report["groups"].append({"label": label, "kind": "categorical", "column": name, "code": code,
                                     "target": None, "m": cnt, "realized": frac_stats([cnt], n)})
            print(f"  {label}: {cnt} rows = {cnt / n * 100:.4g}%", flush=True)
        write_sfcl(os.path.join(args.out_dir, f"laion_{n}_meta_{name}.sfcl"), col)
        write_sfcq(os.path.join(args.out_dir, f"laion_{n}_meta_{name}.sfcq"), groups, nq)

    # numeric range column: natural attribute, controlled selectivity, random window placement
    groups, rep, V = range_groups(px, sels, nq, args.seed, args.tries, args.max_overshoot)
    report["groups"].extend(rep)
    qs = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    report["columns"]["pixels"] = {"distinct": int(len(np.unique(V))),
                                   "quantiles": {str(q): int(V[min(len(V) - 1, int(q * len(V)))]) for q in qs},
                                   "zero_rows": int((V == 0).sum())}
    write_sfcl(os.path.join(args.out_dir, f"laion_{n}_meta_pixels.sfcl"), px)
    write_sfcq(os.path.join(args.out_dir, f"laion_{n}_meta_pixels.sfcq"), groups, nq)

    with open(os.path.join(args.out_dir, f"laion_{n}_meta_workload.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
