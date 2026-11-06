#!/usr/bin/env python3
# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
"""Section VI-D heterogeneous workload: caption keyword AND original-resolution window.

Two real LAION attributes are joined in one predicate. One is a text attribute: a caption token
match, a label-type predicate. The other is a numeric range: the original resolution
`original_width x original_height`, a range-type predicate. A single-type dedicated index covers
one of the two, so the conjunction is not representable as a structure.

Inputs (files of the same N):
  . caption buckets (written by `laion_captions.py --all-queries-per-bucket --emit-predicate-files`)
      <cap>/laion_<N>_caption_all_<slug>.{sfal,sfpr}   per query, the keyword predicate and its passing keys
      <cap>/laion_<N>_caption_all_<slug>.json          the bucket's side-car, read for its label
  . resolution column (written by `laion_column_workload.py` of Section VI-C)
      <raw>/meta/laion_<N>_meta_pixels.sfcl           u32 pixel count per key

Groups (keyword bucket x marginal of the resolution window):
  H1  5-10% x 30%    H2  5-10% x 10%    H3  1-3% x 10%    H4  1-3% x 3%

The window follows Section VI-C. A rank interval [u, u+m) of the sorted values is drawn with a
fixed seed, and its end values become [lo, hi] (m = round(N*w)). When ties make the realized
marginal exceed the target, the window is redrawn (`--max-overshoot`). A window whose joint
passing set is smaller than k (100) is also redrawn, since recall would be undefined. After
`--tries` draws, the window with the largest passing set is kept and the query is counted in the
side-car.

Outputs (<cap>/):
  laion_<N>_hetero_<G>.sfal  joint passing keys (read by the graph arms, the given-set pre-filter and the GT)
  laion_<N>_hetero_<G>.sfpr  the keyword predicate of each query (the SFPR entry of the caption bucket, copied)
  laion_<N>_hetero_<G>.sfrq  the resolution window of each query: "SFRQ", u32 1, u32 nq, then (u32 qidx, u32 lo, u32 hi) x nq
  laion_<N>_hetero_<G>.json  label, configuration, realized joint and marginal selectivity distributions, redraw statistics
  laion_<N>_hetero_groups.json  the side-cars of all groups in one list

Usage:
  python3 hetero_workload.py --n 25000000 --cap-dir data/raw-laion25m/caption64 \\
      --pixels data/raw-laion25m/meta/laion_25000000_meta_pixels.sfcl
"""
import argparse
import json
import os
import struct
import time

import numpy as np

GROUPS = [  # (name, caption bucket slug, window marginal)
    ("H1", "5_10", 0.30),
    ("H2", "5_10", 0.10),
    ("H3", "1_3", 0.10),
    ("H4", "1_3", 0.03),
]


def read_sfal(path, n):
    with open(path, "rb") as f:
        magic = f.read(4)
        ver, nq, nn = struct.unpack("<IIQ", f.read(16))
        if magic != b"SFAL" or ver != 1 or nn != n:
            raise SystemExit(f"{path}: bad SFAL header (N={nn} vs {n})")
        qidx, sets = [], []
        for _ in range(nq):
            q, cnt = struct.unpack("<II", f.read(8))
            qidx.append(q)
            sets.append(np.frombuffer(f.read(4 * cnt), dtype="<u4"))
    return qidx, sets


def read_sfpr(path):
    """Returns the entries as raw bytes, for copying: (qidx, raw bytes)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        ver, nq, ntok = struct.unpack("<III", f.read(12))
        if magic != b"SFPR" or ver != 1:
            raise SystemExit(f"{path}: bad SFPR header")
        out = []
        for _ in range(nq):
            head = f.read(9)
            q, op, cnt = struct.unpack("<IBI", head)
            out.append((q, head + f.read(4 * cnt)))
    return ntok, out


def read_sfcl(path, n):
    with open(path, "rb") as f:
        magic = f.read(4)
        ver, nn, _ = struct.unpack("<IQI", f.read(16))
        if magic != b"SFCL" or ver != 1 or nn != n:
            raise SystemExit(f"{path}: bad SFCL header (N={nn} vs {n})")
        return np.fromfile(f, dtype="<u4", count=n)


def stats(a):
    a = np.sort(np.asarray(a, dtype=np.float64))
    return {"median": float(np.median(a)), "p10": float(np.quantile(a, 0.1)), "p90": float(np.quantile(a, 0.9)),
            "mean": float(a.mean()), "min": float(a[0]), "max": float(a[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cap-dir", required=True)
    ap.add_argument("--pixels", required=True)
    ap.add_argument("--k", type=int, default=100, help="smallest joint passing set (recall must be defined)")
    ap.add_argument("--tries", type=int, default=50)
    ap.add_argument("--max-overshoot", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--groups", default="H1,H2,H3,H4")
    args = ap.parse_args()
    N = args.n
    t0 = time.time()
    V = read_sfcl(args.pixels, N)
    Vs = np.sort(V)
    print(f"pixels: {N} values, {len(np.unique(Vs))} distinct ({time.time() - t0:.0f}s)", flush=True)
    cache = {}
    summary = []
    want = set(args.groups.split(","))
    for gi, (name, slug, w) in enumerate(GROUPS):
        if name not in want:
            continue
        base = os.path.join(args.cap_dir, f"laion_{N}_caption_all_{slug}")
        if slug not in cache:
            qidx, sets = read_sfal(base + ".sfal", N)
            ntok, pr = read_sfpr(base + ".sfpr")
            if [q for q, _ in pr] != qidx:
                raise SystemExit(f"{base}: SFPR query order differs from SFAL")
            with open(base + ".json") as f:
                kwlabel = json.load(f)["label"]
            cache[slug] = (qidx, sets, ntok, pr, kwlabel)
        qidx, sets, ntok, pr, kwlabel = cache[slug]
        nq = len(qidx)
        m = int(round(N * w))
        rng = np.random.default_rng([args.seed, gi])
        label = f"{name}: kw {kwlabel} & px {w * 100:g}%"
        out = os.path.join(args.cap_dir, f"laion_{N}_hetero_{name}")
        joint, wmarg, kwmarg, attempts = [], [], [], []
        short = 0
        t1 = time.time()
        with open(out + ".sfal.part", "wb") as fa, open(out + ".sfrq.part", "wb") as fr, \
                open(out + ".sfpr.part", "wb") as fp:
            fa.write(b"SFAL" + struct.pack("<IIQ", 1, nq, N))
            fr.write(b"SFRQ" + struct.pack("<II", 1, nq))
            fp.write(b"SFPR" + struct.pack("<III", 1, nq, ntok))
            for i, q in enumerate(qidx):
                ks = sets[i]
                vk = V[ks]
                best = None
                for t in range(1, args.tries + 1):
                    u = int(rng.integers(0, N - m + 1))
                    lo, hi = Vs[u], Vs[u + m - 1]
                    wc = int(np.searchsorted(Vs, hi, side="right") - np.searchsorted(Vs, lo, side="left"))
                    sel = (vk >= lo) & (vk <= hi)
                    jc = int(sel.sum())
                    ok = jc >= args.k and wc <= args.max_overshoot * m
                    if best is None or (ok and not best[0]) or (ok == best[0] and jc > best[1]):
                        best = (ok, jc, int(lo), int(hi), wc, sel, t)
                    if ok:
                        break
                ok, jc, lo, hi, wc, sel, t = best
                short += 0 if ok else 1
                jk = ks[sel]
                fa.write(struct.pack("<II", q, len(jk)) + jk.astype("<u4").tobytes())
                fr.write(struct.pack("<III", q, lo, hi))
                fp.write(pr[i][1])
                joint.append(jc / N)
                wmarg.append(wc / N)
                kwmarg.append(len(ks) / N)
                attempts.append(t)
        for ext in (".sfal", ".sfrq", ".sfpr"):
            os.replace(out + ext + ".part", out + ext)
        side = {"label": label, "group": name, "n": N, "nq": nq, "keyword_bucket": kwlabel, "window_marginal": w,
                "joint": stats(joint), "keyword_marginal": stats(kwmarg), "window_realized": stats(wmarg),
                "attempts_mean": float(np.mean(attempts)), "attempts_max": int(np.max(attempts)),
                "queries_not_meeting_constraints": short, "k": args.k, "max_overshoot": args.max_overshoot,
                "seed": args.seed}
        with open(out + ".json", "w") as f:
            json.dump(side, f, indent=1)
        summary.append(side)
        print(f"{label}: joint median {side['joint']['median'] * 100:.4f}% (p10 {side['joint']['p10'] * 100:.4f}, "
              f"p90 {side['joint']['p90'] * 100:.4f}), min passing {int(side['joint']['min'] * N)}, window median "
              f"{side['window_realized']['median'] * 100:.2f}%, unmet {short}, attempts mean "
              f"{side['attempts_mean']:.2f} ({time.time() - t1:.0f}s)", flush=True)
    with open(os.path.join(args.cap_dir, f"laion_{N}_hetero_groups.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
