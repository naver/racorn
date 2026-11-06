#!/usr/bin/env python3
# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
"""LAION natural filter: the caption-keyword predicate workload.

With `--all-queries-per-bucket` (the Section VI-D form) **every query enters every bucket**, so
that a cell has as many queries as in Section VI-A. Each bucket holds nq queries. Each query is
assigned a predicate of that bucket by a seeded round robin over an independent permutation per
bucket. Without the flag, the queries are split across the buckets (about nq / 6 per bucket).
Word choice, OR combinations and passing-key sets are deterministic in the seed and identical
in both forms. Only the query assignment differs. The file tag is `all`, `own` (--own-caption) or
`kw`.

Captions are read from metadata_N.parquet of deploy.laion.ai (the same shard and row order as
img_emb_N.npy). As in the embedding conversion (laion_to_fvecs.py), the first N rows are the
base (keys 0..N-1) and the next nq rows are the queries. The predicate is "the caption contains
the keyword as a word", a token match equivalent to the regex \\bw\\b. The high-selectivity
buckets that no single word fills are completed with OR combinations \\b(w1|w2|...)\\b.

Steps
  1. Download the shards in order, deterministically (retry on failure, then abort so that the row order is kept).
  2. Pass 1: tokenize the base captions and count the document frequency (DF) of each word.
  3. Pick up to --per-bucket words whose DF/N falls in each selectivity bucket. Fill a short bucket
     with OR combinations of 2-6 words from the DF band one step below (the union sizes are verified
     after pass 2).
  4. Pass 2: build the passing-key list of each needed word, and the unions of the OR predicates.
  5. Assign the queries to the buckets (uniformly, or all queries to every bucket), and the predicates
     within a bucket round robin. --own-caption prefers a word of the query's own caption whose DF is
     in that bucket: a naturally correlated workload.
  6. Write one SFAL file per bucket (bench_synthfilter --allowlist-spec) with a side-car JSON, and
     `_buckets.json` over all buckets. With --emit-predicate-files, also write the per-bucket .sfpr
     predicates, the per-row token CSR (.spmat) and the vocabulary (.vocab).

SFAL: magic "SFAL", u32 version=1, u32 nq, u64 N, then per entry u32 qidx, u32 cnt, cnt x u32 key.

Usage:
  python3 laion_captions.py --n 1000000 --nq 992 --meta-dir $ROOT/data/laion_meta \\
      --out-dir $ROOT/data/raw-laion1m/caption64 --all-queries-per-bucket --emit-predicate-files
"""
import argparse, array, collections, json, os, random, re, struct, sys, time, urllib.request

META = "https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata"
TOKEN = re.compile(r"[a-z]+")
# Common function words and the non-semantic tokens that pervade captions. Excluded from the predicate
# candidates only.
STOP = set("""a an the and or of to in on for with by at from as is are was were be been being this that these those
it its into over under about after before between during without within up down out off than then there here
their them they he she his her him you your we our us i me my not no yes all any each few more most other some
such only own same so too very can will just don should now new one two three four five six seven eight nine ten
first second best top free big small large little high low long short old young good great per via vs
jpg jpeg png gif webp http https www com net org html php aspx img image images picture pictures photo photos
stock vector illustration royalty file files download view size pixel pixels wallpaper background thumbnail
""".split())


def download(i, meta_dir, retries):
    local = os.path.join(meta_dir, f"metadata_{i}.parquet")
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local
    url = f"{META}/metadata_{i}.parquet"
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(local + ".part", "wb") as out:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            os.replace(local + ".part", local)
            return local
        except Exception as e:
            print(f"  metadata_{i} attempt {attempt}/{retries} failed ({e})", flush=True)
            time.sleep(10 * attempt)
    raise SystemExit(f"metadata_{i} could not be downloaded; aborting to keep row order deterministic")


def captions(shards, need):
    """Yields (global_row, caption_lower) in shard order, up to `need` rows."""
    import pyarrow.parquet as pq
    row = 0
    for local in shards:
        col = pq.read_table(local, columns=["caption"]).column("caption").to_pylist()
        for c in col:
            if row >= need:
                return
            yield row, (c or "").lower()
            row += 1


def tokens(caption):
    return set(t for t in TOKEN.findall(caption) if len(t) >= 3 and t not in STOP)


def bucket_label(lo, hi):
    def f(x):
        return f"{x*100:g}"
    return f"{f(lo)}-{f(hi)}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--meta-dir", required=True, help="directory of the metadata parquet shards (shared across N)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bucket-edges", default="0.0002,0.001,0.005,0.01,0.03,0.05,0.10",
                    help="selectivity bucket edges (fractions); adjacent pairs form the buckets")
    ap.add_argument("--per-bucket", type=int, default=30, help="maximum number of predicates per bucket")
    ap.add_argument("--max-or", type=int, default=6, help="maximum number of words in an OR combination")
    ap.add_argument("--own-caption", action="store_true",
                    help="prefer a word of the query's own caption as its predicate (naturally correlated workload)")
    ap.add_argument("--emit-predicate-files", action="store_true",
                    help="also write the predicate structures the scanned and indexed pre-filter forms need: the "
                         "per-row token CSR (.spmat), the per-bucket query predicates (.sfpr) and the vocabulary (.vocab)")
    ap.add_argument("--all-queries-per-bucket", action="store_true",
                    help="put every query in every bucket (nq queries per bucket, tag 'all'); the default splits the queries")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.meta_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    N, NQ = args.n, args.nq
    need = N + NQ
    edges = [float(x) for x in args.bucket_edges.split(",")]
    buckets = list(zip(edges[:-1], edges[1:]))
    rng = random.Random(args.seed)

    # 1. shards (the row count is only known after reading, so shards are fetched in order until `need` is covered)
    import pyarrow.parquet as pq
    shards, rows, i = [], 0, 0
    while rows < need:
        local = download(i, args.meta_dir, args.retries)
        shards.append(local)
        rows += pq.read_metadata(local).num_rows
        print(f"  metadata_{i}: cumulative rows {rows}/{need}", flush=True)
        i += 1

    # 2. pass 1: DF over base
    t0 = time.time()
    df = collections.Counter()
    qcaps = {}
    for row, cap in captions(shards, need):
        toks = tokens(cap)
        if row < N:
            df.update(toks)
        else:
            qcaps[row - N] = toks
    print(f"pass1: {len(df)} distinct tokens over {N} base captions ({time.time()-t0:.0f}s)", flush=True)

    # 3. predicate choice per bucket: single word -> chosen[word]=bucket; OR combination -> combos[name]=(bucket, words)
    chosen, combos, per_bucket = {}, {}, []
    for bi, (lo, hi) in enumerate(buckets):
        cands = sorted((w for w, c in df.items() if lo <= c / N < hi), key=lambda w: df[w])
        if len(cands) > args.per_bucket:
            step = len(cands) / args.per_bucket
            cands = [cands[int(j * step)] for j in range(args.per_bucket)]
        per_bucket.append(list(cands))
        for w in cands:
            chosen[w] = bi
    for bi, (lo, hi) in enumerate(buckets):
        if len(per_bucket[bi]) >= args.per_bucket // 3:
            continue
        # Combine 2..max_or words from the DF band one step below so that the summed DF lands in the bucket
        # (the union is verified after pass 2).
        pool = sorted((w for w, c in df.items() if lo / args.max_or <= c / N < lo and w not in chosen),
                      key=lambda w: df[w])
        tries = 0
        while len(per_bucket[bi]) < args.per_bucket and tries < 20000 and len(pool) >= 2:
            tries += 1
            ws = tuple(sorted(rng.sample(pool, rng.randint(2, min(args.max_or, len(pool))))))
            tot = sum(df[w] for w in ws) / N
            name = "|".join(ws)
            if lo <= tot < hi and name not in combos:
                combos[name] = (bi, ws)
                per_bucket[bi].append(name)
    for bi, (lo, hi) in enumerate(buckets):
        n_or = sum(1 for p in per_bucket[bi] if "|" in p)
        print(f"  bucket {bucket_label(lo, hi)}: {len(per_bucket[bi])} predicates ({n_or} OR), "
              f"e.g. {per_bucket[bi][:4]}", flush=True)
        if not per_bucket[bi]:
            print(f"  !! no predicate for bucket {bucket_label(lo, hi)}", file=sys.stderr)

    # 4. pass 2: passing keys per needed word, then unions for OR predicates
    need_words = set(chosen) | {w for _, ws in combos.values() for w in ws}
    t0 = time.time()
    keys = {w: [] for w in need_words}
    # The scanned pre-filter evaluates the predicate on every row without an index, so the driver needs the
    # per-row token sets. The passing-key list (SFAL) is the *result* of that evaluation and cannot stand in
    # for it. The CSR is written in the same sweep. A token id is the position of the word in the
    # lexicographically sorted vocabulary. Each row's ids are stored in ascending order so that they can be
    # binary-searched.
    emit = args.emit_predicate_files
    if emit:
        vocab = sorted(df)
        tok_id = {w: i for i, w in enumerate(vocab)}
        csr_idx = open(os.path.join(args.out_dir, f"laion_{N}_caption_tokens.idx.part"), "wb")
        indptr = array.array("q", [0])
        nnz = 0
    for row, cap in captions(shards, N):
        toks = tokens(cap)
        for w in toks & need_words:
            keys[w].append(row)
        if emit:
            ids = array.array("i", sorted(tok_id[w] for w in toks))
            ids.tofile(csr_idx)
            nnz += len(ids)
            indptr.append(nnz)
    if emit:
        csr_idx.close()
        # big-ann .spmat layout: [int64 nrow][int64 ncol][int64 nnz][int64 indptr[nrow+1]][int32 indices[nnz]].
        # data[] (float32 x nnz) would be all 1.0 and the loader does not read it, so it is not written.
        csr_path = os.path.join(args.out_dir, f"laion_{N}_caption_tokens.spmat")
        with open(csr_path, "wb") as f:
            f.write(struct.pack("<qqq", N, len(vocab), nnz))
            indptr.tofile(f)
            with open(csr_idx.name, "rb") as g:
                while True:
                    chunk = g.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
        os.remove(csr_idx.name)
        with open(os.path.join(args.out_dir, f"laion_{N}_caption_tokens.vocab"), "w") as f:
            f.write("\n".join(vocab))
        print(f"  token CSR: {N} rows, {len(vocab)} tokens, {nnz} entries "
              f"({nnz / N:.1f} per row) -> {csr_path}", flush=True)
    for w in need_words:
        assert len(keys[w]) == df[w], (w, len(keys[w]), df[w])
    for name, (bi, ws) in list(combos.items()):
        u = sorted(set().union(*(keys[w] for w in ws)))
        lo, hi = buckets[bi]
        if lo <= len(u) / N < hi:
            keys[name] = u
        else:  # co-occurrence shrank the union out of the bucket -> dropped
            per_bucket[bi].remove(name)
            del combos[name]
    size = {p: len(keys[p]) for p in keys}
    print(f"pass2: passing sets for {len(need_words)} words, {len(combos)} OR predicates ({time.time()-t0:.0f}s)",
          flush=True)

    # 5. query assignment: buckets evenly, predicates within a bucket round robin (own-caption prefers the query's words)
    order = list(range(NQ))
    rng.shuffle(order)
    nb = len(buckets)
    assign, rr = {}, [0] * nb
    for pos, q in enumerate(order):
        bi = pos % nb
        cands = per_bucket[bi]
        if not cands:
            continue
        p = None
        if args.own_caption:
            own = [t for t in qcaps.get(q, ()) if chosen.get(t) == bi]
            if own:
                p = rng.choice(sorted(own))
        if p is None:
            p = cands[rr[bi] % len(cands)]
            rr[bi] += 1
        assign[q] = (bi, p)
    # bassign[bi] = {q: predicate}. The split assignment carries `assign` over as it is.
    bassign = [dict() for _ in range(nb)]
    if args.all_queries_per_bucket:
        # Every query x every bucket. Predicates are assigned round robin over an independent seeded permutation
        # per bucket. So a query gets a different word in each bucket, and the predicates of a bucket receive
        # equal numbers of queries (+-1). Only the query-independent assignment (kw) is supported.
        if args.own_caption:
            raise SystemExit("--all-queries-per-bucket cannot be combined with --own-caption")
        for bi in range(nb):
            cands = per_bucket[bi]
            if not cands:
                continue
            perm = list(range(NQ))
            random.Random(args.seed * 1000003 + bi).shuffle(perm)
            for pos, q in enumerate(perm):
                bassign[bi][q] = cands[pos % len(cands)]
    else:
        for q, (bi, p) in assign.items():
            bassign[bi][q] = p

    # 6. SFAL per bucket + side-car
    tag = "all" if args.all_queries_per_bucket else ("own" if args.own_caption else "kw")
    kbytes = {}  # predicate -> SFAL payload, packed once (in the 5-10% bucket a query lists about a tenth of the keys)
    summary = []
    for bi, (lo, hi) in enumerate(buckets):
        ba = bassign[bi]
        qs = sorted(ba)
        if not qs:
            continue
        label = bucket_label(lo, hi)
        slug = label.replace("%", "").replace("-", "_").replace(".", "p")
        path = os.path.join(args.out_dir, f"laion_{N}_caption_{tag}_{slug}.sfal")
        with open(path, "wb") as f:
            f.write(b"SFAL")
            f.write(struct.pack("<IIQ", 1, len(qs), N))
            for q in qs:
                name = ba[q]
                if name not in kbytes:
                    kbytes[name] = array.array("I", keys[name]).tobytes()
                f.write(struct.pack("<II", q, size[name]))
                f.write(kbytes[name])
        sel = [size[ba[q]] / N for q in qs]
        side = {"label": label, "lo": lo, "hi": hi, "n": N, "nq": len(qs), "mode": tag,
                "mean_selectivity": sum(sel) / len(sel),
                "queries": [{"q": q, "predicate": ba[q], "passing": size[ba[q]],
                             "selectivity": size[ba[q]] / N} for q in qs]}
        with open(path[:-5] + ".json", "w") as f:
            json.dump(side, f, indent=1)
        if emit:
            # SFPR: magic "SFPR", u32 version=1, u32 nq, u32 ntokens, then per entry
            # u32 qidx, u8 op (0=AND, 1=OR), u32 cnt, cnt x u32 token_id.
            # A single-word predicate is an AND with one entry; an OR combination is an OR with several.
            with open(path[:-5] + ".sfpr", "wb") as f:
                f.write(b"SFPR")
                f.write(struct.pack("<III", 1, len(qs), len(vocab)))
                for q in qs:
                    name = ba[q]
                    ws = name.split("|")
                    f.write(struct.pack("<IBI", q, 1 if len(ws) > 1 else 0, len(ws)))
                    f.write(struct.pack(f"<{len(ws)}I", *(tok_id[w] for w in ws)))
        npred = len(set(ba[q] for q in qs))
        summary.append({"label": label, "nq": len(qs), "mean_selectivity": side["mean_selectivity"],
                        "predicates": npred, "file": path})
        print(f"wrote {path}: {len(qs)} queries, mean sel {side['mean_selectivity']*100:.3f}%, {npred} predicates",
              flush=True)
    with open(os.path.join(args.out_dir, f"laion_{N}_caption_{tag}_buckets.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
