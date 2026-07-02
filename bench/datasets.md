# Dataset Acquisition Guide

bench_acorn supports four standard ANN benchmark datasets. This document
covers where each dataset comes from and how to lay it out under
`data/raw/` so that `bench_acorn` can find it without overrides.

| Dataset | Origin | Format | Dim | File rows | Effective N at run time | Queries | Metric |
|---|---|:-:|---:|---:|---:|---:|---|
| sift_1m         | INRIA TexMex (BIGANN-SIFT1M) | fvecs | 128 |  1,000,000 |  1,000,000 |  10,000 | L2 |
| gist_1m         | INRIA TexMex (GIST1M)        | fvecs | 960 |  1,000,000 |  1,000,000 |   1,000 | L2 |
| text2image_1m † | Yandex T2I                   | fbin  | 200 | 40,000,000 |  1,000,000 | 100,000 | cosine |
| text2image_40m  | Yandex T2I                   | fbin  | 200 | 40,000,000 | 40,000,000 | 100,000 | cosine |

† `text2image_1m` and `text2image_40m` share the same underlying file
(`text2image-40M.fbin`); the difference is the `--n` argument passed to
`bench_acorn` at run time (`--n 1000000` vs `--n 40000000`).

---

## Target directory layout

All benchmark scripts `cd` into `bench/` before running, so the data
directory must be placed at `bench/data/raw/` relative to the repository
root. After completing the steps below, the layout should be:

```text
bench/data/raw/
├── sift_base.fvecs              # 1M × 128 (float32, ~512 MB)
├── sift_query.fvecs             # 10K × 128 (~5 MB)
├── gist_base.fvecs              # 1M × 960 (~3.8 GB)
├── gist_query.fvecs             # 1K × 960 (~3.8 MB)
├── text2image-40M.fbin          # 200d float32 — size depends on §3 recipe
└── text2image-query-100K.fbin   # 100K × 200 (~80 MB)
```

Override with `--data-dir <path>` if you keep the files elsewhere.

> Total disk budget ranges from **~5 GB** (sift_1m + gist_1m + a
> truncated 1M text2image file) up to **~37 GB** (all four datasets
> with a paper-grade 40M text2image file). The text2image file
> dominates the budget — see §3 for the recipe options.

---

## 1. SIFT 1M (BIGANN-SIFT1M)

A 128-dimensional float32 dataset derived from the original BIGANN
collection by INRIA. Standard L2 benchmark for nearest-neighbour search.

**Source**: http://corpus-texmex.irisa.fr/ → "ANN_SIFT1M" → `sift.tar.gz`
(The page labels the row "ANN_SIFT1M" but the actual archive filename
is `sift.tar.gz`.)

The TexMex archive is served over **FTP only** (no HTTP mirror is
available on the source page). If your network blocks outbound FTP,
fetch the tarball on a different host and copy it into
`bench/data/raw/`.

```bash
mkdir -p bench/data/raw && cd bench/data/raw

wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

tar -xzf sift.tar.gz       # extracts into sift/ (some mirrors: ANN_SIFT1M/)
mv sift/sift_base.fvecs sift/sift_query.fvecs .
rm -rf sift sift.tar.gz
```

Sanity check (the first 4 bytes of an `.fvecs` file are the per-row
dimension):

```bash
xxd -l 4 sift_base.fvecs
# Expect: "8000 0000" (little-endian 128)
```

Files in the archive that we do **not** use (you can delete):
`sift_learn.fvecs`, `sift_groundtruth.ivecs`. (`bench_acorn --build-gt`
computes its own ground-truth.)

---

## 2. GIST 1M

960-dimensional float32 dataset of GIST image descriptors. Same INRIA
archive as SIFT but a separate tarball.

**Source**: http://corpus-texmex.irisa.fr/ → "ANN_GIST1M" → `gist.tar.gz`
(Same labelling caveat as SIFT.)

FTP-only, same as SIFT — copy in from another host if your network
blocks FTP.

```bash
cd bench/data/raw

wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz

tar -xzf gist.tar.gz       # extracts into gist/ (some mirrors: ANN_GIST1M/)
mv gist/gist_base.fvecs gist/gist_query.fvecs .
rm -rf gist gist.tar.gz
```

Sanity check:

```bash
xxd -l 4 gist_base.fvecs
# Expect: "c003 0000" (little-endian 960)
```

> GIST is much heavier than SIFT due to the 960-dim vectors — the
> uncompressed `gist_base.fvecs` is roughly **3.8 GB**. The download
> takes proportionally longer than SIFT.

---

## 3. Text-to-Image 40M (and 1M subset)

Yandex Text-to-Image is part of the NeurIPS 2021 / 2023 BigANN benchmark
track. Upstream hosts the **full 1B-vector** set on Yandex Cloud;
bench_acorn's `text2image_40m` and `text2image_1m` setups are both
**derived** from that single upstream file — there is no separate 40M
or 1M file to download directly.

**Source**: https://big-ann-benchmarks.com/ → "Yandex T2I"

### Finding the current upstream URL

The hosting URL has rotated since 2021, so this guide does not hard-
code a deep link. To get the current `base_url`:

- https://big-ann-benchmarks.com/ (look for "Datasets" → Yandex T2I), or
- https://github.com/harsha-simhadri/big-ann-benchmarks —
  `benchmark/datasets.py`, locate the `Text2Image1B` class and copy
  its `base_url` value.

As of **2026-05** the upstream bucket at `<BASE>` publishes:

| File | Size | Header `n` | Purpose |
|---|---:|---:|---|
| `base.1B.fbin`           | ~800 GB | 1,000,000,000 | Full base set — paper-grade `text2image_40m` derives from this |
| `base.10M.fbin`          | ~8 GB   |    10,000,000 | Cheaper alternative usable for `text2image_1m` only |
| `query.public.100K.fbin` | ~80 MB  |       100,000 | Query set, always required |
| ~~`base.40M.fbin`~~      | —       | —             | **Does not exist** — the 40M subset is derived by truncation |

### Recipes by target run

Pick one row based on which `DATASET=` you intend to run. The base
file must end up named `text2image-40M.fbin` regardless of how you
produced it — bench_acorn's loader path is hard-coded.

| Target run | Disk needed | Recipe |
|---|---|---|
| `text2image_1m` (light validation) | ~8 GB | Download `base.10M.fbin` → save as `text2image-40M.fbin`. bench_acorn `--n 1000000` only reads the first 1M; the rest is harmless filler. |
| `text2image_1m`, disk-minimal | <1 GB | as above, then run the [truncate-in-place recipe](#truncate-in-place) with `ROWS=1000000`. |
| `text2image_40m` (paper-grade) | ~32 GB (after truncation) | Download `base.1B.fbin` (~800 GB transient), then run the truncate-in-place recipe with `ROWS=40000000`. |
| Full 1B run | ~800 GB | Out of scope for bench_acorn; reference only. |

Always also fetch the query file:

```bash
cd bench/data/raw
wget -O text2image-query-100K.fbin <BASE>/query.public.100K.fbin
```

> **README's "Reproducing the Experiments" workflow** assumes the
> **40M → 1M chain on a single file**: build `text2image-40M.fbin` once
> (truncating the first 40M rows of `base.1B.fbin`), and the
> `text2image_1m` setup reads the same file with `--n 1000000`. No
> separate 1M file is produced — the 1M setup is just the byte prefix
> of the 40M file. Both subsets therefore share the same query set
> (`query.public.100K.fbin`) and 1M results are nested inside the 40M
> file.

### Verification

The `.fbin` format has an 8-byte header (`n: uint32, dim: uint32`)
followed by `n × dim × float32`. Verify the file you ended up with —
the `n` field depends on which recipe you ran:

```bash
xxd -l 8 text2image-40M.fbin
#  40M file  → "005a 6202 c800 0000"   n=40,000,000  dim=200
#  10M file  → "8096 9800 c800 0000"   n=10,000,000  dim=200
#  1M file   → "4042 0f00 c800 0000"   n=1,000,000   dim=200

xxd -l 8 text2image-query-100K.fbin
#  always    → "a086 0100 c800 0000"   n=100,000     dim=200
```

The second 4 bytes (`c8 00 00 00`) must be 200 — if not, the file
isn't Text-to-Image and you've downloaded the wrong dataset. If `n`
doesn't match what the recipe should have produced, re-check the
truncation step.

### Truncate-in-place

Shared step used by both the `text2image_40m` and the disk-minimal
`text2image_1m` recipes. Pass the target row count via `ROWS`; the
file is rewritten under the same name so bench_acorn's hard-coded
loader path keeps working:

```bash
ROWS=40000000   # or 1000000 for the disk-minimal text2image_1m recipe

python3 - <<PY
import numpy as np
rows = ${ROWS}
with open('text2image-40M.fbin', 'rb') as f:
    n = int(np.frombuffer(f.read(4), dtype='<i4')[0])
    d = int(np.frombuffer(f.read(4), dtype='<i4')[0])
    assert rows <= n, f"file has only {n} rows; cannot extract {rows}"
    vecs = np.fromfile(f, dtype='<f4', count=rows * d).reshape(rows, d)
with open('text2image-40M.fbin', 'wb') as f:
    np.array([rows, d], dtype='<i4').tofile(f)
    vecs.tofile(f)
PY
```

### Metric

The metric defaults to cosine for `--dataset text2image` (inner
product on L2-normalized vectors); override with `--metric l2` if
needed. Both `text2image_1m` and `text2image_40m` use cosine.

---

## Pre-built indexes, ground truth, and clusters

Index volumes, exact ground-truth, and cluster ID files can be built once
upfront so subsequent `bench_acorn` invocations skip those steps. See
**`README.md` → Reproducing the Experiments §1, §3** for the helper
scripts and recommended invocations:

- `bench/build_volumes.sh` — HNSW index volumes
- `bench/build_all_gt.sh` — exact ground truth
- `bench/scripts/build_all_clusters.sh` — per-node / per-query cluster
  IDs (requires `numpy scikit-learn scipy`)

Each helper takes an optional dataset filter (substring match) and
skips combos whose raw file is missing or whose output already exists,
so a partial setup is fine.

---

## License notes

- **SIFT / GIST**: INRIA TexMex corpus is publicly available for research;
  see http://corpus-texmex.irisa.fr/ for the upstream terms.
- **Text-to-Image (Yandex T2I)**: Released as part of the BigANN
  benchmark. See https://big-ann-benchmarks.com/ for the dataset's
  terms of use.

All datasets are **not redistributed** with this artifact — only the
acquisition recipe is included.
