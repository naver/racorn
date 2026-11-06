# Benchmark drivers and the experiments of the paper

The three drivers in this directory produce every measurement of the paper. This document lists
their options. For each experimental section, it gives the commands with the parameters the paper
used. It also gives the rule that turns the output into the plotted numbers. Each command measures
one dataset. To cover a section, run it once per dataset of that section.

## The drivers

All three share the index and the search options. They differ in where the predicate comes from.
Each driver lists its options with `--help`. `--json <path>` writes the result.
The result holds a `config` object with the parameters of the run, including the measured
base-layer degree `mavg`. It also holds one row in `results` per (method, selectivity or group,
`ef`, `br`) cell.

| Driver | Predicate | Sections |
|---|---|---|
| `bench_acorn` | a uniform random attribute per key from a seed (`--predicate random --seed`). It also builds the index (`--build-index`) and the ground truth (`--build-gt`). It also runs the deadline sweeps (`--et-anchor`) | IV-D, VI-A, VI-E, VI-F |
| `bench_synthfilter` | predicates on the real data: an allowlist with a set correlation to the query (`--alpha`), a metadata column (`--column-spec`), a conjunction of `j` attributes (`--filter conj`), or caption keywords (`--allowlist-spec`, `--predicate-csr`). It also runs the pre-filter forms that scan for the passing set or read it from a posting index | VI-B, VI-C, VI-D |
| `bench_realfilter` | the YFCC tag queries of the Big ANN filtered track. Its ground truth is checked against the track's (`--official-gt`) | VI-C |

| Option | Meaning |
|---|---|
| `--modes <list>` | the methods the paper reports: `hnsw`, `acorn-1`, `acorn-1-3hop`, `racorn-1`, `navix-local`, `prefilter`, `prefilter-scan` and `prefilter-index`. `navix-local` is NaviX's adaptive-local as the NaviX authors' Kuzu code runs it, with two exceptions: it does not switch to brute force when at most 30,000 nodes pass, and it does not return an empty result at selectivity 0.1% and below. `prefilter` is given the passing set. `prefilter-scan` obtains the set by scanning the predicate. `prefilter-index` reads the set from a posting index; it is in `bench_synthfilter` and `bench_realfilter` |
| `--selectivity <list>` | the selectivities of the controlled predicate |
| `--ef-values <list>`, `--recall-target <F>` | the `ef` values, in order, and the recall at which a series stops |
| `--bridge-budget-ratios <list>` | RACORN-1's bridge budget `br`, one series per value |
| `--filter-first-threshold <F>` | the entry gate. ACORN-1, its three-hop variant and RACORN-1 run filter-first when the share of passing nodes among the entry node and its one- and two-hop neighbors is at most this value. Otherwise they run as HNSW. The paper uses 0.3 |
| `--three-hop-threshold <F>` | the trigger of the three-hop variant |
| `--stride on\|off` | stride sampling (`on`) or prefix truncation (`off`) in two selections: the two-hop passing cap of ACORN-1 and RACORN-1, and RACORN-1's bridge pool |
| `--k`, `--nq`, `--runs`, `--warmup`, `--search-threads` | 100 neighbors, 992 queries, the median of three timed runs after two warm-up passes, 32 threads |

How the drivers set the search options of `index_search_config_t` (listed in `include/usearch/README.md`) for each method of `--modes`:

| Method | `filter_first_threshold` | `bridge_budget_ratio` | `navix_heuristic` | `three_hop_threshold` |
|---|:-:|:-:|:-:|:-:|
| `hnsw` | `-1` | — | `0` | `0` |
| `acorn-1` | `0.3` (`--filter-first-threshold`) | `0` | `0` | `0` |
| `acorn-1-3hop` | `0.3` | `0` | `0` | `1.0` (`--three-hop-threshold`) |
| `racorn-1` | `0.3` | `br > 0` (`--bridge-budget-ratios`; the paper's main results use `2`) | `0` | `0` |
| `navix-local` | ignored | — | `8` | `0` |

The pre-filtering methods do not search the index: the drivers obtain the passing set and scan it.
The NaviX dispatch comes before the filter-first decision, so `--filter-first-threshold` does not
change `navix-local`. The deadline of Section VI-E (`--et-anchor`) sets
`early_termination_time_threshold` for `hnsw` and `racorn-1` alike.

## Setup

Every command below runs from a working root that holds the raw vectors, the indexes, the ground
truth and the results:

```bash
export REPO=/path/to/this/repository    # holds build/bench/ after the build (README.md, Build)
export ROOT=/path/to/working/root       # will hold data/ and results/
export PATH=$REPO/build/bench:$PATH
export NUMA="numactl --cpunodebind=0 --membind=0"   # one NUMA node; leave empty on one socket
mkdir -p $ROOT/data/raw $ROOT/data/volumes $ROOT/data/gt $ROOT/results && cd $ROOT
```

The table below lists the datasets and the values the commands take for each. `README.md`,
Datasets, says where the files come from:

| Dataset | `D` | `N` | `DIM` | `RAW` | Files in `RAW` |
|---|---|---|---|---|---|
| SIFT 1M | `sift` | 1000000 | 128 | `data/raw` | `sift_base.fvecs`, `sift_query.fvecs` |
| GIST 1M | `gist` | 1000000 | 960 | `data/raw` | `gist_base.fvecs`, `gist_query.fvecs` |
| YFCC 1M | `yfcc` | 1000000 | 192 | `data/raw-yfcc10m` | the YFCC 10M files; the first million base vectors |
| YFCC 10M | `yfcc` | 10000000 | 192 | `data/raw-yfcc10m` | `base.10M.u8bin`, `query.public.100K.u8bin`, `base.metadata.10M.spmat`, `query.metadata.public.100K.spmat`, `GT.public.ibin` |
| LAION 1M | `laion` | 1000000 | 512 | `data/raw-laion1m` | `laion_base.fvecs`, `laion_query.fvecs` |
| LAION 25M | `laion` | 25000000 | 512 | `data/raw-laion25m` | `laion_base.fvecs`, `laion_query.fvecs` |
| Text2Image 40M | `text2image` | 40000000 | 200 | `data/raw` | `text2image-40M.fbin`, `text2image-query-100K.fbin` |
| SIFT 100M | `sift` | 100000000 | 128 | `data/raw-sift100m` | `sift_base.fvecs`, `sift_query.fvecs` |

Each block below sets these values for one dataset. `TAG` names the output file. Build the index
(`M = 16`, `ef_construction = 100`) and the shared ground truth once per dataset:

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
bench_acorn --build-index --dataset $D --n $N --m 16 --ef-construction 100 --build-threads 32 \
    --data-dir $RAW --volume-dir data/volumes
$NUMA bench_acorn --build-gt --predicate random --seed 12345 --dataset $D --n $N --m 16 --ef-construction 100 \
    --k 100 --nq 1000 --selectivity 10%,5%,3%,1%,0.5%,0.1% --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt
```

On SIFT 100M build the ground truth with `--nq 992`. Build the ground truth with the same binaries
as the measurements.

**One process per dataset.** Run each command below as one process with all of its methods.
Latencies are compared within a process.

**Plotted latency.** Figures 3 to 6 plot one latency for each method and cell. It is the lowest
median latency (`latency_ms`) among the `ef` values whose recall is at least 0.9. A method with no
such `ef` has no point. The pre-filter rows have `ef_search` 0 and one point per cell.

## Section VI-A (Figs. 2 and 3) and Table I of Section III-A

The selectivity sweep: seven methods, five selectivities, `ef` from 100 to 1600 until recall 0.97,
on GIST 1M, YFCC 10M, LAION 25M, Text2Image 40M and SIFT 100M.

```bash
D=gist N=1000000 DIM=960 RAW=data/raw TAG=gist1m
$NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
    --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
    --modes prefilter,prefilter-scan,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --selectivity 5%,3%,1%,0.5%,0.1% \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on \
    --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec61_$TAG.json
```

- **Fig. 2**: `qps` against `recall` for each method and selectivity, over the `ef` rows. The
  selectivities are 3%, 1%, 0.5% and 0.1%.
- **Fig. 3**: HNSW's plotted latency divided by RACORN-1's, for each selectivity from 5% to 0.1%.
- **Table I**: ACORN-1 and HNSW at `ef` 100 on GIST 1M, LAION 25M and SIFT 100M. The values are
  `recall`, `never_full_rate` and `latency_p50`. `never_full_rate` is the share of queries that
  return fewer than `k` results.

## Section IV-D (Tables III and IV)

Run A measures HNSW, ACORN-1 and RACORN-1 at `br` 1 and 2, on all eight datasets. It uses two fixed
`ef` values and no recall target.

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
$NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
    --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
    --modes hnsw,acorn-1,racorn-1 --selectivity 10%,5%,3%,1%,0.5%,0.1% \
    --ef-values 100,200 --bridge-budget-ratios 1.0,2.0 \
    --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on \
    --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec44_$TAG.json
```

Each table cell is the minimum and maximum over the eight datasets. Ratios are taken within a
dataset. **Table III** (`ef` 100, selectivities 5% to 0.1%): `recall` and `expand_cycles` of RACORN-1
at `br` 1 and 2, and of HNSW. `expand_cycles` is the count of base-layer cycles. Table III lists
ACORN-1 as `br` 0.
**Table IV** (`ef` 100 and 200): HNSW divided by RACORN-1 for `expand_cycles`, `dist_computed` and
`latency_p50`; RACORN-1 divided by HNSW for `predicate_evals`; and `T/bound`. `T/bound` is
RACORN-1's `expand_cycles` divided by `(ef / s) * br / mavg`, with `mavg` from `config`. The rows
also record `prefill_ms_*`, the time at which the result queue fills, and `recall_at_fill`, the
recall at that moment. The text of the section quotes these two values.

Run B checks saturation, on SIFT 1M and GIST 1M. Its selectivities are not in the shared ground
truth, so it builds its own ground truth, in a directory of its own:

```bash
D=sift N=1000000 DIM=128 RAW=data/raw TAG=sift1m
$NUMA bench_acorn --build-gt --predicate random --seed 12345 --dataset $D --n $N --m 16 --ef-construction 100 \
    --k 100 --nq 1000 --selectivity 0.1%,0.05%,0.01% --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt-44-sat
$NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
    --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
    --modes hnsw,racorn-1 --selectivity 0.1%,0.05%,0.01% \
    --ef-values 100,200 --bridge-budget-ratios 1.0 \
    --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on \
    --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt-44-sat \
    --json results/sec44_sat_$TAG.json
```

## Section VI-B (Fig. 4)

The correlation sweep over `alpha`, on all eight datasets. `--ce on` records the measured
correlation of every group. This is the paper's c_r, printed as `c_e`.

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
$NUMA bench_synthfilter --dataset $D --n $N --d $DIM --m 16 --ef-construction 100 \
    --seed 12345 --alpha -0.8,-0.4,-0.2,0 --selectivity 1%,0.1% --k 100 --nq 992 \
    --modes prefilter,prefilter-scan,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on --ce on \
    --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec62_$TAG.json
```

**Fig. 4**: the plotted latency at 1% against `alpha`, on five of the eight: GIST 1M, YFCC 10M,
LAION 25M, Text2Image 40M and SIFT 100M.

## Section VI-C (Fig. 5)

The filters that come with the datasets. On YFCC 10M, the official tag queries of the Big ANN
filtered track, grouped by selectivity (`--bucket-edges`):

```bash
$NUMA bench_realfilter --n 10000000 --m 16 --ef-construction 100 --k 100 --nq 992 \
    --modes prefilter,prefilter-scan,prefilter-index,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on --ce on \
    --runs 3 --warmup 2 --search-threads 32 \
    --bucket-edges 0.001,0.005,0.01,0.03,0.05,1.0 --official-gt data/raw-yfcc10m/GT.public.ibin \
    --data-dir data/raw-yfcc10m --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec63_yfcc10m.json
```

On LAION 1M and 25M, three metadata columns: NSFW and LICENSE as equality predicates, and windows
on the original pixel count. `tools/laion_column_workload.py` writes the column and query files
from the LAION metadata (`README.md`, Datasets):

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
python3 $REPO/tools/laion_column_workload.py --n $N --nq 992 --meta-dir data/laion_meta --out-dir $RAW/meta
C=$RAW/meta/laion_${N}_meta
$NUMA bench_synthfilter --dataset laion --n $N --d 512 --m 16 --ef-construction 100 --k 100 --nq 992 \
    --modes prefilter,prefilter-scan,prefilter-index,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on --ce on \
    --runs 3 --warmup 2 --search-threads 32 --filter column \
    --column-spec ${C}_nsfw.sfcl:${C}_nsfw.sfcq \
    --column-spec ${C}_license.sfcl:${C}_license.sfcq \
    --column-spec ${C}_pixels.sfcl:${C}_pixels.sfcq \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec63_$TAG.json
```

**Fig. 5**: the plotted latency of each group against its actual selectivity (`selectivity` of its
rows). There are three panels: the YFCC tag buckets, the pixel-count windows
of LAION 25M, and the three categorical groups of LAION 25M. LAION 1M runs the same workload at a
smaller size and is not plotted.

## Section VI-D (Fig. 6)

Four workloads. The allowlist is the `alpha` 0 rows of Section VI-B and has no run of its own. The
other three run with the posting-index pre-filter as well. The first is the conjunction of the first
`j` of six synthetic attributes of cardinality 3. Its selectivity is `3^-j`. It runs on GIST 1M,
YFCC 10M, LAION 25M, Text2Image 40M and SIFT 100M. `--ce on` is set on GIST 1M and YFCC 10M only:

```bash
D=gist N=1000000 DIM=960 RAW=data/raw TAG=gist1m CE=on
$NUMA bench_synthfilter --dataset $D --n $N --d $DIM --m 16 --ef-construction 100 --k 100 --nq 992 \
    --modes prefilter,prefilter-scan,prefilter-index,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on \
    --seed 12345 --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --ce $CE --filter conj --conj-attrs 6 --conj-card 3 --conj-k 1,2,3,4,5,6 \
    --json results/sec64_conj_$TAG.json
```

The second is the caption keywords, on LAION 1M and 25M, in six selectivity buckets. Every query is
in every bucket. `tools/laion_captions.py` writes the token index. Per bucket, it also writes each
query's predicate and passing keys:

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
python3 $REPO/tools/laion_captions.py --n $N --nq 992 --meta-dir data/laion_meta --out-dir $RAW/caption64 \
    --all-queries-per-bucket --emit-predicate-files
C=$RAW/caption64/laion_${N}_caption
$NUMA bench_synthfilter --dataset $D --n $N --d $DIM --m 16 --ef-construction 100 --k 100 --nq 992 \
    --modes prefilter,prefilter-scan,prefilter-index,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on \
    --seed 12345 --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --ce on --predicate-csr ${C}_tokens.spmat \
    --allowlist-spec "${C}_all_0p02_0p1.sfal|0.02-0.1%|${C}_all_0p02_0p1.sfpr" \
    --allowlist-spec "${C}_all_0p1_0p5.sfal|0.1-0.5%|${C}_all_0p1_0p5.sfpr" \
    --allowlist-spec "${C}_all_0p5_1.sfal|0.5-1%|${C}_all_0p5_1.sfpr" \
    --allowlist-spec "${C}_all_1_3.sfal|1-3%|${C}_all_1_3.sfpr" \
    --allowlist-spec "${C}_all_3_5.sfal|3-5%|${C}_all_3_5.sfpr" \
    --allowlist-spec "${C}_all_5_10.sfal|5-10%|${C}_all_5_10.sfpr" \
    --json results/sec64_caption_$TAG.json
```

The third is the heterogeneous predicate, on LAION 25M. It joins a caption keyword with a window
over the original pixel count, in four groups. `tools/hetero_workload.py` reads the caption files
above and the pixel-count column of Section VI-C:

```bash
D=laion N=25000000 DIM=512 RAW=data/raw-laion25m TAG=laion25m
python3 $REPO/tools/hetero_workload.py --n $N --cap-dir $RAW/caption64 --pixels $RAW/meta/laion_${N}_meta_pixels.sfcl
C=$RAW/caption64/laion_${N}
$NUMA bench_synthfilter --dataset $D --n $N --d $DIM --m 16 --ef-construction 100 --k 100 --nq 992 \
    --modes prefilter,prefilter-scan,prefilter-index,hnsw,acorn-1,acorn-1-3hop,navix-local,racorn-1 \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --three-hop-threshold 1.0 --stride on \
    --seed 12345 --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --ce on --predicate-csr ${C}_caption_tokens.spmat --predicate-column $RAW/meta/laion_${N}_meta_pixels.sfcl \
    --allowlist-spec "${C}_hetero_H1.sfal|H1: kw 5-10% & px 30%|${C}_hetero_H1.sfpr|${C}_hetero_H1.sfrq" \
    --allowlist-spec "${C}_hetero_H2.sfal|H2: kw 5-10% & px 10%|${C}_hetero_H2.sfpr|${C}_hetero_H2.sfrq" \
    --allowlist-spec "${C}_hetero_H3.sfal|H3: kw 1-3% & px 10%|${C}_hetero_H3.sfpr|${C}_hetero_H3.sfrq" \
    --allowlist-spec "${C}_hetero_H4.sfal|H4: kw 1-3% & px 3%|${C}_hetero_H4.sfpr|${C}_hetero_H4.sfrq" \
    --json results/sec64_hetero_$TAG.json
```

**Fig. 6**: the plotted latency of each group on LAION 25M, in three panels. The panels show the
conjunction against `j`, the caption buckets against their selectivity, and the four
keyword-and-pixel-count groups against their selectivity.

## Section VI-E (Fig. 7)

Deadline-based early termination, on all eight datasets. HNSW runs at `ef` 100 and 200. RACORN-1
runs at `ef` 200 with five values of `br`. Each series runs without a deadline and under seven
deadlines. The driver derives the deadlines in the same process, from the median time at which HNSW
fills its result queue when it runs without a deadline. The deadlines are `D100` times 1/4, 1/2 and
1, and `D200` times 1, 2, 4 and 8.

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
$NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
    --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
    --modes hnsw,racorn-1 --selectivity 3%,1%,0.5%,0.1% \
    --ef-values 100,200 --mode-ef racorn-1:200 \
    --et-anchor hnsw@100:0.25,0.5,1 --et-anchor hnsw@200:1,2,4,8 \
    --bridge-budget-ratios 1.0,2.0,4.0,6.0,8.0 --filter-first-threshold 0.3 --stride on \
    --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec65_$TAG.json
```

**Fig. 7**: on SIFT 100M at 1%, `recall` against the deadline (`early_termination_ms` of the rows).
The series are HNSW at `ef` 100 and 200 and RACORN-1 at `br` 2 and 4. The rows without a deadline
give the dashed levels. A series could not hold a deadline that is below its measured `latency_ms`.

## Section VI-F (Fig. 8)

Three ablations. **The bridge budget** (panel b): RACORN-1 at six values of `br`.

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
$NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
    --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
    --modes racorn-1 --selectivity 3%,1%,0.5%,0.1% \
    --ef-values 100,150,200,300,400,600,800,1200,1600 --recall-target 0.97 \
    --bridge-budget-ratios 1.0,2.0,3.0,4.0,6.0,8.0 --filter-first-threshold 0.3 --stride on \
    --runs 3 --warmup 2 --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
    --json results/sec66_$TAG.json
```

Panel (b) uses 64 cells: eight datasets, four selectivities, and recall 0.9 and 0.95. For each cell
and each `br`, it takes the lowest latency at which that `br` reaches the cell's recall. It divides
this latency by the latency of the cell's best `br`. It draws the median and the worst cell.

**Stride sampling** (panel a): ACORN-1 and RACORN-1 with stride sampling on and off, one pass per
cell. The `ef` list runs to its end, with no recall target, and is cut to `ef < N * s`. On the 1M
datasets, five predicate seeds. The 0.1% cell runs in a second process, up to `ef` 800:

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
for seed in 12345 23456 34567 45678 56789; do
  [ $seed = 12345 ] || $NUMA bench_acorn --build-gt --build-gt-k 100 --predicate random --seed $seed \
      --dataset $D --n $N --m 16 --ef-construction 100 --k 100 --nq 992 --selectivity 3%,1%,0.5%,0.1% \
      --search-threads 32 --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt
  for stride in on off; do
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
        --predicate random --seed $seed --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 3%,1%,0.5% \
        --ef-values 100,150,200,300,400,600,800,1200,1600 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
        --json results/sec66s_${TAG}_rnd${seed}_${stride}.json
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
        --predicate random --seed $seed --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 0.1% \
        --ef-values 100,150,200,300,400,600,800 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
        --json results/sec66s_${TAG}_rnd${seed}_${stride}_low.json
  done
done
```

On the datasets of 10M vectors and more, the first three seeds. All four selectivities and the full
`ef` list run in one process:

```bash
D=laion N=25000000 DIM=512 RAW=data/raw-laion25m TAG=laion25m
for seed in 12345 23456 34567; do
  [ $seed = 12345 ] || $NUMA bench_acorn --build-gt --build-gt-k 100 --predicate random --seed $seed \
      --dataset $D --n $N --m 16 --ef-construction 100 --k 100 --nq 992 --selectivity 3%,1%,0.5%,0.1% \
      --search-threads 32 --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt
  for stride in on off; do
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
        --predicate random --seed $seed --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 3%,1%,0.5%,0.1% \
        --ef-values 100,150,200,300,400,600,800,1200,1600 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt \
        --json results/sec66s_${TAG}_rnd${seed}_${stride}.json
  done
done
```

Panel (a) is at 1% and `br` 2. Against `ef`, it draws RACORN-1's recall with stride on minus its
recall with stride off, averaged over the seeds.

**Build variance**: the same measurements at seed 12345, on indexes built independently. The index
is built five times on each dataset, and three times on SIFT 100M. On the datasets of 10M and more,
run the single measurement of the block above instead of the two below:

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
mkdir -p data/volumes-var
for b in 1 2 3 4 5; do
  rm -f data/volumes-var/*.usearch
  bench_acorn --build-index --dataset $D --n $N --m 16 --ef-construction 100 --build-threads 32 \
      --data-dir $RAW --volume-dir data/volumes-var
  for stride in on off; do
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
        --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 3%,1%,0.5% \
        --ef-values 100,150,200,300,400,600,800,1200,1600 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir data/volumes-var --gt-dir data/gt \
        --json results/sec66b_${TAG}_b${b}_${stride}.json
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 \
        --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 0.1% \
        --ef-values 100,150,200,300,400,600,800 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir data/volumes-var --gt-dir data/gt \
        --json results/sec66b_${TAG}_b${b}_${stride}_low.json
  done
done
```

**Cosine controls**: the five-seed stride measurements of SIFT 1M and GIST 1M with `--metric cosine`.
The index and the ground truth go in directories of their own:

```bash
D=sift N=1000000 DIM=128 RAW=data/raw TAG=sift1m-cosine
V=data/volumes-$TAG G=data/gt-$TAG
bench_acorn --build-index --dataset $D --n $N --m 16 --ef-construction 100 --metric cosine \
    --build-threads 32 --data-dir $RAW --volume-dir $V
for seed in 12345 23456 34567 45678 56789; do
  $NUMA bench_acorn --build-gt --build-gt-k 100 --predicate random --seed $seed --metric cosine \
      --dataset $D --n $N --m 16 --ef-construction 100 --k 100 --nq 992 --selectivity 3%,1%,0.5%,0.1% \
      --search-threads 32 --data-dir $RAW --volume-dir $V --gt-dir $G
  for stride in on off; do
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 --metric cosine \
        --predicate random --seed $seed --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 3%,1%,0.5% \
        --ef-values 100,150,200,300,400,600,800,1200,1600 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir $V --gt-dir $G \
        --json results/sec66s_${TAG}_rnd${seed}_${stride}.json
    $NUMA bench_acorn --dataset $D --n $N --m 16 --ef-construction 100 --metric cosine \
        --predicate random --seed $seed --k 100 --nq 992 --skip-exact \
        --modes acorn-1,racorn-1 --selectivity 0.1% \
        --ef-values 100,150,200,300,400,600,800 \
        --bridge-budget-ratios 2.0 --filter-first-threshold 0.3 --stride $stride \
        --runs 1 --warmup 0 --search-threads 32 \
        --data-dir $RAW --volume-dir $V --gt-dir $G \
        --json results/sec66s_${TAG}_rnd${seed}_${stride}_low.json
  done
done
```

**The index degree** (panel c): RACORN-1 at `ef` 100 on the `M = 16` index and on an `M = 32` index.
The run uses eight selectivities. The `M = 32` run reads the ground truth of the `M = 16` index
through a link, under the name the driver looks for.

```bash
D=laion N=1000000 DIM=512 RAW=data/raw-laion1m TAG=laion1m
bench_acorn --build-index --dataset $D --n $N --m 32 --ef-construction 100 --build-threads 32 \
    --data-dir $RAW --volume-dir data/volumes
$NUMA bench_acorn --build-gt --build-gt-k 100 --predicate random --seed 12345 --dataset $D --n $N --m 16 \
    --ef-construction 100 --k 100 --nq 992 --selectivity 10%,5%,4%,3%,2%,1%,0.5%,0.1% --search-threads 32 \
    --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt-m32
for f in data/gt-m32/${D}_${N}_${DIM}_m16_*; do ln -f "$f" "${f/_m16_/_m32_}"; done
for M in 16 32; do
  $NUMA bench_acorn --dataset $D --n $N --m $M --ef-construction 100 \
      --predicate random --seed 12345 --k 100 --nq 992 --skip-exact \
      --modes racorn-1 --selectivity 10%,5%,4%,3%,2%,1%,0.5%,0.1% \
      --ef-values 100 --bridge-budget-ratios 1.0,2.0 --filter-first-threshold 0.3 --stride on \
      --runs 1 --warmup 0 --search-threads 32 \
      --data-dir $RAW --volume-dir data/volumes --gt-dir data/gt-m32 \
      --json results/sec66m_${TAG}_m$M.json
done
```

Panel (c) draws `T/bound` against selectivity. `T/bound` is `expand_cycles` divided by
`(ef / s) * br / mavg`, with `mavg` from the `config` of each file. It shows the median and the
range over the eight datasets.
