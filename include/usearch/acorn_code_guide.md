# ACORN-1 / RACORN-1 / RACORN-1+ — Code Guide

This document maps the algorithm to the implementation to help reviewers
navigate the filtered-search code paths.

---

## 0. Source File Layout

The implementation is split across three headers under `include/usearch/`:

| File | Contents |
|---|---|
| `index.hpp` | `index_gt` class definition, HNSW build/search infrastructure, `search_to_find_in_base_()`, `search_hnsw_in_base_()`, `search_exact_()` |
| `racorn.hpp` | ACORN-1 / RACORN-1 search member functions (class-body fragment, `#include`d inside `index_gt`): `search_acorn1_in_base_`, `acorn_expand_two_hop_`, `exact_fallback`, `racorn_fallback_`, `acorn_3hop_expansion_` |
| `racorn_util.hpp` | Namespace-scope utilities used only by the RACORN path: `gen_hash_set_gt` (O(1) generation-counter hash set), `stride_downsample_inplace` |

`racorn.hpp` is a **class-body fragment** — it has no include guard and is
designed to be `#include`d exactly once, inside the `index_gt` class body
in `index.hpp`. `racorn_util.hpp` is a standard standalone header.

---

## 1. Entry Points

### `search()` (public API, `index_gt`)
Sets per-query context fields from `index_search_config_t`, then calls:
- `search_for_one_()` — greedy descent through upper levels to the best
  starting slot for base-layer search
- `search_to_find_in_base_()` — base-layer dispatcher

### `search_to_find_in_base_()`
Initialises priority queues and dispatches:

```
two_hop_threshold >= 0  AND  predicate != dummy
      └─→  search_acorn1_in_base_()   (ACORN-1 / RACORN-1 / RACORN-1+)

otherwise
      └─→  search_hnsw_in_base_()    (standard HNSW)
```

---

## 2. Function Reference

### `search_hnsw_in_base_()`
Standard HNSW greedy BFS on the base layer.
Evaluates the predicate **after** distance computation.
Predicate-failed nodes are excluded from `top` but pushed into `next`
to preserve graph connectivity.

---

### `search_acorn1_in_base_()`
ACORN-1 main loop. For each candidate popped from `next`:

1. Collect unvisited 1-hop neighbors → `acorn_passed[0..unvisited_1hop)`.
2. Apply predicate → partition passed to front.
3. Compute `ratio = passed_1hop / unvisited_1hop`.
4. **High-density branch** (`ratio > two_hop_threshold`):
   - 2-hop expansion skipped.
   - All unvisited 1-hop pushed into `next`; only predicate-passed into `top`.
5. **Low-density branch** (`ratio ≤ two_hop_threshold`):
   - Delegates to `acorn_expand_two_hop_()`.

Exits early when `context.fallback_to_exact` is set by `racorn_fallback_()`.

**Context fields read per cycle:**

| Field | Role |
|---|---|
| `two_hop_threshold` | gate for 2-hop expansion (TT=0 ≈ HNSW, TT=1 = full ACORN) |
| `racorn` | enables RACORN-1 bridge fallback |
| `acorn_passed_candidates` | scratch buffer for 1-hop + 2-hop candidates |
| `acorn_failed_candidates` | scratch buffer for failed candidates (bridge pool) |
| `acorn_hop_visits` | per-cycle dedup set (O(1) `clear()` via generation counter) |

---

### `acorn_expand_two_hop_()`
Called from the low-density branch of `search_acorn1_in_base_()`.

**Steps:**
1. Seed `acorn_hop_visits` with all unvisited 1-hop slots (prevents
   re-collection as 2-hop candidates within the same cycle).
2. *(RACORN-1 only)* Stash 1-hop predicate-failed into
   `acorn_failed[0..failed_1hop)` before the buffer is overwritten.
3. Collect 2-hop candidates: for each 1-hop neighbor, walk its unvisited,
   unseen neighbors into `acorn_passed[passed_1hop..)`.
4. Apply predicate to 2-hop candidates; compact passed to front of
   `acorn_passed`, failed into `acorn_failed[failed_1hop..)`.
5. Budget cap:
   ```
   limit_1hop = min(passed_1hop, M_base)
   limit_2hop = min(passed_2hop, M_base - limit_1hop)
   ```
   RACORN-1 stride-samples the 2-hop passed pool when it exceeds the
   remaining budget, spreading the selection instead of taking the head.
6. Push `acorn_passed[0..limit_1hop + limit_2hop)` into `next` / `top`.
7. Dispatch:
   - `racorn == true` → `racorn_fallback_()`
   - `three_hop_threshold >= 0` (and `racorn == false`) → `acorn_3hop_expansion_()`

**Buffer layout after this function:**
```
acorn_passed : [0..passed_1hop)           = 1-hop predicate-passed
               [passed_1hop..+passed_2hop) = 2-hop predicate-passed

acorn_failed : [0..failed_1hop)           = 1-hop failed stash  (RACORN-1)
               [failed_1hop..+failed_2hop) = 2-hop failed / bridge pool
```

---

### `racorn_fallback_()`
RACORN-1 bridge insertion. Fires when:

```
passed_2hop < bridge_budget   AND   failed_2hop > 0
where  bridge_budget = one_hop_count × bridge_budget_ratio
```

**Steps:**
1. Bulk-mark all **1-hop + 2-hop** predicate-failed slots as visited
   (prevents re-evaluation in subsequent cycles; see `acorn_failed` layout).
2. If `top` is already full → skip bridge insertion, return early
   (counters not advanced for this cycle).
3. Advance `fallback_count / fallback_evaluated / fallback_passed`.
4. *(RACORN-1+)* Call `exact_fallback()`. If it triggers, set
   `context.fallback_to_exact = true` and return. The caller exits its
   loop and `search()` dispatches `search_exact_()`.
5. Stride-sample the bridge pool (`acorn_failed[failed_1hop..)`) and
   push selected bridges into `next`.

---

### `exact_fallback()`
RACORN-1+ heuristic. Returns `true` (trigger exact search) when
**all** conditions hold:

- `fallback_exact_threshold > 0` — feature enabled
- `fb_count > top_limit / 2` — warm-up window passed
- `fb_pass / fb_eval ≤ fallback_exact_threshold` — bridge selectivity low

Per-query snapshots (`fallback_*_before`) are stored at `search()` entry
so only the **current query's** delta is measured, not the cumulative
thread-local total.

---

### `acorn_3hop_expansion_()` *(USEARCH_ENABLE_3HOP=1 only)*
3-hop expansion from 2-hop predicate-failed hubs. Fires when:

```
passed_2hop < connectivity_base × three_hop_threshold
AND  failed_2hop > 0
AND  top is not full
```

Reuses `acorn_hop_visits` without clearing — 3-hop candidates that
coincide with 1-hop / 2-hop slots already in the set are automatically
skipped. The set was seeded in `acorn_expand_two_hop_()`.

Mutually exclusive with RACORN-1 (`else-if` branch).

---

## 3. Call Graph

```
search()
 └─ search_to_find_in_base_()
      ├─ [two_hop_threshold < 0 | no predicate]
      │    └─ search_hnsw_in_base_()
      │
      └─ [two_hop_threshold >= 0 AND predicate]
           └─ search_acorn1_in_base_()
                ├─ [ratio > TT]  high-density → push 1-hop to next+top
                └─ [ratio ≤ TT]  low-density
                      └─ acorn_expand_two_hop_()
                            ├─ [racorn == true]
                            │    └─ racorn_fallback_()
                            │          └─ [RACORN-1+] exact_fallback()
                            │                └─ [triggered]
                            │                      context.fallback_to_exact = true
                            │                      → search() calls search_exact_()
                            └─ [three_hop_threshold >= 0, racorn == false]
                                 └─ acorn_3hop_expansion_()
```

---

## 4. Key Data Structures

### `index_search_config_t` — per-query activation flags

| Field | Default | Effect |
|---|:-:|---|
| `two_hop_threshold` | −1 | ≥ 0 activates ACORN-1; =0 ≈ HNSW, =1 = full ACORN |
| `racorn` | false | Activates RACORN-1 bridge fallback |
| `bridge_budget_ratio` | 1.0 | Bridge budget scale: `one_hop_count × ratio` |
| `three_hop_threshold` | −1 | ≥ 0 activates 3-hop (mutually exclusive with racorn) |
| `fallback_exact_threshold` | 0.0 | > 0 activates RACORN-1+ exact fallback |
| `racorn_stride_sampling` | true | Stride-sample passed/failed pools; set false for ablation |

### `context_t` — per-thread scratch (ACORN/RACORN fields)

| Field | Description |
|---|---|
| `acorn_passed_candidates` | 1-hop + 2-hop predicate-passed candidates |
| `acorn_failed_candidates` | 1-hop stash + 2-hop failed / bridge pool |
| `acorn_hop_visits` | Per-cycle dedup hash set (`gen_hash_set_gt`, defined in `racorn_util.hpp`); O(1) clear via generation counter |
| `fallback_count` | Cycles that fired bridge fallback (cumulative, this thread) |
| `fallback_evaluated` | Denominator for observed bridge selectivity |
| `fallback_passed` | Numerator for observed bridge selectivity |
| `fallback_to_exact` | Set by `racorn_fallback_` to trigger `search_exact_` |
| `predicate_evaluations` | Total predicate calls across all paths (this thread) |

### `gen_hash_set_gt` — per-cycle dedup set (`racorn_util.hpp`)

Fixed-capacity hash set with O(1) clear via a generation counter.
On each `clear()` only a counter is bumped; slots whose stored generation
does not match the current one are treated as empty. Used as
`cycle_visits_hash_set_t` (`acorn_hop_visits`) in `context_t` to dedup
1-hop / 2-hop (/ 3-hop) candidates within a single ACORN expansion cycle.

### Scratch buffer sizing (`try_reserve`)

```
acorn_buf_size  = M_base × (M_base + 1)          # ~M²
acorn_pool_size = acorn_buf_size                   # default (2-hop)
               = M_base × acorn_buf_size  [3-hop]  # ~M³
```

Both `acorn_passed_candidates` and `acorn_failed_candidates` are
sized to `acorn_pool_size` so no reallocation occurs mid-search.

---

## 5. Mode Activation Summary

| Mode | `two_hop_threshold` | `racorn` | `three_hop_threshold` | `fallback_exact_threshold` |
|---|:-:|:-:|:-:|:-:|
| Plain HNSW | < 0 | — | — | — |
| ACORN-1 | ≥ 0 | false | < 0 | — |
| ACORN-1 + 3-hop | ≥ 0 | false | ≥ 0 | — |
| RACORN-1 | ≥ 0 | true | < 0 | 0 |
| RACORN-1+ | ≥ 0 | true | < 0 | > 0 |
