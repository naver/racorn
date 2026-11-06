# Search options

`index_search_config_t` (`index.hpp`) holds the per-query options of `index_gt::search`. `W` is the
result heap of size `ef`. `C` is the candidate queue. `M0` is the base-layer degree bound of the index.
The library example of the root README sets `expansion`, `filter_first_threshold` and
`bridge_budget_ratio`. `bench/README.md` lists what the drivers set for each method.

| Field | Default | Effect |
|---|:-:|---|
| `expansion` | `64` | `ef`, the capacity of `W`. Also the minimum size of `C` |
| `filter_first_threshold` | `-1.0` | `< 0`: HNSW. `>= 1`: ACORN-1. `[0, 1)`: probe the entry node's neighborhood and run ACORN-1 when the share that passes is at or below the value. The drivers' default is `0.3` |
| `bridge_budget_ratio` | `0.0` | the bridge budget `br`. `0` is ACORN-1. `> 0` enables Adaptive Search Fallback on the ACORN-1 path, with the target `⌊n·br⌋` per expansion, where `n` is the number of unvisited one-hop neighbors of the expanded node |
| `stride_sampling` | `true` | stride-sample two selections: the two-hop passing cap and the bridge pool. `false` is prefix truncation, the ablation baseline |
| `navix_heuristic` | `0` | `8` is NaviX's adaptive-local, as the NaviX authors' Kuzu code runs it. A non-zero value bypasses the filter-first dispatch |
| `three_hop_threshold` | `0.0` | `> 0` enables the third-hop expansion when a cycle's two-hop expansion returns fewer than `M0 × value` passing two-hop nodes |
| `early_termination_time_threshold` | negative (off) | a soft deadline on the base-layer traversal (DET). Once the deadline passes, the search stops at the next check, provided `W` is full. The check runs every 10 cycles, and the cycle in flight always finishes |
