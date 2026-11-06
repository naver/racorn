// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 *  @file   racorn.hpp
 *  @brief  ACORN-1 and RACORN-1 search member functions of `index_gt` — a
 *          class-body fragment.
 *
 *  index.hpp includes this file once, inside the `index_gt` class body. So
 *  everything here is a private member of `index_gt`. It sees the types of
 *  `index_gt` (`context_t`, `candidate_t`, `acorn_candidate_t`,
 *  `acorn_bridge_state_t`, `neighbors_ref_t`, ...) and its helpers (`node_at_`,
 *  `neighbors_base_`, `citerator_at`, `sort_neighbors_into_buffer_`). The file
 *  has no include guard and must not be included directly: include
 *  <usearch/index.hpp>.
 *
 *  What stays in index.hpp: the per-query knobs (`index_search_config_t::
 *  filter_first_threshold`, `bridge_budget_ratio`, `stride_sampling`,
 *  `three_hop_threshold`), the per-thread scratch and counters in `context_t`,
 *  the per-query deltas in `search_result_t`, the buffer sizing in
 *  `try_reserve`, and the dispatch in `search_to_find_in_base_`. The dispatch
 *  calls `entry_filter_pass_ratio_` and `search_acorn1_in_base_` below.
 *
 *  References:
 *    - ACORN-1: Patel et al., "ACORN: Performant and Predicate-Agnostic Search
 *      over Vector Embeddings and Structured Data", SIGMOD 2024.
 *    - RACORN-1: "RACORN: Robust Predicate-Agnostic Filtered Vector Search
 *      under Extreme Selectivity" (this artifact's paper).
 *
 *  Functions, in call order:
 *    - entry_filter_pass_ratio_       filter-first probe: pass ratio of the
 *                                     entry's 1-hop + 2-hop neighborhood,
 *                                     compared against filter_first_threshold
 *    - search_acorn1_in_base_         ACORN-1 / RACORN-1 base-layer loop
 *    - acorn_expand_two_hop_          two-hop expansion, sparse 1-hop branch
 *                                     (<= 1 passing 1-hop neighbor)
 *    - acorn_fill_m_passed_           two-hop expansion, dense 1-hop branch
 *                                     (>= 2 passing), ordered walk to M0
 *    - distance_compute_and_promote_  shared tail of both branches
 *    - acorn_expand_third_hop_        third-hop expansion (expansion-depth
 *                                     ablation, three_hop_threshold > 0)
 *    - racorn_fallback_               Adaptive Search Fallback: bridge
 *                                     insertion (bridge_budget_ratio > 0)
 *
 *  Namespace-scope utilities (`gen_hash_set_gt`, `stride_downsample_inplace`)
 *  are in racorn_util.hpp. The NaviX reimplementation is in navix.hpp.
 */

#ifndef UNUM_USEARCH_HPP
#error "racorn.hpp is a class-body fragment of index.hpp; include <usearch/index.hpp> instead"
#endif

    /**
     *  @brief  Measures the filter pass ratio over the base-level entry node's
     *          own slot + 1-hop + 2-hop neighborhood. This is the "first filter"
     *          probe that drives the ACORN-1 vs HNSW dispatch.
     *
     *  Collects the distinct base-level neighbors of `start_slot`, up to the
     *  worst-case 2-hop pool size. Primes caches via `preevaluate`; the batch is
     *  sorted by `neighbor_cmp` first, to honor its ascending-key contract. Then
     *  counts predicate passes. The entry node itself is part of the selectivity
     *  sample. Its predicate verdict (`start_passes`) is supplied by the caller,
     *  which already evaluated it for the top-K seed. So the entry counts toward
     *  the ratio without a re-evaluation.
     *
     *  Pure measurement: does NOT touch `visits` / `next` / `top`. Predicate
     *  evals route through `context.evaluate`, so they count toward the
     *  per-query `predicate_evaluations` counter. Reuses
     *  `context.acorn_hop_visits` (cleared here) for dedup and
     *  `context.acorn_passed_candidates` as the sample buffer.
     *
     *  @param[in] start_passes  the entry node's predicate verdict (from caller).
     *  @return Pass ratio in [0, 1] over {entry} ∪ 1-hop ∪ 2-hop. The denominator
     *          always includes at least the entry.
     */
    template <typename predicate_at, typename preevaluate_at, typename neighbor_cmp_at>
    double entry_filter_pass_ratio_(predicate_at&& predicate, preevaluate_at&& preevaluate,
                                    neighbor_cmp_at&& neighbor_cmp, compressed_slot_t start_slot, bool start_passes,
                                    context_t& context) const noexcept {
        neighbors_ref_t one_hop = neighbors_base_(node_at_(start_slot));
        cycle_visits_hash_set_t& seen = context.acorn_hop_visits;
        seen.clear();
        seen.set(start_slot); // entry counted via start_passes; keep it out of the neighbor pool
        // The sample buffer (acorn_pool_size = connectivity_base * (connectivity_base + 1))
        // is sized to the worst-case distinct 1-hop + 2-hop pool, so no bound
        // check is needed. acorn_expand_two_hop_ relies on the same invariant.
        buffer_gt<acorn_candidate_t>& probe = context.acorn_passed_candidates;
        std::size_t probe_count = 0;
        // A node's neighbor list has no duplicates and never contains the node
        // itself, so every 1-hop slot is distinct. Collect them unconditionally.
        // Marking `seen` only excludes them from the 2-hop pass below.
        for (compressed_slot_t hop1 : one_hop) {
            seen.set(hop1);
            probe[probe_count++] = {hop1};
        }
        for (compressed_slot_t hop1 : one_hop) {
            neighbors_ref_t two_hop = neighbors_base_(node_at_(hop1));
            for (compressed_slot_t hop2 : two_hop) {
                if (seen.set(hop2))
                    continue;
                probe[probe_count++] = {hop2};
            }
        }
        // The preevaluate contract requires ascending-key batches under
        // neighbor_cmp. The probe concatenates 1-hop and 2-hop slots, which
        // are not jointly sorted, so sort here. The entry slot is not in the
        // probe; its verdict is reused via start_passes below.
        if (!is_dummy<neighbor_cmp_at>() && probe_count > 1) {
            std::sort(probe.data(), probe.data() + probe_count,
                      [&](acorn_candidate_t const& a, acorn_candidate_t const& b) {
                          return neighbor_cmp(member_cref_t{node_at_(a.slot).ckey(), a.slot},
                                              member_cref_t{node_at_(b.slot).ckey(), b.slot});
                      });
        }
        if (!is_dummy<preevaluate_at>() && probe_count > 0) {
            acorn_candidates_iterator_t qbegin(*this, probe.data());
            acorn_candidates_iterator_t qend(*this, probe.data() + probe_count);
            preevaluate(qbegin, qend);
        }
        // Entry node is part of the sample: seed numerator/denominator with its
        // reused verdict, then add the 1-hop + 2-hop neighbor passes.
        std::size_t passes = start_passes ? 1u : 0u;
        std::size_t total = probe_count + 1; // +1 for the entry itself
        for (std::size_t i = 0; i < probe_count; ++i) {
            if (context.evaluate(predicate, member_cref_t{node_at_(probe[i].slot).ckey(), probe[i].slot}))
                passes++;
        }
        return static_cast<double>(passes) / static_cast<double>(total);
    }

    /**
     *  @brief  ACORN-1 filtered search loop for the base layer. Every cycle
     *          with an unvisited 1-hop neighbor runs a two-hop expansion. The
     *          number of passing 1-hop neighbors (<= 1 or >= 2) selects the
     *          branch.
     *  @return `true` if the procedure succeeded, `false` if it ran out of memory.
     */
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at,
              typename preevaluate_at, typename neighbor_cmp_at>
    bool search_acorn1_in_base_(                                                                //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        preevaluate_at&& preevaluate, neighbor_cmp_at&& neighbor_cmp,                           //
        distance_t& radius, std::size_t top_limit, context_t& context,
        early_termination_gate_t& early_termination) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates;
        top_candidates_t& top = context.top_candidates;

        // Per-thread scratch buffers reused across cycles for 1-hop / 2-hop candidates.
        // Sized once at try_reserve time.
        buffer_gt<acorn_candidate_t>& acorn_passed = context.acorn_passed_candidates;

        enum { stop_queue = 0, stop_rule = 1, stop_deadline = 2 } stop = stop_queue;
        std::size_t const dist0 = context.computed_distances, pred0 = context.predicate_evaluations;
        bool fill_marked = false;
        while (!next.empty()) {
            candidate_t candidate = next.top();
            if ((-candidate.distance) > radius && top.size() == top_limit) {
                stop = stop_rule;
                break;
            }

            if (early_termination.should_terminate(context.iteration_cycles, top.size(), top_limit)) {
                stop = stop_deadline;
                break;
            }

            next.pop();
            context.iteration_cycles++;
            context.expansion_cycles++;
            // Phase of this cycle: W still filling, or already at `top_limit`.
            bool const prefill = top.size() < top_limit;
            if (prefill) {
                context.cycles_prefill++;
            } else if (!fill_marked) {
                context.dist_prefill += context.computed_distances - dist0;
                context.pred_prefill += context.predicate_evaluations - pred0;
                // One clock read per query at most: this branch runs once.
                context.prefill_ms = early_termination.elapsed_ms();
                snapshot_fill_(context, top);
                fill_marked = true;
            }

            neighbors_ref_t candidate_neighbors = neighbors_base_(node_at_(candidate.slot));
            std::size_t const neighbor_count = candidate_neighbors.size();

            compressed_slot_t* sorted_data =
                sort_neighbors_into_buffer_(candidate_neighbors, neighbor_count, context, neighbor_cmp);

            auto get_neighbor_slot = [&](std::size_t ni) -> compressed_slot_t {
                return !is_dummy<neighbor_cmp_at>() ? sorted_data[ni] : candidate_neighbors[ni];
            };

            // ---- Single-pass 1-hop scan. Unvisited slots go to the front of
            // acorn_passed, the source of the predicate partition. Visited slots
            // go to the acorn_visited_candidates buffer (slot only), which
            // acorn_fill_m_passed_ consumes as its Phase 3 source. Visited slots
            // are predicate-evaluated inline only to feed the full-1hop
            // filter-pass count (passed_total_1hop) used by branch dispatch. Their
            // verdict is not stored; it is resolved via the caller's cache.
            buffer_gt<acorn_candidate_t>& visited_stash = context.acorn_visited_candidates;
            std::size_t unvisited_1hop = 0;
            std::size_t visited_1hop_count = 0;
            std::size_t passed_visited_1hop = 0;
            for (std::size_t ni = 0; ni < neighbor_count; ++ni) {
                compressed_slot_t slot = get_neighbor_slot(ni);
                if (visits.test(slot)) {
                    if (context.evaluate(predicate, member_cref_t{node_at_(slot).ckey(), slot}))
                        passed_visited_1hop++;
                    visited_stash[visited_1hop_count++].slot = slot;
                } else {
                    acorn_passed[unvisited_1hop++] = {slot};
                }
            }

            if (unvisited_1hop == 0) {
                context.expansion_empty_cycles++;
                continue;
            }

            if (!is_dummy<preevaluate_at>()) {
                acorn_candidates_iterator_t qbegin(*this, acorn_passed.data());
                acorn_candidates_iterator_t qend(*this, acorn_passed.data() + unvisited_1hop);
                preevaluate(qbegin, qend);
            }

            // Partition: predicate-passed slots are compacted to the front of
            // acorn_passed, with the sorted order preserved. Predicate-failed slots
            // are written straight into acorn_failed, the dense Phase-1 expansion
            // source. Every unvisited 1-hop slot is also seeded into the per-cycle
            // 2-hop dedup set here, so both 2-hop branches share a single
            // materialization pass. Predicate-failing 1-hop and 2-hop slots
            // accumulate in acorn_failed (the bridge pool). Neither branch marks
            // them visited; only racorn_fallback_ does, on the cycles it fires. On
            // every other cycle they stay unvisited and may resurface through
            // another candidate's neighborhood.
            buffer_gt<acorn_candidate_t>& acorn_failed = context.acorn_failed_candidates;
            cycle_visits_hash_set_t& hop2_seen = context.acorn_hop_visits;
            hop2_seen.clear();
            std::size_t passed_1hop = 0;
            std::size_t failed_1hop = 0;
            for (std::size_t i = 0; i < unvisited_1hop; ++i) {
                acorn_candidate_t c = acorn_passed[i];
                hop2_seen.set(c.slot);
                if (context.evaluate(predicate, member_cref_t{node_at_(c.slot).ckey(), c.slot}))
                    acorn_passed[passed_1hop++] = c;
                else
                    acorn_failed[failed_1hop++] = c;
            }

            // 2-hop expansion: the dispatch uses the full-1hop filter-pass count
            // (unvisited + visited). <=1 pass -> acorn_expand_two_hop_ (full
            // 2-hop pool). >=2 -> acorn_fill_m_passed_ (failed -> passed ->
            // visited 1-hop). The latter receives visited_1hop_count, so its Phase 3
            // can expand visited 1-hop slots without re-walking candidate_neighbors.
            std::size_t passed_total_1hop = passed_1hop + passed_visited_1hop;

            acorn_bridge_state_t bridge{};
            if (passed_total_1hop <= 1) {
                // ---- Sparse 1-hop (<=1 pass): collect all 2-hop at once, stride-sample.
                if (!acorn_expand_two_hop_(query, metric, predicate, prefetch, preevaluate, neighbor_cmp, //
                                           context, candidate_neighbors, sorted_data, neighbor_count, passed_1hop,
                                           unvisited_1hop, radius, top_limit, bridge))
                    return false; // OOM
            } else {
                // ---- Dense 1-hop (>=2 pass): fill M0 passing nodes in priority order
                //      failed -> passed -> visited, distance-compute the M0.
                if (!acorn_fill_m_passed_(query, metric, predicate, prefetch, preevaluate, neighbor_cmp, //
                                          context, passed_1hop, unvisited_1hop, visited_1hop_count,      //
                                          radius, top_limit, bridge))
                    return false; // OOM
            }

            // What this EXPAND returned: 1-hop passing + 2-hop passing, bridges
            // excluded. Counted here, before the third-hop and bridge paths add
            // to the same cycle.
            context.expansion_size_sum += bridge.total_passed_limit;
            if (prefill)
                context.expand_prefill += bridge.total_passed_limit;
            if (!bridge.total_passed_limit)
                context.expansion_empty_cycles++;

            // ---- ACORN-1 third-hop expansion (depth-axis ablation) ----
            // No-op unless three_hop_threshold > 0.
            if (context.three_hop_threshold > 0.0) {
                if (!acorn_expand_third_hop_(query, metric, predicate, context, acorn_failed, bridge, radius,
                                             top_limit))
                    return false; // OOM
            }

            // ---- RACORN-1 bridge fallback ----
            // Fills the passing shortfall with stride-sampled 2-hop failed slots.
            // Runs only when bridge_budget_ratio > 0.0 (no-op for plain ACORN-1).
            if (context.bridge_budget_ratio > 0.0) {
                std::size_t const bridges_before = context.bridge_size_sum;
                std::size_t const fires_before = context.fallback_count;
                if (!racorn_fallback_(query, metric, prefetch, neighbor_cmp, //
                                      context, next, acorn_failed,           //
                                      bridge, top_limit, top))
                    return false; // OOM
                if (prefill) {
                    context.bridge_prefill += context.bridge_size_sum - bridges_before;
                    context.fallback_prefill += context.fallback_count - fires_before;
                }
            }
        }
        if (!fill_marked) {
            context.dist_prefill += context.computed_distances - dist0;
            context.pred_prefill += context.predicate_evaluations - pred0;
        }
        context.dist_base += context.computed_distances - dist0;
        context.pred_base += context.predicate_evaluations - pred0;
        if (stop == stop_rule)
            context.stop_radius++;
        else if (stop == stop_deadline)
            context.stop_budget++;
        else
            context.stop_queue_empty++;
        if (top.size() < top_limit)
            context.never_full++;
        return true;
    }

    /**
     *  @brief  Low-density 2-hop expansion branch of `search_acorn1_in_base_`.
     *          It collects the 2-hop neighbors, partitions them by the predicate,
     *          and inserts the passed list into next/top.
     *
     *  @param[in]  candidate_neighbors  the candidate node's 1-hop neighbors (raw order)
     *  @param[in]  sorted_data          optional sorted neighbor list (nullptr -> use raw order)
     *  @param[in]  neighbor_count       size of `candidate_neighbors`
     *  @param[in]  passed_1hop          number of 1-hop predicate-passed slots, sitting at
     *                                   acorn_passed[0..passed_1hop)
     *  @param[in]  unvisited_1hop       total unvisited 1-hop slots (passed + failed) at
     *                                   acorn_passed[0..unvisited_1hop)
     *  @param[in,out] radius            search radius (top-K worst distance), updated when
     *                                   `top` admits a closer candidate
     *  @param[in]  top_limit            top-K capacity (= ef_search)
     *  @return `true` on success, `false` on OOM.
     */
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at,
              typename preevaluate_at, typename neighbor_cmp_at>
    bool acorn_expand_two_hop_(                                                                 //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        preevaluate_at&& preevaluate, neighbor_cmp_at&& neighbor_cmp,                           //
        context_t& context,                                                                     //
        neighbors_ref_t candidate_neighbors,                                                    //
        compressed_slot_t const* sorted_data,                                                   //
        std::size_t neighbor_count,                                                             //
        std::size_t passed_1hop, std::size_t unvisited_1hop,                                    //
        distance_t& radius, std::size_t top_limit, acorn_bridge_state_t& bridge) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        std::size_t const connectivity_base = config_.connectivity_base;
        buffer_gt<acorn_candidate_t>& acorn_passed = context.acorn_passed_candidates;
        buffer_gt<acorn_candidate_t>& acorn_failed = context.acorn_failed_candidates;

        auto get_neighbor_slot = [&](std::size_t ni) -> compressed_slot_t {
            return sorted_data ? sorted_data[ni] : candidate_neighbors[ni];
        };

        // The 1-hop failed slots are already in acorn_failed[0..failed_1hop_stash),
        // and the per-cycle 2-hop dedup set is already seeded. The caller's
        // partition did both. The bridge pool's 2-hop section starts at
        // failed_1hop_stash.
        std::size_t const failed_1hop_stash = unvisited_1hop - passed_1hop;
        cycle_visits_hash_set_t& hop2_seen = context.acorn_hop_visits;

        // Collect the 2-hop neighbors into the buffer, after the 1-hop passed slots.
        std::size_t two_hop_count = 0;
        for (std::size_t ni = 0; ni < neighbor_count; ++ni) {
            compressed_slot_t hop1_slot = get_neighbor_slot(ni);
            neighbors_ref_t hop2_neighbors = neighbors_base_(node_at_(hop1_slot));
            std::size_t const hop2_count = hop2_neighbors.size();
            for (std::size_t hi = 0; hi < hop2_count; ++hi) {
                compressed_slot_t hop2_slot = hop2_neighbors[hi];
                if (visits.test(hop2_slot))
                    continue;
                if (hop2_seen.set(hop2_slot))
                    continue;
                acorn_passed[passed_1hop + two_hop_count] = {hop2_slot};
                two_hop_count++;
            }
        }

        // Nothing to process this cycle.
        if (two_hop_count == 0 && passed_1hop == 0)
            return true;

        // Sort 2-hop candidates by neighbor_cmp so preevaluate / prefetch operate in priority order.
        if (!is_dummy<neighbor_cmp_at>() && two_hop_count > 1) {
            std::sort(acorn_passed.data() + passed_1hop, acorn_passed.data() + passed_1hop + two_hop_count,
                      [&](acorn_candidate_t const& a, acorn_candidate_t const& b) {
                          return neighbor_cmp(member_cref_t{node_at_(a.slot).ckey(), a.slot},
                                              member_cref_t{node_at_(b.slot).ckey(), b.slot});
                      });
        }

        // preevaluate the 2-hop portion (the caller preevaluated the 1-hop slots before its partition)
        if (!is_dummy<preevaluate_at>() && two_hop_count > 0) {
            acorn_candidates_iterator_t qbegin(*this, acorn_passed.data() + passed_1hop);
            acorn_candidates_iterator_t qend(*this, acorn_passed.data() + passed_1hop + two_hop_count);
            preevaluate(qbegin, qend);
        }

        // Predicate on the 2-hop slots: passed ones are compacted after the 1-hop
        // slots in acorn_passed. Failed slots are appended at offset
        // failed_1hop_stash to form the bridge pool (acorn_failed layout: see
        // acorn_bridge_state_t).
        std::size_t passed_2hop = 0;
        std::size_t failed_2hop = 0;
        for (std::size_t i = 0; i < two_hop_count; ++i) {
            acorn_candidate_t c = acorn_passed[passed_1hop + i];
            if (context.evaluate(predicate, member_cref_t{node_at_(c.slot).ckey(), c.slot})) {
                acorn_passed[passed_1hop + passed_2hop] = c;
                passed_2hop++;
            } else {
                acorn_failed[failed_1hop_stash + failed_2hop++] = c;
            }
        }

        // 1-hop passed + 2-hop passed share the budget connectivity_base (M0).
        // Stride-sample the 2-hop portion when it would otherwise exceed the
        // remaining budget. This spreads the selection across the passed pool
        // instead of taking only the head. With stride_sampling off the call is
        // skipped, and the head `limit_2hop` entries stay in place. That is
        // prefix truncation, the ablation baseline.
        std::size_t limit_1hop = (std::min)(passed_1hop, connectivity_base);
        std::size_t remaining = connectivity_base - limit_1hop;
        std::size_t limit_2hop = (std::min)(passed_2hop, remaining);

        if (context.stride_sampling)
            stride_downsample_inplace(acorn_passed.data() + passed_1hop, passed_2hop, limit_2hop);

        std::size_t total_passed_limit = limit_1hop + limit_2hop;
        if (!visits.reserve(visits.size() + total_passed_limit))
            return false;

        if (!is_dummy<prefetch_at>() && total_passed_limit > 0) {
            // Merge the two sorted halves in place: the 1-hop passed prefix and the
            // 2-hop block, each already sorted. Prefetch then sees neighbor_cmp
            // order, the HNSW contract. distance_compute_and_promote_ below does
            // not depend on the order of its input, so reordering acorn_passed is safe.
            if (!is_dummy<neighbor_cmp_at>() && passed_1hop > 0 && limit_1hop < total_passed_limit) {
                std::inplace_merge(acorn_passed.data(), acorn_passed.data() + passed_1hop,
                                   acorn_passed.data() + total_passed_limit,
                                   [&](acorn_candidate_t const& a, acorn_candidate_t const& b) {
                                       return neighbor_cmp(member_cref_t{node_at_(a.slot).ckey(), a.slot},
                                                           member_cref_t{node_at_(b.slot).ckey(), b.slot});
                                   });
            }
            acorn_candidates_iterator_t pbegin(*this, acorn_passed.data());
            acorn_candidates_iterator_t pend(*this, acorn_passed.data() + total_passed_limit);
            prefetch(pbegin, pend);
        }

        distance_compute_and_promote_(query, metric, context, total_passed_limit, radius, top_limit);

        // Hand the failed pool + counts to the caller's fallback
        // (acorn_failed layout: see acorn_bridge_state_t). In the sparse branch
        // the 2-hop failed slots are extracted in order, so no re-sort is needed
        // (needs_failed_sort=false).
        bridge.failed_1hop = failed_1hop_stash;
        bridge.failed_2hop = failed_2hop;
        bridge.passed_2hop = passed_2hop;
        bridge.one_hop_count = unvisited_1hop;
        bridge.total_passed_limit = total_passed_limit;
        bridge.needs_failed_sort = false;
        return true;
    }

    /**
     *  @brief  Dense-1hop branch of `search_acorn1_in_base_` (>=2 filter-passing
     *          1-hop neighbors). It walks each 1-hop neighbor's 2-hop list in
     *          turn and appends filter-passing 2-hop slots until M0 passing
     *          nodes are collected in total. M0 = connectivity_base, the code's
     *          local `M`; the 1-hop passed prefix counts toward M0. Then it
     *          distance-computes the M0 nodes and inserts them into next/top
     *          when the radius condition holds.
     *
     *  Unlike `acorn_expand_two_hop_`, this path does NOT collect the full 2-hop
     *  pool and does NOT stride-sample it. It stops as soon as M0 passing nodes
     *  are found. The 1-hop failed slots are the Phase-1 expansion source; the
     *  caller's partition wrote them into acorn_failed. The filter-failed 2-hop
     *  slots it evaluates are appended to the bridge pool
     *  (acorn_failed[failed_1hop_stash..)).
     *
     *  @param[in] passed_1hop     unvisited 1-hop predicate-passed slots, sitting
     *                             at acorn_passed[0..passed_1hop) (sorted prefix).
     *  @param[in] unvisited_1hop  total unvisited 1-hop (passed + failed).
     */
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at,
              typename preevaluate_at, typename neighbor_cmp_at>
    bool acorn_fill_m_passed_(                                                                  //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        preevaluate_at&& preevaluate, neighbor_cmp_at&& neighbor_cmp,                           //
        context_t& context,                                                                     //
        std::size_t passed_1hop, std::size_t unvisited_1hop, std::size_t visited_1hop_count,    //
        distance_t& radius, std::size_t top_limit, acorn_bridge_state_t& bridge) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        std::size_t const M = config_.connectivity_base;
        buffer_gt<acorn_candidate_t>& acorn_passed = context.acorn_passed_candidates;
        buffer_gt<acorn_candidate_t>& acorn_failed = context.acorn_failed_candidates;
        // Per-1hop 2-hop batch scratch (sized connectivity_base == max 2-hop
        // fan-out). The prefetch path sorts acorn_passed in place.
        buffer_gt<acorn_candidate_t>& batch = context.acorn_batch_candidates;

        // acorn_failed layout: see acorn_bridge_state_t. Consumed by the
        // fallback after this branch.
        std::size_t failed_2hop = 0;

        // Target M0 passing nodes: 1-hop passed prefix (capped at M0) + 2-hop passed.
        std::size_t const limit_1hop = (std::min)(passed_1hop, M);
        std::size_t collected = limit_1hop; // acorn_passed[0..limit_1hop) already = 1-hop passed

        // The 1-hop failed slots are already in acorn_failed[0..failed_1hop_count),
        // the dense Phase-1 source, and the per-cycle 2-hop dedup set is already
        // seeded. The caller's partition did both.
        std::size_t const failed_1hop_count = unvisited_1hop - passed_1hop;
        std::size_t const failed_1hop_stash = failed_1hop_count;
        cycle_visits_hash_set_t& hop2_seen = context.acorn_hop_visits;

        // Phase 3 source: visited 1-hop slots, stashed by the caller's scan in
        // the acorn_visited_candidates buffer (slot only).
        buffer_gt<acorn_candidate_t>& visited_stash = context.acorn_visited_candidates;

        // Per-1hop 2-hop batch expansion: gather that 1-hop neighbor's unvisited,
        // unseen 2-hop candidates; preevaluate them (cache prime); evaluate the
        // predicate; append passing slots until M0, and failing slots to the
        // bridge pool. Used by the three phases below.
        auto expand_2hop_from = [&](compressed_slot_t hop1_slot) {
            if (collected >= M)
                return;
            neighbors_ref_t hop2_neighbors = neighbors_base_(node_at_(hop1_slot));
            std::size_t const hop2_count = hop2_neighbors.size();

            std::size_t batch_n = 0;
            for (std::size_t hi = 0; hi < hop2_count; ++hi) {
                compressed_slot_t hop2_slot = hop2_neighbors[hi];
                if (visits.test(hop2_slot))
                    continue;
                if (hop2_seen.set(hop2_slot))
                    continue;
                batch[batch_n++] = {hop2_slot};
            }
            if (batch_n == 0)
                return;

            // Sort the batch by neighbor_cmp before preevaluate. The preevaluate
            // contract requires ascending-key batches.
            if (!is_dummy<neighbor_cmp_at>() && batch_n > 1) {
                std::sort(batch.data(), batch.data() + batch_n,
                          [&](acorn_candidate_t const& a, acorn_candidate_t const& b) {
                              return neighbor_cmp(member_cref_t{node_at_(a.slot).ckey(), a.slot},
                                                  member_cref_t{node_at_(b.slot).ckey(), b.slot});
                          });
            }

            if (!is_dummy<preevaluate_at>()) {
                acorn_candidates_iterator_t qbegin(*this, batch.data());
                acorn_candidates_iterator_t qend(*this, batch.data() + batch_n);
                preevaluate(qbegin, qend);
            }

            for (std::size_t i = 0; i < batch_n && collected < M; ++i) {
                if (!context.evaluate(predicate, member_cref_t{node_at_(batch[i].slot).ckey(), batch[i].slot})) {
                    acorn_failed[failed_1hop_stash + failed_2hop++] = batch[i];
                    continue;
                }
                acorn_passed[collected++] = batch[i];
            }
        };

        // Phase 1: 2-hop expansion of UNVISITED FILTER-FAILED 1-hop neighbors.
        // These are the "bypass" stepping stones ACORN's 2-hop expansion was
        // designed for. Their 2-hop neighbors are the ones most likely to
        // surface new filter-passing candidates that the 1-hop layer hides.
        for (std::size_t i = 0; i < failed_1hop_count && collected < M; ++i)
            expand_2hop_from(acorn_failed[i].slot);

        // Phase 2: 2-hop expansion of UNVISITED FILTER-PASSED 1-hop neighbors.
        // It runs only if Phase 1 did not fill M0. Each of these nodes is in the
        // `next` queue, so their 2-hop neighbors will also be reached by natural
        // traversal in later cycles. They therefore have lower bypass priority.
        // acorn_passed[0..passed_1hop) holds the 1-hop passed slots in cmp-sorted
        // partition order.
        for (std::size_t i = 0; i < passed_1hop && collected < M; ++i)
            expand_2hop_from(acorn_passed[i].slot);

        // Phase 3: 2-hop expansion of ALREADY-VISITED 1-hop neighbors (lowest
        // priority). Visited slots were stashed by the caller in visited_stash
        // (acorn_visited_candidates). visited_1hop_count == 0 skips the loop.
        for (std::size_t i = 0; i < visited_1hop_count && collected < M; ++i)
            expand_2hop_from(visited_stash[i].slot);

        // Bridge state for the fallback, with the same semantics as
        // acorn_expand_two_hop_. In the dense branch the failed pool concatenates
        // per-1hop batches, so it is not globally ordered. The fallback therefore
        // re-sorts the post-stride bridge_pool before prefetch (needs_failed_sort=true).
        std::size_t const passed_2hop = collected - limit_1hop;
        bridge.failed_1hop = failed_1hop_stash;
        bridge.failed_2hop = failed_2hop;
        bridge.passed_2hop = passed_2hop;
        bridge.one_hop_count = unvisited_1hop;
        bridge.total_passed_limit = collected;
        bridge.needs_failed_sort = true;

        if (collected == 0)
            return true;

        if (!visits.reserve(visits.size() + collected))
            return false;

        if (!is_dummy<prefetch_at>()) {
            // Sort acorn_passed in place for the prefetch is_sorted contract.
            // The only downstream consumer, distance_compute_and_promote_, does
            // not depend on the order of its input, so reordering the collected
            // range is safe.
            if (!is_dummy<neighbor_cmp_at>() && collected > 1) {
                std::sort(acorn_passed.data(), acorn_passed.data() + collected,
                          [&](acorn_candidate_t const& a, acorn_candidate_t const& b) {
                              return neighbor_cmp(member_cref_t{node_at_(a.slot).ckey(), a.slot},
                                                  member_cref_t{node_at_(b.slot).ckey(), b.slot});
                          });
            }
            acorn_candidates_iterator_t pbegin(*this, acorn_passed.data());
            acorn_candidates_iterator_t pend(*this, acorn_passed.data() + collected);
            prefetch(pbegin, pend);
        }

        // All collected slots are filter-passing -> admit to both next and top.
        distance_compute_and_promote_(query, metric, context, collected, radius, top_limit);
        return true;
    }

    /// @brief  Distance-compute and promote a contiguous run of
    ///         filter-passing slots in acorn_passed[0..count). Each slot is
    ///         marked visited (skip on duplicate) and measured against the
    ///         query. When `top` is not full or the distance is below `radius`,
    ///         the slot is inserted into both `next` and `top`, and `radius`
    ///         follows the worst entry of `top`.
    template <typename value_at, typename metric_at>
    inline void distance_compute_and_promote_(value_at&& query, metric_at&& metric, context_t& ctx, std::size_t count,
                                              distance_t& radius, std::size_t top_limit) const noexcept {
        visits_hash_set_t& visits = ctx.visits;
        next_candidates_t& next = ctx.next_candidates;
        top_candidates_t& top = ctx.top_candidates;
        buffer_gt<acorn_candidate_t>& passed = ctx.acorn_passed_candidates;
        for (std::size_t i = 0; i < count; ++i) {
            compressed_slot_t slot = passed[i].slot;
            if (visits.set(slot))
                continue;
            ctx.promoted_count++;
            distance_t dist = ctx.measure(query, citerator_at(slot), metric);
            if (top.size() < top_limit || dist < radius) {
                next.insert({-dist, slot});
                top.insert({dist, slot}, top_limit);
                radius = top.top().distance;
            }
        }
    }

    /**
     *  @brief  Third-hop expansion for the ACORN-1 path, an ablation of the
     *          expansion-depth axis. No-op unless `three_hop_threshold > 0`.
     *
     *  Fires only when this cycle's 2-hop expansion came up short, with fewer
     *  than `connectivity_base * three_hop_threshold` passing 2-hop nodes, and
     *  the result heap still has room. It then walks the predicate-failing
     *  2-hop nodes one level deeper. That pool is the very one RACORN-1 would
     *  bridge from. It promotes the predicate-passing 3-hop neighbors, capped
     *  at `connectivity_base` (M0). The cap mirrors ACORN's per-expansion
     *  truncation to the degree bound.
     *
     *  Failing 3-hop nodes stay conduits: no distance computation, no candidate
     *  queue insert. That is exactly what separates a deeper fixed expansion
     *  from RACORN-1's frontier-admitted bridges.
     *
     *  The walk is the naive one: hubs in pool order, their neighbors in
     *  adjacency order. It stops as soon as `connectivity_base` passing nodes
     *  are collected. Nothing caps how many candidates may be examined before
     *  that. So at extreme selectivity a cycle can sweep the whole O(M0^3) pool;
     *  that cost is part of what the ablation measures. Dedup is by `visits`
     *  only. A failing 3-hop node may therefore be predicate-tested twice within
     *  a cycle. This costs a predicate call but cannot change the result.
     *
     *  @return `true` on success, `false` if out of memory.
     */
    template <typename value_at, typename metric_at, typename predicate_at>
    bool acorn_expand_third_hop_(                                       //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, //
        context_t& context, buffer_gt<acorn_candidate_t> const& acorn_failed,
        acorn_bridge_state_t const& bridge, distance_t& radius,
        std::size_t top_limit) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates;
        top_candidates_t& top = context.top_candidates;

        std::size_t const cap = config_.connectivity_base;
        std::size_t const trigger = static_cast<std::size_t>(static_cast<double>(cap) * context.three_hop_threshold);
        if (bridge.passed_2hop >= trigger || !bridge.failed_2hop || top.size() >= top_limit)
            return true;

        std::size_t const hub_begin = bridge.failed_1hop;
        std::size_t const num_hubs = bridge.failed_2hop;
        std::size_t promoted = 0;
        bool out_of_memory = false;

        auto walk_hub = [&](std::size_t hub_index) {
            neighbors_ref_t hop3 = neighbors_base_(node_at_(acorn_failed[hub_begin + hub_index].slot));
            std::size_t const hop3_count = hop3.size();
            if (!visits.reserve(visits.size() + hop3_count)) {
                out_of_memory = true;
                return;
            }
            for (std::size_t i = 0; i != hop3_count && promoted < cap; ++i) {
                compressed_slot_t slot = hop3[i];
                if (visits.test(slot))
                    continue;
                if (!context.evaluate(predicate, member_cref_t{node_at_(slot).ckey(), slot}))
                    continue; // conduit only — never distance-computed, never queued
                if (visits.set(slot))
                    continue;
                distance_t dist = context.measure(query, citerator_at(slot), metric);
                if (top.size() < top_limit || dist < radius) {
                    next.insert({-dist, slot});
                    top.insert({dist, slot}, top_limit);
                    radius = top.top().distance;
                }
                ++promoted;
            }
        };

        for (std::size_t i = 0; i < num_hubs && promoted < cap && !out_of_memory; ++i)
            walk_hub(i);
        return !out_of_memory;
    }

    /**
     *  @brief  RACORN-1 bridge fallback. When the 2-hop expansion did not return
     *          enough passing candidates, insert a stride-sampled subset of the
     *          2-hop predicate-failed slots into `next`. The traversal then
     *          continues through filter-failed nodes.
     *
     *  Let `bridge_budget = floor(one_hop_count * bridge_budget_ratio)`. The
     *  call returns without effect when `passed_2hop >= bridge_budget`, when
     *  `failed_2hop == 0`, or when `total_passed_limit >= bridge_budget` (no
     *  room for bridges). Otherwise it marks every failing 1-hop and 2-hop slot
     *  of this cycle visited, so later cycles skip their predicate evals. Then,
     *  if `top` is not yet full, it counts the cycle in `fallback_count` and
     *  inserts up to `bridge_budget - total_passed_limit` stride-sampled failing
     *  2-hop slots into `next`. `top` is not full at this point, so every one of
     *  them is inserted. When `top` is already full, the marking stands but
     *  nothing is inserted or counted.
     *
     *  Runs only when `bridge_budget_ratio > 0.0` (guarded at the call site).
     *  `acorn_failed` layout: see acorn_bridge_state_t.
     */
    template <typename value_at, typename metric_at, typename prefetch_at, typename neighbor_cmp_at>
    bool racorn_fallback_(                                            //
        value_at&& query, metric_at&& metric, prefetch_at&& prefetch, //
        neighbor_cmp_at&& neighbor_cmp,                               //
        context_t& context, next_candidates_t& next,                  //
        buffer_gt<acorn_candidate_t>& acorn_failed,                   //
        acorn_bridge_state_t const& bridge, std::size_t top_limit,    //
        top_candidates_t& top) const usearch_noexcept_m {

        std::size_t const failed_1hop = bridge.failed_1hop;
        std::size_t const failed_2hop = bridge.failed_2hop;
        std::size_t const passed_2hop = bridge.passed_2hop;
        std::size_t const one_hop_count = bridge.one_hop_count;
        std::size_t const total_passed_limit = bridge.total_passed_limit;
        bool const needs_failed_sort = bridge.needs_failed_sort;

        // Bridge budget scales with one_hop_count via bridge_budget_ratio.
        auto const bridge_budget =
            static_cast<std::size_t>(static_cast<double>(one_hop_count) * context.bridge_budget_ratio);

        // Off-cycle guard: the 2-hop layer alone already met the budget, or there is no
        // failing 2-hop slot to bridge from. passed_1hop is left out because it is the
        // already-rich graph-local region. The paper's Algorithm 1 has no such test; it
        // runs the fallback whenever |P| < tau, which is the guard below. This one is an
        // extra of the code. In the sparse branch `passed_2hop` counts the passing 2-hop
        // nodes before the M0 cap, so it can exceed total_passed_limit and skip a fallback
        // that Algorithm 1 would run. In the dense branch it is the capped count, and the
        // guard below implies this one. In the paper's experiments no cycle had
        // passed_2hop >= bridge_budget together with total_passed_limit < bridge_budget,
        // so the code and Algorithm 1 never disagreed.
        if (passed_2hop >= bridge_budget || failed_2hop == 0)
            return true;

        // No bridge room: the paper's test |P| < tau, with total_passed_limit as |P|. When
        // the passing candidates already meet the budget, fallback_limit would be 0, so
        // there is nothing to insert. Return before marking and counting, so fallback_count
        // and the failed-pool visit-marking reflect only cycles that attempt a bridge
        // insertion.
        if (total_passed_limit >= bridge_budget)
            return true;

        // Bulk-mark the 1-hop + 2-hop predicate-failed slots as visited BEFORE stride
        // sampling, which would overwrite the non-selected 2-hop entries. Marking must
        // happen first, on the contiguous original list. Off-cycles that returned above
        // leave both ranges unmarked. This matches ACORN-1's "no failed-mark" behavior.
        {
            visits_hash_set_t& visits = context.visits;
            std::size_t const mark_total = failed_1hop + failed_2hop;
            if (!visits.reserve(visits.size() + mark_total))
                return false;
            for (std::size_t i = 0; i < mark_total; ++i)
                visits.set(acorn_failed[i].slot);
        }

        // Top already full: no point inserting more bridges. Counters are not advanced;
        // only cycles that actually attempt bridge insertion count toward fallback_count.
        if (top.size() >= top_limit)
            return true;

        context.fallback_count++;

        // total_passed_limit < bridge_budget is guaranteed by the early return above,
        // so this subtraction is always positive.
        std::size_t fallback_need = bridge_budget - total_passed_limit;
        std::size_t fallback_limit = (std::min)(failed_2hop, fallback_need);
        acorn_candidate_t* bridge_pool = acorn_failed.data() + failed_1hop;
        // Skipped when stride_sampling is off: the head `fallback_limit` entries
        // stand as the bridge selection (prefix truncation, ablation baseline).
        if (context.stride_sampling)
            stride_downsample_inplace(bridge_pool, failed_2hop, fallback_limit);

        // Sort the bridge pool by neighbor_cmp before prefetch. This is done only
        // when the caller signals that the failed-2hop pool is not already
        // key-sorted (needs_failed_sort, set per branch at the bridge-state assignments).
        if (needs_failed_sort && !is_dummy<neighbor_cmp_at>() && fallback_limit > 1) {
            std::sort(bridge_pool, bridge_pool + fallback_limit,
                      [&](acorn_candidate_t const& a, acorn_candidate_t const& b) {
                          return neighbor_cmp(member_cref_t{node_at_(a.slot).ckey(), a.slot},
                                              member_cref_t{node_at_(b.slot).ckey(), b.slot});
                      });
        }

        if (!is_dummy<prefetch_at>() && fallback_limit > 0) {
            acorn_candidates_iterator_t pbegin(*this, bridge_pool);
            acorn_candidates_iterator_t pend(*this, bridge_pool + fallback_limit);
            prefetch(pbegin, pend);
        }

        // Insert the selected bridges into next; the visited mark was done above. W is
        // not full here, because of the return above. So the radius test admits every
        // bridge and `inserted` equals `fallback_limit`.
        std::size_t inserted = 0;
        for (std::size_t i = 0; i < fallback_limit; ++i) {
            compressed_slot_t slot = bridge_pool[i].slot;
            distance_t dist = context.measure(query, citerator_at(slot), metric);
            if (top.size() < top_limit || dist < top.top().distance) {
                next.insert({-dist, slot});
                inserted++;
            }
        }
        context.bridge_size_sum += inserted;
        context.bridge_sampled_sum += fallback_limit;
        return true;
    }
