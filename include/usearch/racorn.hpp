// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 *  @file   racorn.hpp
 *  @brief  RACORN-1 / ACORN-1 search member functions — class-body fragment.
 *
 *  Included inside the index_gt class body in index.hpp.
 *  Everything in this file becomes a member of index_gt.
 *
 *  References:
 *    - ACORN-1: "ACORN: Performant and Predicate-Agnostic Approximate Nearest
 *      Neighbor Search over Vector Embeddings and Structured Data"
 *      https://arxiv.org/abs/2403.04871
 *    - RACORN-1 / RACORN-1+: "RACORN: Recall-Adaptive Filtered Approximate
 *      Nearest Neighbor Search via Bridge-Node Fallback"
 *      https://arxiv.org/abs/2607.00768
 *
 *  Functions defined here:
 *    - search_acorn1_in_base_    ACORN-1 / RACORN-1 base-layer search loop
 *    - acorn_expand_two_hop_     2-hop expansion branch
 *    - exact_fallback            RACORN-1+ adaptive exact-search heuristic
 *    - racorn_fallback_          RACORN-1 bridge-node fallback
 *    - acorn_3hop_expansion_     3-hop expansion (USEARCH_ENABLE_3HOP only)
 *
 *  Namespace-scope utility (stride_downsample_inplace) is in racorn_util.hpp,
 *  included at namespace scope before this file is reached.
 *
 *  Do not include this file directly from user code.
 */

    /**
     *  @brief  ACORN-1 filtered search loop for the base layer.
     *          Uses adaptive two-hop expansion when predicate density is low.
     *  @return `true` if procedure succeeded, `false` if run out of memory.
     */
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at>
    bool search_acorn1_in_base_(                                                                //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        distance_t& radius, std::size_t top_limit, context_t& context) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates;
        top_candidates_t& top = context.top_candidates;
        double const two_hop_threshold = context.two_hop_threshold;

        // Per-thread scratch buffers reused across cycles for 1-hop / 2-hop candidates
        // (and 3-hop when USEARCH_ENABLE_3HOP=1). Sized once at try_reserve time.
        buffer_gt<candidate_t>& acorn_passed = context.acorn_passed_candidates;

        while (!next.empty()) {
            // RACORN-1+ exact-fallback heuristic flagged this query — exit
            // the base traversal so `search()` can run `search_exact_`.
            if (context.fallback_to_exact)
                break;
            candidate_t candidate = next.top();
            if ((-candidate.distance) > radius && top.size() == top_limit)
                break;

            next.pop();
            context.iteration_cycles++;

            neighbors_ref_t candidate_neighbors = neighbors_base_(node_at_(candidate.slot));
            std::size_t const neighbor_count = candidate_neighbors.size();

            // ---- Collect unvisited 1-hop neighbors, predicate ----
            std::size_t unvisited_1hop = 0;
            for (compressed_slot_t slot : candidate_neighbors) {
                if (visits.test(slot))
                    continue;
                acorn_passed[unvisited_1hop++] = {0, slot};
            }

            if (unvisited_1hop == 0)
                continue;

            // Partition: swap predicate-passed to front of acorn_passed.
            if (!is_dummy<predicate_at>())
                context.predicate_evaluations += unvisited_1hop;
            std::size_t passed_1hop = 0;
            for (std::size_t i = 0; i < unvisited_1hop; ++i) {
                if (predicate(member_cref_t{node_at_(acorn_passed[i].slot).ckey(), acorn_passed[i].slot})) {
                    std::swap(acorn_passed[passed_1hop], acorn_passed[i]);
                    passed_1hop++;
                }
            }

            double ratio = static_cast<double>(passed_1hop) / static_cast<double>(unvisited_1hop);

            if (ratio > two_hop_threshold) {
                // ---- High predicate density: no 2-hop expansion ----
                // Intentional asymmetry vs the low-density branch: here we still
                // compute distance for predicate-failed 1-hop slots and push them
                // into `next`, mirroring vanilla HNSW traversal. Rationale:
                //   * Low-selectivity filters are where HNSW alone struggles, so
                //     RACORN-1's fallback / 2-hop expansion exists for them.
                //   * Mid/high selectivity prefers recall, so we want
                //     graph-traversal continuity through filter-failed nodes —
                //     the same nodes HNSW would visit anyway.
                // Order vs HNSW: HNSW computes distance first then filters at
                // the result stage; ACORN evaluates the predicate first and
                // computes distance on every unvisited 1-hop here regardless.
                // Only predicate-passed slots (i < passed_1hop) are admitted to
                // `top`; failed slots act purely as traversal stepping stones.
                if (!visits.reserve(visits.size() + unvisited_1hop))
                    return false;

                for (std::size_t i = 0; i < unvisited_1hop; ++i) {
                    compressed_slot_t slot = acorn_passed[i].slot;
                    if (visits.set(slot))
                        continue;
                    distance_t dist = context.measure(query, citerator_at(slot), metric);
                    if (top.size() < top_limit || dist < radius) {
                        next.insert({-dist, slot});
                        if (i < passed_1hop) {
                            top.insert({dist, slot}, top_limit);
                            radius = top.top().distance;
                        }
                    }
                }
                continue;
            }

            // ---- Low predicate density: 2-hop expansion ----
            if (!acorn_expand_two_hop_(query, metric, predicate, prefetch, //
                                       context, candidate_neighbors, passed_1hop, unvisited_1hop, radius, top_limit))
                return false; // OOM
        }
        return true;
    }

    /**
     *  @brief  Low-density 2-hop expansion branch of `search_acorn1_in_base_`, including
     *          1-hop-failed stash (RACORN-1), 2-hop neighbor collection, predicate
     *          partition, stride sampling (RACORN-1), unified passed-list insert into
     *          next/top, and the RACORN-1 fallback / 3-hop expansion dispatch. The
     *          1-hop + 2-hop failed visits-marking is deferred to `racorn_fallback_`
     *          and only applied on cycles that actually fire bridge supplementation.
     *
     *  Splits the inner ACORN loop into a focused unit so the main while-loop body
     *  stays readable. Behavior is identical to the previous inlined block; only the
     *  function boundary changes.
     *
     *  @param[in]  candidate_neighbors  the candidate node's 1-hop neighbors (raw order)
     *  @param[in]  passed_1hop          number of 1-hop predicate-passed slots, sitting at
     *                                   acorn_passed[0..passed_1hop)
     *  @param[in]  unvisited_1hop       total unvisited 1-hop slots (passed + failed) at
     *                                   acorn_passed[0..unvisited_1hop)
     *  @param[in,out] radius            search radius (top-K worst distance), updated when
     *                                   `top` admits a closer candidate
     *  @param[in]  top_limit            top-K capacity (= ef_search)
     *  @return `true` on success, `false` on OOM.
     */
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at>
    bool acorn_expand_two_hop_(                                                                 //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        context_t& context,                                                                     //
        neighbors_ref_t candidate_neighbors,                                                    //
        std::size_t passed_1hop, std::size_t unvisited_1hop,                                    //
        distance_t& radius, std::size_t top_limit) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates;
        top_candidates_t& top = context.top_candidates;
        std::size_t const connectivity_base = config_.connectivity_base;
        buffer_gt<candidate_t>& acorn_passed = context.acorn_passed_candidates;
        buffer_gt<candidate_t>& acorn_failed = context.acorn_failed_candidates;

        // Seed the per-cycle hop-visit set with all unvisited 1-hop (passed +
        // failed) so they are not re-collected as 2-hop candidates within this
        // cycle. The same set is reused (without clearing) by
        // acorn_3hop_expansion_, so 1-hop / 2-hop slots seen here are
        // automatically excluded from the 3-hop candidate pool.
        cycle_visits_hash_set_t& hop2_seen = context.acorn_hop_visits;
        hop2_seen.clear();
        for (std::size_t i = 0; i < unvisited_1hop; ++i) {
            hop2_seen.set(acorn_passed[i].slot);
        }

        // RACORN-1 only: stash 1-hop predicate-failed slots into acorn_failed[0..) before
        // 2-hop collection overwrites acorn_passed[passed_1hop..). They are NOT marked
        // visited yet — racorn_fallback_ does the bulk-mark only on cycles where it
        // actually fires, so off-cycles (high/mid selectivity, where bridge
        // supplementation is unnecessary) leave 1-hop failed unmarked. This recovers
        // ACORN-1's broader-exploration recall in those ranges. ACORN-1 (racorn=false)
        // keeps stash=0, so acorn_failed retains its prior layout.
        std::size_t const failed_1hop_stash = context.racorn ? unvisited_1hop - passed_1hop : 0;
        for (std::size_t i = 0; i < failed_1hop_stash; ++i) {
            acorn_failed[i] = acorn_passed[passed_1hop + i];
        }

        // Collect 2-hop neighbors after 1-hop passed in the buffer.
        std::size_t two_hop_count = 0;
        for (compressed_slot_t hop1_slot : candidate_neighbors) {
            neighbors_ref_t hop2_neighbors = neighbors_base_(node_at_(hop1_slot));
            for (compressed_slot_t hop2_slot : hop2_neighbors) {
                if (visits.test(hop2_slot))
                    continue;
                if (hop2_seen.set(hop2_slot))
                    continue;
                acorn_passed[passed_1hop + two_hop_count] = {0, hop2_slot};
                two_hop_count++;
            }
        }

        // Nothing to process this cycle.
        if (two_hop_count == 0 && passed_1hop == 0)
            return true;

        // predicate on 2-hop → compact passed after 1-hop in acorn_passed, failed
        // appended after the 1-hop stash in acorn_failed (offset = failed_1hop_stash;
        // 0 for ACORN-1 which keeps the original [0..failed_2hop) layout).
        if (!is_dummy<predicate_at>())
            context.predicate_evaluations += two_hop_count;
        std::size_t passed_2hop = 0;
        std::size_t failed_2hop = 0;
        for (std::size_t i = 0; i < two_hop_count; ++i) {
            candidate_t c = acorn_passed[passed_1hop + i];
            if (predicate(member_cref_t{node_at_(c.slot).ckey(), c.slot})) {
                acorn_passed[passed_1hop + passed_2hop] = c;
                passed_2hop++;
            } else {
                acorn_failed[failed_1hop_stash + failed_2hop++] = c;
            }
        }

        // Unified processing: 1-hop passed + 2-hop passed with budget = connectivity_base.
        // RACORN-1 stride-samples the 2-hop portion when it would otherwise exceed the
        // remaining budget, spreading the selection across the passed pool instead of
        // taking only the head.
        std::size_t limit_1hop = (std::min)(passed_1hop, connectivity_base);
        std::size_t remaining = connectivity_base - limit_1hop;
        std::size_t limit_2hop = (std::min)(passed_2hop, remaining);

        if (context.racorn && context.racorn_stride_sampling)
            stride_downsample_inplace(acorn_passed.data() + passed_1hop, passed_2hop, limit_2hop);

        std::size_t total_passed_limit = limit_1hop + limit_2hop;
        if (!visits.reserve(visits.size() + total_passed_limit))
            return false;

        for (std::size_t i = 0; i < total_passed_limit; ++i) {
            compressed_slot_t slot = acorn_passed[i].slot;
            if (visits.set(slot))
                continue;
            distance_t dist = context.measure(query, citerator_at(slot), metric);
            if (top.size() < top_limit || dist < radius) {
                next.insert({-dist, slot});
                top.insert({dist, slot}, top_limit);
                radius = top.top().distance;
            }
        }

        // ── RACORN-1 / 3-hop dispatch (mutually exclusive).
        // Failed slots' visited-mark is deferred:
        //   - RACORN-1: racorn_fallback_ bulk-marks 1-hop + 2-hop failed only on cycles
        //     where it actually fires (passed_2hop < bridge_budget && failed_2hop > 0).
        //     Off-cycles leave both unmarked, recovering ACORN-1's broader-exploration
        //     recall in high/mid selectivity ranges where fallback rarely triggers.
        //   - ACORN-1 (incl. 3-hop): never bulk-marked so the same node may resurface
        //     via another candidate's neighborhood in later cycles.
        if (context.racorn) {
            if (!racorn_fallback_(query, metric, context, next, acorn_failed,  //
                                  failed_1hop_stash, failed_2hop, passed_2hop, //
                                  unvisited_1hop, two_hop_count, total_passed_limit, top_limit, top))
                return false; // OOM
        }
#if USEARCH_ENABLE_3HOP
        else if (context.three_hop_threshold >= 0) {
            if (!acorn_3hop_expansion_(query, metric, predicate,              //
                                       context, next, top, radius, top_limit, //
                                       passed_2hop, failed_2hop))
                return false; // OOM
        }
#endif
        return true;
    }

    /**
     *  @brief  RACORN-1+ adaptive exact-search heuristic. Returns `true` when
     *          this query should bail out of the RACORN-1 fallback and finish
     *          via `search_exact_` instead.
     *
     *  Trigger: enabled (`fallback_exact_threshold > 0`) AND the warm-up window
     *  has passed (`fb_count > top_limit / 2`, scales with `ef_search`) AND the
     *  observed bridge selectivity (`fb_pass / fb_eval`) is at or below the
     *  threshold. Only this query's deltas are used — `fallback_*_before`
     *  snapshots are taken once at `search()` entry.
     */
    bool exact_fallback(context_t& context, std::size_t top_limit) const noexcept {
        if (context.fallback_exact_threshold <= 0)
            return false;
        std::size_t const fb_count = context.fallback_count - context.fallback_count_before;
        if (fb_count <= top_limit / 2)
            return false;
        std::size_t const fb_eval = context.fallback_evaluated - context.fallback_evaluated_before;
        std::size_t const fb_pass = context.fallback_passed - context.fallback_passed_before;
        double const fb_sel = fb_eval > 0 ? static_cast<double>(fb_pass) / static_cast<double>(fb_eval) : 0.0;
        return fb_sel <= context.fallback_exact_threshold;
    }

    /**
     *  @brief  RACORN-1 fallback: stride-picks a subset of failed 2-hop candidates into
     *          the next queue. Bulk-marks all 1-hop + 2-hop failed as visited whenever the
     *          initial trigger guard passes (sparse 2-hop + non-empty failed set), even
     *          if `top` is already full — preventing redundant predicate evals in
     *          subsequent cycles. When `top` is full, counters
     *          (`fallback_count` / `fallback_evaluated` / `fallback_passed`) are NOT
     *          incremented either — only cycles that actually attempt bridge insertion
     *          contribute to those statistics.
     *
     *  RACORN-1+ extension: at the end of every fallback-firing cycle, calls
     *  `exact_fallback` and sets `context.fallback_to_exact = true` if the
     *  heuristic decides the query is better served by exhaustive exact
     *  search. The caller (`search()`) then dispatches `search_exact_` after
     *  the base loop bails out.
     *
     *  @param[in]  query              query vector.
     *  @param[in]  metric             distance metric.
     *  @param[in]  context            per-thread scratch space; fallback counters updated.
     *  @param[in]  next               min-heap of unprocessed candidates (modified).
     *  @param[in]  acorn_failed       flat buffer holding stashed failed slots:
     *                                   [0..failed_1hop)               = 1-hop predicate-failed
     *                                   [failed_1hop..failed_1hop+failed_2hop) = 2-hop predicate-failed (bridge pool)
     *  @param[in]  failed_1hop        number of 1-hop predicate-failed slots in acorn_failed.
     *  @param[in]  failed_2hop        number of 2-hop predicate-failed slots in acorn_failed.
     *  @param[in]  passed_2hop        number of 2-hop predicate-passed slots (fallback trigger guard).
     *  @param[in]  one_hop_count      total unvisited 1-hop neighbor count (= bridge budget scale).
     *  @param[in]  two_hop_count      total 2-hop candidates collected this cycle.
     *  @param[in]  total_passed_limit total 1-hop + 2-hop passed actually processed this cycle.
     *  @param[in]  top_limit          top-K capacity (= ef_search).
     *  @param[in]  top                sorted top-K result buffer.
     *  @return  `true` on success (normal completion or no-op), `false` on OOM.
     */
    template <typename value_at, typename metric_at>
    bool racorn_fallback_(                                                             //
        value_at&& query, metric_at&& metric,                                          //
        context_t& context, next_candidates_t& next,                                   //
        buffer_gt<candidate_t>& acorn_failed,                                          //
        std::size_t failed_1hop, std::size_t failed_2hop,                              //
        std::size_t passed_2hop, std::size_t one_hop_count, std::size_t two_hop_count, //
        std::size_t total_passed_limit, std::size_t top_limit,                         //
        top_candidates_t& top) const noexcept {

        // Bridge budget scales with one_hop_count via bridge_budget_ratio (default 1.0).
        // Used both as the fallback trigger threshold and as the ceiling for fallback_need.
        auto const bridge_budget =
            static_cast<std::size_t>(static_cast<double>(one_hop_count) * context.bridge_budget_ratio);

        // Fallback triggers when 2-hop expansion did not find enough passed candidates
        // (i.e., 2-hop layer is sparse). passed_1hop is excluded since it represents
        // the already-rich graph-local region, while bridges extend into filter-failed zones.
        // ratio=0 disables fallback entirely.
        if (passed_2hop >= bridge_budget || failed_2hop == 0)
            return true;

        // Bulk-mark 1-hop + 2-hop predicate-failed as visited BEFORE stride sampling.
        // acorn_failed layout (set up by acorn_expand_two_hop_):
        //   [0..failed_1hop)                       = 1-hop failed (stash)
        //   [failed_1hop..failed_1hop+failed_2hop) = 2-hop failed (bridge pool)
        // Stride sampling below would overwrite non-selected 2-hop entries, so marking
        // must happen first on the contiguous original list.
        // Off-cycles short-circuited by the guard above leave both ranges unmarked,
        // matching ACORN-1's "no failed-mark" exploration on cycles that don't need
        // bridge supplementation.
        {
            visits_hash_set_t& visits = context.visits;
            std::size_t const mark_total = failed_1hop + failed_2hop;
            if (!visits.reserve(visits.size() + mark_total))
                return false;
            for (std::size_t i = 0; i < mark_total; ++i) {
                visits.set(acorn_failed[i].slot);
            }
        }

        // Top is already full: no point inserting more bridges (search has converged
        // its K-best). Counters are deliberately not advanced — only cycles that
        // actually attempt bridge insertion count toward FBCount. The bulk-mark above
        // still applies, so subsequent cycles skip these slots' predicate evals.
        if (top.size() >= top_limit)
            return true;

        context.fallback_evaluated += one_hop_count + two_hop_count;
        context.fallback_passed += total_passed_limit;
        context.fallback_count++;

        // RACORN-1+ adaptive exact-search heuristic. Evaluated *before* stride
        // sampling and bridge insertion so a positive decision skips both —
        // saving the stride pass + per-bridge distance compute that would
        // otherwise be wasted on a cycle whose results never get consumed.
        // The caller (`search_acorn1_in_base_`) breaks out of the main loop
        // on the next iteration via the `context.fallback_to_exact` flag,
        // and `search()` then dispatches `search_exact_`.
        if (exact_fallback(context, top_limit)) {
            context.fallback_to_exact = true;
            return true;
        }

        // Saturating subtraction guards against total_passed_limit > bridge_budget
        // (safe under guard above, but explicit for robustness).
        std::size_t fallback_need = bridge_budget > total_passed_limit ? bridge_budget - total_passed_limit : 0;
        std::size_t fallback_limit = (std::min)(failed_2hop, fallback_need);
        candidate_t* bridge_pool = acorn_failed.data() + failed_1hop;
        if (context.racorn_stride_sampling)
            stride_downsample_inplace(bridge_pool, failed_2hop, fallback_limit);

        // Insert selected bridges into next. (visit-marking already done in the bulk-mark
        // step above, so no need to mark per-bridge here.)
        for (std::size_t i = 0; i < fallback_limit; ++i) {
            compressed_slot_t slot = bridge_pool[i].slot;
            distance_t dist = context.measure(query, citerator_at(slot), metric);
            if (top.size() < top_limit || dist < top.top().distance) {
                next.insert({-dist, slot});
            }
        }
        return true;
    }

#if USEARCH_ENABLE_3HOP
    /**
     *  @brief  3-hop expansion: when `passed_2hop < connectivity_base * three_hop_threshold`,
     *          expand from each 2-hop failed node to its neighbors (= 3-hop from candidate),
     *          run predicate on them, and add passed entries to next/top.
     *
     *  Mirrors the 2-hop expansion pattern (collect → predicate-partition →
     *  budget → distance + insert). Mutually exclusive with `racorn` (bridge fallback).
     *
     *  Trigger guards: skip when 2-hop layer is already rich enough, no failed-2-hop hubs
     *  to expand from, or `top` is already full (no room for new results).
     *
     *  Buffer reuse:
     *    - `acorn_passed` (1-hop+2-hop already processed) is reused for the 3-hop pool.
     *      Its capacity is `connectivity_base * acorn_buf_size` (≈ M_base³), wide enough
     *      to hold every unique 3-hop candidate without early termination.
     *    - `acorn_failed` (still holds 2-hop failed) is read first as the hub list, then
     *      reused for 3-hop predicate-failed entries (same buffer, same large capacity).
     *    - `acorn_hop_visits` is the per-cycle dedup set already populated by
     *      `acorn_expand_two_hop_` (1-hop + 2-hop slots). 3-hop adds onto it
     *      *without* clearing, so any 1-hop / 2-hop slot already evaluated in
     *      this cycle is automatically excluded from the 3-hop candidate pool.
     *      The set's capacity is sized to the 3-hop pool (≈ M³) when 3-hop is
     *      enabled, and the generation counter keeps per-cycle clear O(1).
     *
     *  Notes:
     *    - Budget cap: 3-hop passed are inserted into next/top up to `connectivity_base`.
     *      When the passed pool exceeds the budget, stride sampling spreads the selection
     *      across the pool to avoid head-bias toward earlier hubs.
     *    - 3-hop predicate-failed nodes are NOT bulk-marked visited (matches the
     *      ACORN-1 convention; 3-hop expansion is ACORN-1 only with racorn=false).
     *
     *  @return `true` on success, `false` on OOM.
     */
    template <typename value_at, typename metric_at, typename predicate_at>
    bool acorn_3hop_expansion_(                                           //
        value_at&& query, metric_at&& metric, predicate_at&& predicate,   //
        context_t& context, next_candidates_t& next,                      //
        top_candidates_t& top, distance_t& radius, std::size_t top_limit, //
        std::size_t passed_2hop, std::size_t failed_2hop) const usearch_noexcept_m {

        std::size_t const connectivity_base = config_.connectivity_base;
        auto const trigger_threshold =
            static_cast<std::size_t>(static_cast<double>(connectivity_base) * context.three_hop_threshold);
        if (passed_2hop >= trigger_threshold || failed_2hop == 0 || top.size() >= top_limit)
            return true;

        visits_hash_set_t& visits = context.visits;
        buffer_gt<candidate_t>& acorn_passed = context.acorn_passed_candidates;
        buffer_gt<candidate_t>& acorn_failed = context.acorn_failed_candidates;
        // Reuse the same per-cycle dedup set seeded by acorn_expand_two_hop_
        // (1-hop + 2-hop slots already inserted). No clear() — that's the whole
        // point: 3-hop candidates that coincide with already-evaluated 1-hop /
        // 2-hop slots short-circuit at hop_seen.set() and skip predicate work.
        cycle_visits_hash_set_t& hop_seen = context.acorn_hop_visits;

        // Snapshot 2-hop failed list (hubs) into a stable view since we'll overwrite acorn_failed.
        std::size_t const num_hubs = failed_2hop;

        // Collect 3-hop candidates into acorn_passed (reusing the buffer).
        std::size_t three_hop_count = 0;
        for (std::size_t hi = 0; hi < num_hubs; ++hi) {
            compressed_slot_t hub = acorn_failed[hi].slot;
            neighbors_ref_t hub_neighbors = neighbors_base_(node_at_(hub));
            for (compressed_slot_t hop3_slot : hub_neighbors) {
                if (visits.test(hop3_slot))
                    continue;
                if (hop_seen.set(hop3_slot))
                    continue;
                acorn_passed[three_hop_count++] = {0, hop3_slot};
            }
        }

        if (three_hop_count == 0)
            return true;

        // Predicate partition: passed at front of acorn_passed, failed reused into acorn_failed.
        if (!is_dummy<predicate_at>())
            context.predicate_evaluations += three_hop_count;
        std::size_t passed_3hop = 0;
        std::size_t failed_3hop = 0;
        for (std::size_t i = 0; i < three_hop_count; ++i) {
            candidate_t c = acorn_passed[i];
            if (predicate(member_cref_t{node_at_(c.slot).ckey(), c.slot})) {
                acorn_passed[passed_3hop++] = c;
            } else {
                acorn_failed[failed_3hop++] = c;
            }
        }

        std::size_t limit_3hop = (std::min)(passed_3hop, connectivity_base);

        if (!visits.reserve(visits.size() + limit_3hop))
            return false;

        // Insert 3-hop passed into next + top.
        for (std::size_t i = 0; i < limit_3hop; ++i) {
            compressed_slot_t slot = acorn_passed[i].slot;
            if (visits.set(slot))
                continue;
            distance_t dist = context.measure(query, citerator_at(slot), metric);
            if (top.size() < top_limit || dist < radius) {
                next.insert({-dist, slot});
                top.insert({dist, slot}, top_limit);
                radius = top.top().distance;
            }
        }

        return true;
    }
#endif // USEARCH_ENABLE_3HOP
