// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 *  @file   navix.hpp
 *  @brief  NaviX filtered-search heuristics, reimplemented on the base-layer
 *          loop of `index_gt` — a class-body fragment.
 *
 *  index.hpp includes this file once, inside the `index_gt` class body, next
 *  to racorn.hpp. It has no include guard and must not be included directly:
 *  include <usearch/index.hpp>.
 *
 *  Reference: Gaurav Sehgal and Semih Salihoglu, "NaviX: A Native Vector
 *  Index Design for Graph DBMSs With Robust Predicate-Agnostic Search
 *  Performance", VLDB 2025. The authors' code: https://github.com/gaurav8297/kuzu
 *  (MIT License). It is the paper's artifact: branch gaurav8297/hnsw_index at
 *  commit 5f149a2a716c8caecc0b6140dd4a2504f58fbd8b, function navixFilteredSearch
 *  in src/function/gds/vector_search.cpp. The benchmark's navix-local mode runs
 *  heuristic 8, the form of that code. Its navix-local-paper mode runs 5, the
 *  form of the paper's text. Two rules of that code are not kept: the switch to
 *  brute force when at most 30,000 nodes pass, and the empty result at
 *  selectivity 0.1% and below. Here the graph search runs throughout. The
 *  heuristics (onehop-s, blind, directed, adaptive-local) are selected per
 *  query by `index_search_config_t::navix_heuristic`. `search_to_find_in_base_`
 *  dispatches here before the filter-first / ACORN-1 decision.
 *
 *  The reimplementation shares the data structures of the ACORN-1 path
 *  (`context_t::acorn_hop_visits`, `acorn_visited_candidates`,
 *  `qualifying_candidates`). The algorithms are therefore compared on
 *  identical graphs and with identical scratch costs. Predicate-failing nodes
 *  are two-hop conduits only: they never enter the candidate queue. This is
 *  the defining contrast with RACORN-1 bridges (racorn.hpp).
 */

#ifndef UNUM_USEARCH_HPP
#error "navix.hpp is a class-body fragment of index.hpp; include <usearch/index.hpp> instead"
#endif

    /**
     *  @brief  NaviX (VLDB 2025) filtered-search heuristics on the base-layer
     *          loop of ACORN-1/RACORN-1. Selected via
     *          `index_search_config_t::navix_heuristic`:
     *            1 = onehop-s       : only predicate-passing 1-hop neighbors are
     *                                 distance-computed and expanded.
     *            2 = blind          : passing 1-hop neighbors first, then 2-hop
     *                                 through every 1-hop conduit in adjacency
     *                                 order. Stops once M0 selected nodes are
     *                                 collected in this expansion.
     *            3 = directed       : distances to ALL 1-hop neighbors are paid up
     *                                 front. 2-hop conduits are processed
     *                                 nearest-first, with the same M0 collection cap.
     *            8 = adaptive-local as the paper's artifact (gaurav8297/kuzu,
     *                                 navixFilteredSearch) runs it. Before the base loop,
     *                                 the first 10 passing nodes in slot order join both
     *                                 queues. Then, per candidate, with d = 1-hop list size
     *                                 and f = passing nodes in it:
     *                                 f/d >= 0.4                     -> onehop-s
     *                                 (d*f + f)*0.4 > d + (d - f)    -> directed: distances
     *                                   to the unvisited 1-hop only. The unvisited failing
     *                                   ones are conduits, nearest first, and are marked
     *                                   visited. Stop once d passing nodes have been seen.
     *                                 else -> full two-hop: the unvisited failing 1-hop
     *                                   neighbors are expanded and marked visited. Every
     *                                   passing 1- and 2-hop node joins. No collection cap.
     *                                 In the code, 8 resolves per candidate to 1 (onehop-s),
     *                                 6 (the authors' directed form above) or 9 (the full
     *                                 two-hop above).
     *            5 = adaptive-local as the NaviX paper's text states it:
     *                                 sigma_l >= 0.5              -> onehop-s
     *                                 sigma_l*(M0+1)*M0 >= M0*lf  -> directed (lf=3)
     *                                 else                        -> blind.
     *          Faithfulness notes: in 2, 3 and 5 the per-expansion collection
     *          cap is the base-layer degree bound M0 (`connectivity_base`), which
     *          matches the paper's M, and the thresholds are the paper defaults.
     *          8 follows the authors' code instead: no cap in the full two-hop,
     *          and the thresholds above. Conduits are walked in raw adjacency
     *          order; neighbor_cmp is intentionally not applied. The exception is
     *          directed, which orders them nearest-first.
     */
    template <typename value_at, typename metric_at, typename predicate_at, typename prefetch_at>
    bool search_navix_in_base_(                                                                 //
        value_at&& query, metric_at&& metric, predicate_at&& predicate, prefetch_at&& prefetch, //
        distance_t& radius, std::size_t top_limit, context_t& context,
        early_termination_gate_t& early_termination) const usearch_noexcept_m {

        visits_hash_set_t& visits = context.visits;
        next_candidates_t& next = context.next_candidates;
        top_candidates_t& top = context.top_candidates;
        // Scratch reuse: passing-1hop *indices* (values < M0 fit any slot type),
        // and (distance, slot) pairs for the directed 1-hop ordering.
        buffer_gt<acorn_candidate_t>& idx_stash = context.acorn_visited_candidates;
        buffer_gt<candidate_t>& onehop_buf = context.qualifying_candidates;
        cycle_visits_hash_set_t& hop_seen = context.acorn_hop_visits;

        std::size_t const heuristic_cfg = context.navix_heuristic;
        std::size_t const m0 = config_.connectivity_base;
        double const onehop_s_ub = 0.5; // NaviX ub default
        double const leniency = 3.0;    // NaviX lf default

        // Same phase accounting as the HNSW and ACORN-1 loops.
        enum { stop_queue = 0, stop_rule = 1, stop_deadline = 2 } stop = stop_queue;
        std::size_t const dist0 = context.computed_distances, pred0 = context.predicate_evaluations;
        bool fill_marked = false;
        if (heuristic_cfg == 8) {
            // The authors' code: the first 10 passing nodes in slot order join both queues
            // before the base loop starts (addFilteredNodesToCandidates).
            std::size_t seeded = 0;
            for (std::size_t s = 0; s != size() && seeded != 10; ++s) {
                compressed_slot_t const slot = static_cast<compressed_slot_t>(s);
                if (!context.evaluate(predicate, member_cref_t{node_at_(slot).ckey(), slot}))
                    continue;
                seeded++;
                if (!visits.reserve(visits.size() + 1))
                    return false;
                if (visits.set(slot))
                    continue;
                distance_t const dist = context.measure(query, citerator_at(slot), metric);
                if (top.size() < top_limit || dist < radius) {
                    next.insert({-dist, slot});
                    top.insert({dist, slot}, top_limit);
                    radius = top.top().distance;
                }
            }
        }
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
            if (top.size() < top_limit) {
                context.cycles_prefill++;
            } else if (!fill_marked) {
                context.dist_prefill += context.computed_distances - dist0;
                context.pred_prefill += context.predicate_evaluations - pred0;
                // One clock read per query at most: this branch runs once.
                context.prefill_ms = early_termination.elapsed_ms();
                snapshot_fill_(context, top);
                fill_marked = true;
            }

            neighbors_ref_t nbrs = neighbors_base_(node_at_(candidate.slot));
            std::size_t const n1 = nbrs.size();
            if (n1 == 0)
                continue;

            if (!is_dummy<prefetch_at>()) {
                candidates_range_t missing{*this, nbrs, visits};
                prefetch(missing.begin(), missing.end());
            }

            // Semimask pass: one predicate verdict per 1-hop neighbor, visited or
            // not. It feeds sigma_l and the passing-index list. No distances here.
            std::size_t passed_1hop = 0;
            for (std::size_t ni = 0; ni != n1; ++ni) {
                compressed_slot_t slot = nbrs[ni];
                if (context.evaluate(predicate, member_cref_t{node_at_(slot).ckey(), slot}))
                    idx_stash[passed_1hop++].slot = static_cast<compressed_slot_t>(ni);
            }

            std::size_t heuristic = heuristic_cfg;
            if (heuristic == 8) { // adaptive-local, the Kuzu artifact's choice
                double const d = static_cast<double>(n1), f = static_cast<double>(passed_1hop);
                if (f / d >= 0.4)
                    heuristic = 1;
                else if ((d * f + f) * 0.4 > d + (d - f))
                    heuristic = 6; // directed, the authors' form
                else
                    heuristic = 9; // full two-hop, Kuzu form
            } else if (heuristic == 5) { // adaptive-local, the paper's text
                double sigma_l = static_cast<double>(passed_1hop) / static_cast<double>(n1);
                if (sigma_l >= onehop_s_ub)
                    heuristic = 1;
                else if (sigma_l * static_cast<double>((m0 + 1) * m0) >= static_cast<double>(m0) * leniency)
                    heuristic = 3;
                else
                    heuristic = 2;
            }

            if (!visits.reserve(visits.size() + n1))
                return false;

            std::size_t collected = 0;
            auto promote = [&](compressed_slot_t slot, distance_t dist) {
                if (top.size() < top_limit || dist < radius) {
                    next.insert({-dist, slot});
                    top.insert({dist, slot}, top_limit);
                    radius = top.top().distance;
                }
                collected++;
            };

            if (heuristic == 6 || heuristic == 9) {
                // The authors' two-hop forms. `pass(ni)` reuses the semimask
                // verdicts (idx_stash holds the passing indices in ascending order).
                std::size_t pi = 0;
                auto pass = [&](std::size_t ni) {
                    bool const p = pi != passed_1hop && static_cast<std::size_t>(idx_stash[pi].slot) == ni;
                    pi += p;
                    return p;
                };
                if (heuristic == 6) {
                    std::size_t seen_passing = passed_1hop, nc = 0;
                    for (std::size_t ni = 0; ni != n1; ++ni) {
                        compressed_slot_t slot = nbrs[ni];
                        bool const p = pass(ni);
                        if (visits.test(slot))
                            continue;
                        if (p)
                            visits.set(slot);
                        distance_t dist = context.measure(query, citerator_at(slot), metric);
                        if (p)
                            promote(slot, dist);
                        onehop_buf[nc++] = {dist, slot};
                    }
                    std::sort(onehop_buf.data(), onehop_buf.data() + nc,
                              [](candidate_t const& a, candidate_t const& b) { return a.distance < b.distance; });
                    for (std::size_t ci = 0; ci != nc && seen_passing < n1; ++ci) {
                        compressed_slot_t conduit = onehop_buf[ci].slot;
                        if (visits.set(conduit))
                            continue; // passing 1-hop (already visited) are not conduits
                        neighbors_ref_t nbrs2 = neighbors_base_(node_at_(conduit));
                        if (!visits.reserve(visits.size() + nbrs2.size()))
                            return false;
                        for (std::size_t hi = 0; hi != nbrs2.size(); ++hi) {
                            compressed_slot_t slot2 = nbrs2[hi];
                            bool const p2 = context.evaluate(predicate, member_cref_t{node_at_(slot2).ckey(), slot2});
                            seen_passing += p2;
                            if (!p2 || visits.set(slot2))
                                continue;
                            promote(slot2, context.measure(query, citerator_at(slot2), metric));
                        }
                    }
                } else {
                    // Kuzu's full two-hop: passing unvisited 1-hop neighbors join. Unvisited
                    // failing 1-hop neighbors are the conduits, in adjacency order, and are
                    // marked visited when expanded.
                    std::size_t nc = 0;
                    for (std::size_t ni = 0; ni != n1; ++ni) {
                        compressed_slot_t slot = nbrs[ni];
                        bool const p = pass(ni);
                        if (visits.test(slot))
                            continue;
                        if (p) {
                            visits.set(slot);
                            promote(slot, context.measure(query, citerator_at(slot), metric));
                        } else
                            onehop_buf[nc++].slot = slot;
                    }
                    for (std::size_t ci = 0; ci != nc; ++ci) {
                        compressed_slot_t conduit = onehop_buf[ci].slot;
                        if (visits.set(conduit))
                            continue;
                        neighbors_ref_t nbrs2 = neighbors_base_(node_at_(conduit));
                        if (!visits.reserve(visits.size() + nbrs2.size()))
                            return false;
                        for (std::size_t hi = 0; hi != nbrs2.size(); ++hi) {
                            compressed_slot_t slot2 = nbrs2[hi];
                            bool const p2 = context.evaluate(predicate, member_cref_t{node_at_(slot2).ckey(), slot2});
                            if (!p2 || visits.set(slot2))
                                continue;
                            promote(slot2, context.measure(query, citerator_at(slot2), metric));
                        }
                    }
                }
                continue;
            }

            if (heuristic == 3) {
                // directed: the full 1-hop distance cost is paid up front. The distances
                // promote the passing 1-hop neighbors and also order the 2-hop conduits.
                for (std::size_t ni = 0; ni != n1; ++ni) {
                    compressed_slot_t slot = nbrs[ni];
                    onehop_buf[ni] = {context.measure(query, citerator_at(slot), metric), slot};
                }
                for (std::size_t i = 0; i != passed_1hop; ++i) {
                    std::size_t ni = static_cast<std::size_t>(idx_stash[i].slot);
                    compressed_slot_t slot = onehop_buf[ni].slot;
                    if (visits.set(slot))
                        continue;
                    promote(slot, onehop_buf[ni].distance);
                }
                std::sort(onehop_buf.data(), onehop_buf.data() + n1,
                          [](candidate_t const& a, candidate_t const& b) { return a.distance < b.distance; });
            } else {
                // onehop-s / blind: distances only for passing 1-hop neighbors.
                for (std::size_t i = 0; i != passed_1hop; ++i) {
                    compressed_slot_t slot = nbrs[static_cast<std::size_t>(idx_stash[i].slot)];
                    if (visits.set(slot))
                        continue;
                    promote(slot, context.measure(query, citerator_at(slot), metric));
                }
            }

            if (heuristic == 1)
                continue; // onehop-s: no 2-hop expansion

            // 2-hop walk through every 1-hop conduit (blind: adjacency order,
            // directed: nearest-first). It collects predicate-passing nodes until
            // the M0 cap for this expansion is reached.
            hop_seen.clear();
            for (std::size_t ni = 0; ni != n1; ++ni)
                hop_seen.set(nbrs[ni]);
            for (std::size_t ni = 0; ni != n1 && collected < m0; ++ni) {
                compressed_slot_t conduit = heuristic == 3 ? onehop_buf[ni].slot : nbrs[ni];
                neighbors_ref_t nbrs2 = neighbors_base_(node_at_(conduit));
                std::size_t const n2 = nbrs2.size();
                if (!visits.reserve(visits.size() + n2))
                    return false;
                for (std::size_t hi = 0; hi != n2 && collected < m0; ++hi) {
                    compressed_slot_t slot2 = nbrs2[hi];
                    if (hop_seen.set(slot2))
                        continue;
                    if (visits.test(slot2))
                        continue;
                    if (!context.evaluate(predicate, member_cref_t{node_at_(slot2).ckey(), slot2}))
                        continue; // failing nodes stay unvisited conduits
                    visits.set(slot2);
                    promote(slot2, context.measure(query, citerator_at(slot2), metric));
                }
            }
        }

        if (!fill_marked) { // W never filled: the whole loop was prefill
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
