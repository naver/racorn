// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 * @file bench_acorn.cpp
 * @brief Performance comparison: filtered search with/without ACORN-1
 */
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;

// ── Load fvecs file ─────────────────────────────────────────────────────────
// fvecs format: [dim(int32)] [float x dim] repeated per vector
static bool load_fvecs(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;

    // Read first dimension
    int32_t d = 0;
    if (!ifs.read(reinterpret_cast<char*>(&d), sizeof(d)))
        return false;
    if (d <= 0 || d > 100000)
        return false;
    dims = static_cast<std::size_t>(d);

    // Get file size to determine count
    ifs.seekg(0, std::ios::end);
    auto file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    std::size_t record_size = sizeof(int32_t) + dims * sizeof(float);
    count = static_cast<std::size_t>(file_size) / record_size;
    if (count == 0)
        return false;

    data.resize(count * dims);
    for (std::size_t i = 0; i < count; ++i) {
        int32_t dim_check = 0;
        ifs.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check));
        if (static_cast<std::size_t>(dim_check) != dims)
            return false;
        ifs.read(reinterpret_cast<char*>(data.data() + i * dims), dims * sizeof(float));
    }
    return ifs.good() || ifs.eof();
}

// ── Load fbin file ──────────────────────────────────────────────────────────
// fbin format: [count(int32)] [dims(int32)] [float x count*dims]
// If max_count > 0, read at most max_count vectors (avoids loading entire file).
static bool load_fbin(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims,
                      std::size_t max_count = 0) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;

    int32_t n = 0, d = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), sizeof(n)))
        return false;
    if (!ifs.read(reinterpret_cast<char*>(&d), sizeof(d)))
        return false;
    if (n <= 0 || d <= 0 || d > 100000)
        return false;

    count = static_cast<std::size_t>(n);
    dims = static_cast<std::size_t>(d);
    if (max_count > 0 && count > max_count)
        count = max_count;
    data.resize(count * dims);
    ifs.read(reinterpret_cast<char*>(data.data()), count * dims * sizeof(float));
    return ifs.good() || ifs.eof();
}

// ── Vector store ────────────────────────────────────────────────────────────
template <typename T = float> class vector_store_t {
  public:
    vector_store_t(std::vector<T>&& data, std::size_t dims) : dims_(dims), data_(std::move(data)) {}

    std::size_t size() const noexcept { return data_.size() / dims_; }
    std::size_t dims() const noexcept { return dims_; }
    T const* row(std::size_t i) const noexcept { return data_.data() + i * dims_; }

    // L2-normalize all vectors in-place (unit length)
    void normalize_l2() {
        std::size_t n = size();
        for (std::size_t i = 0; i < n; ++i) {
            T* v = data_.data() + i * dims_;
            T norm_sq = 0;
            for (std::size_t d = 0; d < dims_; ++d)
                norm_sq += v[d] * v[d];
            T norm = std::sqrt(norm_sq);
            if (norm > static_cast<T>(0)) {
                T inv = static_cast<T>(1) / norm;
                for (std::size_t d = 0; d < dims_; ++d)
                    v[d] *= inv;
            }
        }
    }

  private:
    std::size_t dims_;
    std::vector<T> data_;
};

// ── L2 squared metric ───────────────────────────────────────────────────────
template <typename T = float, typename key_t = std::size_t, typename slot_t = std::size_t> struct l2_metric_t {
    using index_typed_t = index_gt<T, key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    vector_store_t<T> const* store = nullptr;
    std::size_t dimensions = 0;

    T const* row(std::size_t s) const noexcept { return store->row(s); }

    float operator()(member_cref_t const& a, member_cref_t const& b) const {
        return metric_l2sq_gt<T, float>{}(row(get_key(b)), row(get_key(a)), dimensions);
    }
    float operator()(T const* q, member_cref_t const& m) const {
        return metric_l2sq_gt<T, float>{}(q, row(get_key(m)), dimensions);
    }
    float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
        return metric_l2sq_gt<T, float>{}(row(get_key(*b)), row(get_key(*a)), dimensions);
    }
    float operator()(T const* q, member_citerator_t const& m) const {
        return metric_l2sq_gt<T, float>{}(q, row(get_key(*m)), dimensions);
    }
};

// ── Cosine metric ───────────────────────────────────────────────────────────
template <typename T = float, typename key_t = std::size_t, typename slot_t = std::size_t> struct cosine_metric_t {
    using index_typed_t = index_gt<T, key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;

    vector_store_t<T> const* store = nullptr;
    std::size_t dimensions = 0;

    T const* row(std::size_t s) const noexcept { return store->row(s); }

    float operator()(member_cref_t const& a, member_cref_t const& b) const {
        return metric_cos_gt<T, float>{}(row(get_key(b)), row(get_key(a)), dimensions);
    }
    float operator()(T const* q, member_cref_t const& m) const {
        return metric_cos_gt<T, float>{}(q, row(get_key(m)), dimensions);
    }
    float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
        return metric_cos_gt<T, float>{}(row(get_key(*b)), row(get_key(*a)), dimensions);
    }
    float operator()(T const* q, member_citerator_t const& m) const {
        return metric_cos_gt<T, float>{}(q, row(get_key(*m)), dimensions);
    }
};

// ── Aligned allocator wrapper ───────────────────────────────────────────────
template <typename index_at> struct aligned_index_t {
    using index_t = index_at;
    using alloc_t = aligned_allocator_gt<index_t, 64>;
    index_t* ptr = nullptr;

    template <typename... A> explicit aligned_index_t(A&&... args) {
        ptr = alloc_t{}.allocate(1);
        new (ptr) index_t(std::forward<A>(args)...);
    }
    ~aligned_index_t() {
        if (ptr) {
            ptr->~index_t();
            alloc_t{}.deallocate(ptr, 1);
        }
    }
    aligned_index_t(aligned_index_t const&) = delete;
    aligned_index_t& operator=(aligned_index_t const&) = delete;
};

// ── Timer utility ───────────────────────────────────────────────────────────
struct stopwatch_t {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    stopwatch_t() : start(clock::now()) {}
    double elapsed_ms() const { return std::chrono::duration<double, std::milli>(clock::now() - start).count(); }
};

// ── Constants / Defaults ────────────────────────────────────────────────────
static constexpr std::size_t SIFT_N_MAX = 1'000'000;
static constexpr std::size_t GIST_N_MAX = 1'000'000;
static constexpr std::size_t TEXT2IMAGE_N_MAX = 40'000'000;
static double ACORN_THRESHOLD = 0.3;

// Tuneable defaults (overridable via command-line)
static std::size_t N_VAL = 1'000'000;
static std::size_t D_VAL = 128; // set at runtime based on dataset
static std::size_t M_VAL = 16;
static std::size_t K_VAL = 100;
static std::size_t EF_CONSTRUCTION_VAL = 100;
static std::size_t EF_SEARCH_VAL = 200;
static std::string METRIC_NAME = "";             // "l2" or "cosine" (empty = auto per dataset)
static std::string DATASET_NAME = "sift";        // "sift" or "text2image"
static double BRIDGE_BUDGET_RATIO = 1.0;         // RACORN-1 bridge budget ratio
static double THREE_HOP_THRESHOLD = -1.0;        // 3-hop expansion threshold (-1 = disabled)
static double FALLBACK_EXACT_THRESHOLD = 0.003;  // RACORN-1+ exact-fallback threshold (0 = disabled)
static bool RACORN_STRIDE_SAMPLING = true;       // RACORN-1 stride sampling toggle (false → ablation off)

// ── Cluster correlation (ACORN paper-style query/filter correlation) ──────
//   positive  → predicate matches when node.cluster_id == query.cluster_id
//               (selectivity ~ 1/K, query-correlated).
//   negative  → predicate matches when node.cluster_id == TARGET[q], where
//               TARGET[q] is one cluster id chosen at random from
//               {0..K-1} \ {query.cluster_id} per query, deterministically
//               seeded. Selectivity ~ 1/K (same magnitude as positive) but
//               uncorrelated with the query — i.e. the filter looks like a
//               1/K random partition unrelated to the query's geometry.
enum class cluster_correlation_t { none, positive, negative };
static cluster_correlation_t CLUSTER_CORRELATION = cluster_correlation_t::none;
static std::size_t CLUSTER_K = 100;                        // K used for k-means cluster files
static std::string CLUSTER_DIR = "data/clusters";          // cluster file directory
static std::vector<uint32_t> NODE_CLUSTERS;                // base[i] -> cluster id (loaded once)
static std::vector<uint32_t> QUERY_CLUSTERS;               // queries[i] -> cluster id (loaded once)
static std::vector<uint32_t> NEGATIVE_TARGET_CLUSTERS;     // queries[i] -> random non-self cluster id (negative mode)
static std::uint64_t NEGATIVE_TARGET_SEED = 42;            // deterministic per-query target seed
static std::size_t NUM_QUERIES = 100;            // number of benchmark queries
static std::size_t SEARCH_THREADS = 1;           // parallel search threads
static bool SEARCH_THREADS_SET = false;          // user explicitly passed --search-threads
static std::string DATA_DIR = "data/raw";        // directory for raw dataset files
static std::string VOLUME_DIR = "data/volumes";  // directory for index volume files
static std::string GT_DIR = "data/gt";           // directory for ground truth files
static bool BUILD_INDEX_ONLY = false;            // --build-index mode
static bool BUILD_GT_ONLY = false;               // --build-gt mode
static bool NUM_QUERIES_SET = false;             // user explicitly passed --nq
static std::size_t BUILD_GT_K = 0;               // --build-gt-k: GT results per query (0 = use K)
static std::vector<double> FILTER_SELECTIVITIES; // if non-empty, run only these selectivities (detail mode)
static std::size_t BENCH_WARMUP = 3;             // warmup runs before measurement
static std::size_t BENCH_RUNS = 10;              // measurement runs
static bool QPS_MODE = false;                    // --qps mode: sweep ef_search, report recall & QPS
static bool SKIP_EXACT = false;                  // --skip-exact: skip exact latency benchmarking
static std::vector<std::string> FILTER_MODES;    // --modes: only run these modes (empty = all)
static std::vector<std::size_t> QPS_EF_VALUES = {128, 256, 384, 512};

// Case-insensitive string compare
static bool iequals(std::string const& a, std::string const& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i])))
            return false;
    return true;
}

static bool mode_matches(char const* mode_name) {
    if (FILTER_MODES.empty())
        return true;
    for (auto const& m : FILTER_MODES)
        if (iequals(m, mode_name))
            return true;
    return false;
}

// Keep old names as macros pointing to the variables for minimal code change
#define N N_VAL
#define D D_VAL
#define M M_VAL
#define K K_VAL
#define EF_CONSTRUCTION EF_CONSTRUCTION_VAL
#define EF_SEARCH EF_SEARCH_VAL

using vec_key_t = std::size_t;
using slot_t = std::size_t;
using index_t = index_gt<float, vec_key_t, slot_t>;

// ── Selectivity-based predicate ─────────────────────────────────────────────
// Passes if key % modulus < threshold. e.g. modulus=10,threshold=1 => 10%, modulus=10,threshold=3 => 30%
struct selectivity_predicate_t {
    std::size_t modulus;
    std::size_t threshold;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        return (get_key(m) % modulus) < threshold;
    }
};

// ── Cluster correlation predicate ──────────────────────────────────────────
// Always an equality test against a per-query target cluster id. The caller
// chooses the target:
//   * positive correlation → target = query's own cluster id (QUERY_CLUSTERS)
//   * negative correlation → target = random non-self cluster id chosen per
//                            query at startup (NEGATIVE_TARGET_CLUSTERS)
// Both modes therefore see the same selectivity scale (~1/K); the difference
// is whether the surviving cluster is the query's natural one or an arbitrary
// other cluster.
struct cluster_predicate_t {
    uint32_t const* node_clusters;
    uint32_t target_cluster_id;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        return node_clusters[get_key(m)] == target_cluster_id;
    }
};

// Cluster file format:
//   bytes  0..3  magic "CL01"
//   bytes  4..7  count (uint32)
//   bytes  8..11 K (uint32)
//   bytes 12..15 reserved
//   bytes 16..   cluster IDs (uint32 × count)
// `exact_count` selects validation mode:
//   true  → file count must equal `expected_min_count` (used for base file)
//   false → file count must be ≥ `expected_min_count` (used for query file —
//                                                       q_max-built file with
//                                                       smaller bench nq).
static bool load_cluster_file(std::string const& path, std::vector<uint32_t>& out, std::size_t expected_min_count,
                              std::size_t expected_k, bool exact_count = true) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Cluster file not found: " << path << "\n"
                  << "  Build with: python cpp/scripts/build_clusters.py "
                  << "--dataset <name> --n <N> --k <K>\n";
        return false;
    }
    char magic[4] = {};
    uint32_t count = 0, k = 0, reserved = 0;
    ifs.read(magic, 4);
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
    ifs.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    if (!ifs || std::string(magic, 4) != "CL01") {
        std::cerr << "Bad cluster file (magic / header): " << path << "\n";
        return false;
    }
    bool count_ok = exact_count ? (count == expected_min_count) : (count >= expected_min_count);
    if (!count_ok) {
        std::cerr << "Cluster file count mismatch: expected " << (exact_count ? "" : ">=") << expected_min_count
                  << " got " << count << " (" << path << ")\n";
        return false;
    }
    if (k != expected_k) {
        std::cerr << "Cluster file K mismatch: expected " << expected_k << " got " << k << " (" << path << ")\n";
        return false;
    }
    out.resize(count);
    ifs.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count) * sizeof(uint32_t));
    return ifs.good() || ifs.eof();
}

// ── Negative-correlation target picker ─────────────────────────────────────
// For each query, picks a random cluster id from {0..K-1} \ {query's own},
// using a deterministic per-query seed so a rerun with the same seed produces
// the same targets (and so the value at index q is independent of --nq).
static bool populate_negative_targets(std::vector<uint32_t> const& query_clusters, std::size_t k_clusters,
                                      std::uint64_t seed, std::vector<uint32_t>& out) {
    if (k_clusters < 2) {
        std::cerr << "negative correlation requires --cluster-k >= 2 (got " << k_clusters << ")\n";
        return false;
    }
    out.resize(query_clusters.size());
    for (std::size_t qi = 0; qi < query_clusters.size(); ++qi) {
        std::mt19937_64 rng(seed + static_cast<std::uint64_t>(qi));
        std::uniform_int_distribution<std::uint32_t> dist(0, static_cast<std::uint32_t>(k_clusters - 2));
        std::uint32_t my = query_clusters[qi];
        std::uint32_t r = dist(rng);
        // map r ∈ [0, K-2] onto [0, K-1] \ {my} by skipping `my`
        out[qi] = (my < k_clusters && r >= my) ? r + 1 : r;
    }
    return true;
}

// ── Cluster-mode selectivity statistics ─────────────────────────────────────
// Used by cluster-correlation experiments. Each query's "selectivity" is the
// fraction of base nodes whose cluster id equals that query's target cluster
// (= query's own cluster id in positive mode, a random non-self cluster in
// negative mode). We report mean and p50/p90/p95/p99 to give a feel for the
// per-query distribution (cluster sizes vary with K).
struct cluster_sel_stats_t {
    double mean = 0.0;
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

static cluster_sel_stats_t compute_cluster_selectivity_stats(std::vector<uint32_t> const& node_clusters,
                                                             std::vector<uint32_t> const& target_clusters,
                                                             std::size_t k_clusters, std::size_t n_base) {
    cluster_sel_stats_t out;
    if (target_clusters.empty() || n_base == 0)
        return out;
    std::vector<std::size_t> sizes(k_clusters, 0);
    for (uint32_t cid : node_clusters)
        if (cid < k_clusters)
            ++sizes[cid];
    std::vector<double> per_q;
    per_q.reserve(target_clusters.size());
    for (uint32_t tc : target_clusters) {
        std::size_t match = (tc < k_clusters) ? sizes[tc] : 0;
        per_q.push_back(static_cast<double>(match) / static_cast<double>(n_base));
    }
    double sum = 0.0;
    for (double s : per_q)
        sum += s;
    out.mean = sum / static_cast<double>(per_q.size());
    std::vector<double> sorted = per_q;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) -> double {
        std::size_t idx = static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1) + 0.5);
        if (idx >= sorted.size())
            idx = sorted.size() - 1;
        return sorted[idx];
    };
    out.p50 = pct(0.50);
    out.p90 = pct(0.90);
    out.p95 = pct(0.95);
    out.p99 = pct(0.99);
    out.max = sorted.back();
    return out;
}

// ── Compute recall against ground truth ─────────────────────────────────────
template <typename result_at>
double compute_recall(result_at const& result, std::vector<vec_key_t> const& gt, std::size_t k) {
    std::size_t limit = std::min({gt.size(), k});
    std::unordered_set<vec_key_t> gt_set(gt.begin(), gt.begin() + limit);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < result.size() && i < k; ++i)
        if (gt_set.count(result[i].member.key))
            ++hits;
    return limit > 0 ? static_cast<double>(hits) / static_cast<double>(limit) : 0.0;
}

// ── Search result summary ───────────────────────────────────────────────────
struct search_result_summary_t {
    std::string label;
    double selectivity;
    bool acorn;
    bool exact;
    double latency_ms;
    double dist_computed;
    double nodes_visited;
    double recall;
    double predicate_evals; // ACORN-1 / RACORN-1 / 3-hop only; 0 otherwise
    double fallback_count;
    double fallback_selectivity;
    bool fallback_to_exact; // RACORN-1+ exact-fallback fired for any query
};

// ── Median helper ───────────────────────────────────────────────────────────
// Strips NaN entries first so callers can pass per-query stats that include
// NaN sentinels (e.g. fallback_selectivity for queries that never entered the
// racorn fallback path) without corrupting nth_element's strict-weak-ordering
// requirement. Returns NaN when every entry is NaN (i.e. nothing to summarize).
static double median(std::vector<double>& v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
    if (v.empty())
        return std::nan("");
    std::size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 1)
        return v[n / 2];
    double upper = v[n / 2];
    auto it = std::max_element(v.begin(), v.begin() + n / 2);
    return (*it + upper) / 2.0;
}

// ── Percentile helper (expects pre-sorted input) ────────────────────────────
static double percentile_sorted(std::vector<double> const& sorted_v, double p) {
    if (sorted_v.empty())
        return 0.0;
    if (p <= 0.0)
        return sorted_v.front();
    if (p >= 1.0)
        return sorted_v.back();
    double idx = p * static_cast<double>(sorted_v.size() - 1);
    std::size_t lo = static_cast<std::size_t>(idx);
    std::size_t hi = lo + 1;
    if (hi >= sorted_v.size())
        return sorted_v.back();
    double frac = idx - static_cast<double>(lo);
    return sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac;
}

// ── Persistent worker pool ──────────────────────────────────────────────────
// Pre-spawns `st` worker threads that wait on a condition variable for each
// new batch. Each batch dispatches `fn(q, tid)` over query indices
// [chunk*tid, chunk*tid+chunk) for tid in [0, st). Replaces the previous
// per-iteration spawn/join — the spawn cost was visible in QPS-mode
// wall-clock (timer wraps the entire batch), and per-batch CPU placement
// was inconsistent because the OS picks CPUs anew at every spawn.
class query_pool_t {
  public:
    query_pool_t(std::size_t st, std::size_t nq) : st_(st), nq_(nq), chunk_size_((nq + st - 1) / st) {
        if (st_ <= 1)
            return; // serial path; no workers to spin up
        workers_.reserve(st_);
        for (std::size_t t = 0; t < st_; ++t)
            workers_.emplace_back([this, t]() { worker_loop(t); });
    }

    ~query_pool_t() {
        if (st_ <= 1)
            return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++generation_;
        }
        work_cv_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

    query_pool_t(query_pool_t const&) = delete;
    query_pool_t& operator=(query_pool_t const&) = delete;

    // Run `fn(q, tid)` for q in [0, nq) across `st` workers in parallel.
    // Blocks until all workers complete this batch.
    template <typename Fn> void run(Fn&& fn) {
        if (st_ <= 1) {
            for (std::size_t q = 0; q < nq_; ++q)
                fn(q, std::size_t(0));
            return;
        }
        std::function<void(std::size_t, std::size_t)> task = std::forward<Fn>(fn);
        {
            std::lock_guard<std::mutex> lk(mu_);
            task_ = std::move(task);
            pending_ = st_;
            ++generation_;
        }
        work_cv_.notify_all();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this]() { return pending_ == 0; });
    }

  private:
    void worker_loop(std::size_t t) {
        std::size_t qbegin = t * chunk_size_;
        std::size_t qend = std::min(qbegin + chunk_size_, nq_);
        std::size_t my_gen = 0;
        std::function<void(std::size_t, std::size_t)> local_task;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                work_cv_.wait(lk, [this, my_gen]() { return stop_ || generation_ > my_gen; });
                if (stop_)
                    return;
                my_gen = generation_;
                local_task = task_;
            }
            for (std::size_t q = qbegin; q < qend; ++q)
                local_task(q, t);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--pending_ == 0)
                    done_cv_.notify_one();
            }
        }
    }

    std::size_t const st_;
    std::size_t const nq_;
    std::size_t const chunk_size_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::function<void(std::size_t, std::size_t)> task_;
    std::size_t generation_ = 0;
    std::size_t pending_ = 0;
    bool stop_ = false;
};

// ── Run a single search test (multi-query, median aggregation) ──────────────
// `make_predicate(query_idx)` is invoked per query so cluster-correlation
// mode can swap in a query-specific predicate. For modulo predicates the
// caller's factory just returns the same fixed predicate object every call.
template <typename metric_at, typename make_predicate_at>
search_result_summary_t run_search(index_t& index, metric_at& metric, std::vector<float const*> const& queries,
                                   std::size_t k, std::size_t ef_search, make_predicate_at&& make_predicate,
                                   double two_hop_threshold, bool racorn, bool is_exact, char const* label,
                                   double selectivity, bool is_acorn, double fallback_exact_threshold,
                                   std::vector<std::vector<vec_key_t>> const& ground_truths) {

    index_search_config_t cfg;
    cfg.expansion = ef_search;
    cfg.two_hop_threshold = two_hop_threshold;
    cfg.racorn = racorn;
    cfg.bridge_budget_ratio = BRIDGE_BUDGET_RATIO;
    cfg.three_hop_threshold = THREE_HOP_THRESHOLD;
    cfg.fallback_exact_threshold = fallback_exact_threshold;
    cfg.racorn_stride_sampling = RACORN_STRIDE_SAMPLING;
    cfg.exact = is_exact;

    std::size_t nq = queries.size();
    std::size_t st = std::min(SEARCH_THREADS, nq);
    int WARMUP = static_cast<int>(BENCH_WARMUP);
    int RUNS = static_cast<int>(BENCH_RUNS);

    // Persistent worker pool — spawned once per run_search, reused across all
    // WARMUP/RUNS iterations. Avoids per-iteration spawn/join overhead and
    // keeps thread/CPU placement more stable between iterations.
    query_pool_t pool(st, nq);
    auto parallel_queries = [&](auto&& fn) { pool.run(std::forward<decltype(fn)>(fn)); };

    for (int i = 0; i < WARMUP; ++i) {
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            auto r = index.search(queries[q], k, metric, c, pred);
            (void)r;
        });
    }

    // Per-query accumulators: for each query, collect metrics across RUNS, then take median per query
    // Final reported value = median across queries
    std::vector<std::vector<double>> per_q_ms(nq);
    std::vector<std::vector<double>> per_q_distances(nq);
    std::vector<std::vector<double>> per_q_visited(nq);
    std::vector<std::vector<double>> per_q_recall(nq);
    std::vector<std::vector<double>> per_q_result_count(nq);
    std::vector<std::vector<double>> per_q_fb_count(nq);
    std::vector<std::vector<double>> per_q_fb_selectivity(nq);
    std::vector<std::vector<double>> per_q_pred_evals(nq);
    std::atomic<bool> any_fallback_to_exact{false};

    for (std::size_t q = 0; q < nq; ++q) {
        per_q_ms[q].reserve(RUNS);
        per_q_distances[q].reserve(RUNS);
        per_q_visited[q].reserve(RUNS);
        per_q_recall[q].reserve(RUNS);
        per_q_result_count[q].reserve(RUNS);
        per_q_fb_count[q].reserve(RUNS);
        per_q_fb_selectivity[q].reserve(RUNS);
        per_q_pred_evals[q].reserve(RUNS);
    }

    for (int i = 0; i < RUNS; ++i) {
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            stopwatch_t t;
            auto r = index.search(queries[q], k, metric, c, pred);
            per_q_ms[q].push_back(t.elapsed_ms());
            per_q_result_count[q].push_back(static_cast<double>(r.size()));
            per_q_distances[q].push_back(static_cast<double>(r.computed_distances));
            per_q_visited[q].push_back(static_cast<double>(r.visited_members));
            per_q_fb_count[q].push_back(static_cast<double>(r.fallback_count));
            per_q_fb_selectivity[q].push_back(r.fallback_selectivity);
            per_q_pred_evals[q].push_back(static_cast<double>(r.predicate_evaluations));
            if (r.fallback_to_exact)
                any_fallback_to_exact.store(true, std::memory_order_relaxed);
            per_q_recall[q].push_back(r ? compute_recall(r, ground_truths[q], k) : 0.0);
        });
    }

    // For each query, compute its median across RUNS, then take median of those across queries
    std::vector<double> q_medians_ms(nq), q_medians_dist(nq), q_medians_visit(nq);
    std::vector<double> q_medians_recall(nq), q_medians_result(nq);
    std::vector<double> q_medians_fb_count(nq), q_medians_fb_sel(nq);
    std::vector<double> q_medians_pred_evals(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        q_medians_ms[q] = median(per_q_ms[q]);
        q_medians_dist[q] = median(per_q_distances[q]);
        q_medians_visit[q] = median(per_q_visited[q]);
        q_medians_recall[q] = median(per_q_recall[q]);
        q_medians_result[q] = median(per_q_result_count[q]);
        q_medians_fb_count[q] = median(per_q_fb_count[q]);
        q_medians_fb_sel[q] = median(per_q_fb_selectivity[q]);
        q_medians_pred_evals[q] = median(per_q_pred_evals[q]);
    }

    double med_ms = median(q_medians_ms);
    double med_distances = median(q_medians_dist);
    double med_visited = median(q_medians_visit);
    double med_result_count = median(q_medians_result);
    double med_fb_count = median(q_medians_fb_count);
    double med_fb_selectivity = median(q_medians_fb_sel);
    double med_pred_evals = median(q_medians_pred_evals);

    // Recall: compute avg and percentiles (sort once)
    double avg_recall = 0.0;
    for (double r : q_medians_recall)
        avg_recall += r;
    avg_recall /= static_cast<double>(q_medians_recall.size());
    std::vector<double> recall_sorted(q_medians_recall);
    std::sort(recall_sorted.begin(), recall_sorted.end());
    double p50_recall = percentile_sorted(recall_sorted, 0.50);
    double p90_recall = percentile_sorted(recall_sorted, 0.10); // 90% of queries >= this
    double p95_recall = percentile_sorted(recall_sorted, 0.05); // 95% of queries >= this
    double p99_recall = percentile_sorted(recall_sorted, 0.01); // 99% of queries >= this

    std::cout << "\n[" << label << "]\n";
    std::cout << "  Queries             : " << nq << " x " << RUNS << " runs (" << WARMUP << " warmup, " << st
              << " threads)\n";
    std::cout << "  Med results returned: " << std::fixed << std::setprecision(1) << med_result_count << " / " << k
              << "\n";
    std::cout << "  Recall@" << k << std::string(13 - std::to_string(k).size(), ' ') << ": "
              << "avg=" << std::fixed << std::setprecision(4) << avg_recall << "  p50=" << p50_recall
              << "  p90=" << p90_recall << "  p95=" << p95_recall << "  p99=" << p99_recall << "\n";
    std::cout << "  Med latency         : " << std::fixed << std::setprecision(3) << med_ms << " ms\n";
    std::cout << "  Med dist. computed  : " << std::fixed << std::setprecision(0) << med_distances << "\n";
    std::cout << "  Med nodes visited   : " << std::fixed << std::setprecision(0) << med_visited << "\n";
    std::cout << "  Med dist/visit      : " << std::fixed << std::setprecision(1)
              << (med_visited > 0 ? med_distances / med_visited : 0) << "\n";
    if (med_fb_count > 0) {
        std::cout << "  Med fallback count  : " << std::fixed << std::setprecision(1) << med_fb_count << "\n";
        std::cout << "  Fallback selectivity: " << std::fixed << std::setprecision(4) << med_fb_selectivity << "\n";
    }
    if (med_pred_evals > 0)
        std::cout << "  Med predicate evals : " << std::fixed << std::setprecision(0) << med_pred_evals << "\n";
    bool fb_to_exact = any_fallback_to_exact.load(std::memory_order_relaxed);
    if (fb_to_exact)
        std::cout << "  ** Some queries switched to exact search (RACORN-1+) **\n";

    return {label,      selectivity,    is_acorn,     is_exact,           med_ms,     med_distances, med_visited,
            avg_recall, med_pred_evals, med_fb_count, med_fb_selectivity, fb_to_exact};
}

// ── QPS search result ───────────────────────────────────────────────────────
struct qps_result_t {
    double recall;
    double qps;
    double latency_ms;
};

// ── Run a single QPS measurement (all queries batched, median QPS) ──────────
// `make_predicate(query_idx)` is invoked per query so cluster-correlation
// mode can swap in a query-specific predicate. For modulo predicates the
// caller's factory just returns the same fixed predicate object every call.
template <typename metric_at, typename make_predicate_at>
qps_result_t run_qps_search(index_t& index, metric_at& metric, std::vector<float const*> const& queries, std::size_t k,
                            std::size_t ef_search, make_predicate_at&& make_predicate, double two_hop_threshold,
                            bool racorn, double fallback_exact_threshold,
                            std::vector<std::vector<vec_key_t>> const& ground_truths) {

    index_search_config_t cfg;
    cfg.expansion = ef_search;
    cfg.two_hop_threshold = two_hop_threshold;
    cfg.racorn = racorn;
    cfg.bridge_budget_ratio = BRIDGE_BUDGET_RATIO;
    cfg.three_hop_threshold = THREE_HOP_THRESHOLD;
    cfg.fallback_exact_threshold = fallback_exact_threshold;
    cfg.racorn_stride_sampling = RACORN_STRIDE_SAMPLING;
    cfg.exact = false;

    std::size_t nq = queries.size();
    std::size_t st = std::min(SEARCH_THREADS, nq);

    // Persistent worker pool — spawned once per run_qps_search, reused across
    // warmup, the recall pass, and all RUNS measurement batches. Critical for
    // QPS mode where the timer wraps the whole batch and previously included
    // per-iteration thread spawn/join cost.
    query_pool_t pool(st, nq);
    auto parallel_queries = [&](auto&& fn) { pool.run(std::forward<decltype(fn)>(fn)); };

    // Warmup
    for (std::size_t i = 0; i < BENCH_WARMUP; ++i) {
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            auto r = index.search(queries[q], k, metric, c, pred);
            (void)r;
        });
    }

    // Recall: single pass
    std::vector<double> recalls(nq);
    parallel_queries([&](std::size_t q, std::size_t tid) {
        auto c = cfg;
        c.thread = tid;
        auto pred = make_predicate(q);
        auto r = index.search(queries[q], k, metric, c, pred);
        recalls[q] = r ? compute_recall(r, ground_truths[q], k) : 0.0;
    });
    double avg_recall = 0.0;
    for (double r : recalls)
        avg_recall += r;
    avg_recall /= static_cast<double>(nq);

    // QPS & latency: batch all queries, measure wall time, repeat BENCH_RUNS times
    std::vector<double> qps_values;
    std::vector<double> latency_values;
    qps_values.reserve(BENCH_RUNS);
    latency_values.reserve(BENCH_RUNS);
    for (std::size_t run = 0; run < BENCH_RUNS; ++run) {
        stopwatch_t timer;
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            auto r = index.search(queries[q], k, metric, c, pred);
            (void)r;
        });
        double elapsed_ms = timer.elapsed_ms();
        double elapsed_s = elapsed_ms / 1000.0;
        qps_values.push_back(static_cast<double>(nq) / elapsed_s);
        latency_values.push_back(elapsed_ms * static_cast<double>(st) / static_cast<double>(nq));
    }
    double med_qps = median(qps_values);
    double med_latency = median(latency_values);

    return {avg_recall, med_qps, med_latency};
}

// ── Compute exact ground truth for a given predicate ────────────────────────
template <typename metric_at, typename predicate_at>
std::vector<vec_key_t> compute_ground_truth(index_t& index, metric_at& metric, float const* query, std::size_t k,
                                            predicate_at&& predicate, std::size_t thread_id = 0) {
    index_search_config_t exact_cfg;
    exact_cfg.exact = true;
    exact_cfg.thread = thread_id;
    auto result = index.search(query, k, metric, exact_cfg, predicate);
    std::vector<vec_key_t> gt;
    gt.reserve(result.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        gt.push_back(result[i].member.key);
    return gt;
}

// ── Index volume file name ───────────────────────────────────────────────────
static std::string index_volume_path() {
    std::ostringstream oss;
    oss << VOLUME_DIR << "/" << DATASET_NAME << "_" << N << "_" << D << "_m" << M << "_ef" << EF_CONSTRUCTION
        << ".usearch";
    return oss.str();
}

// ── Cluster file paths ──────────────────────────────────────────────────────
static std::string cluster_base_path() {
    std::ostringstream oss;
    oss << CLUSTER_DIR << "/cluster_" << DATASET_NAME << "_n" << N << "_k" << CLUSTER_K << ".bin";
    return oss.str();
}

static std::string cluster_query_path() {
    std::ostringstream oss;
    // Query cluster IDs are tied to the base centroids learned at N vectors,
    // so the filename must include N — otherwise builds of the same dataset
    // at different N (e.g. text2image 1m vs 40m) would clobber each other's
    // query files and the predicate would compare query/base cluster IDs
    // from two different centroid spaces.
    oss << CLUSTER_DIR << "/cluster_" << DATASET_NAME << "_query_n" << N << "_k" << CLUSTER_K << ".bin";
    return oss.str();
}

// Suffix appended to GT filenames when running in cluster-correlation mode so
// the file disambiguates from the modulo-based GT for the same (dataset, N, K, nq).
static std::string cluster_gt_suffix() {
    if (CLUSTER_CORRELATION == cluster_correlation_t::none)
        return "";
    std::ostringstream oss;
    oss << "_cluster_" << (CLUSTER_CORRELATION == cluster_correlation_t::positive ? "pos" : "neg") << "_k" << CLUSTER_K;
    return oss.str();
}

// ── Ground truth file name ───────────────────────────────────────────────────
static std::string gt_file_path(std::size_t gt_k, std::size_t nq) {
    std::ostringstream oss;
    oss << GT_DIR << "/" << DATASET_NAME << "_" << N << "_" << D << "_m" << M << "_ef" << EF_CONSTRUCTION << "_gt_k"
        << gt_k << "_nq" << nq << cluster_gt_suffix() << ".bin";
    return oss.str();
}

// ── Parse command-line arguments ────────────────────────────────────────────
static void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_val = [&]() -> std::size_t {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            return static_cast<std::size_t>(std::stoull(argv[++i]));
        };
        if (arg == "--ef-search")
            EF_SEARCH_VAL = next_val();
        else if (arg == "--m")
            M_VAL = next_val();
        else if (arg == "--ef-construction")
            EF_CONSTRUCTION_VAL = next_val();
        else if (arg == "--k")
            K_VAL = next_val();
        else if (arg == "--n") {
            N_VAL = next_val();
            // validation deferred until after --dataset is parsed
        } else if (arg == "--metric") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --metric\n";
                std::exit(1);
            }
            METRIC_NAME = argv[++i];
            if (METRIC_NAME != "l2" && METRIC_NAME != "cosine") {
                std::cerr << "Unknown metric: " << METRIC_NAME << " (use 'l2' or 'cosine')\n";
                std::exit(1);
            }
        } else if (arg == "--dataset") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --dataset\n";
                std::exit(1);
            }
            DATASET_NAME = argv[++i];
            if (DATASET_NAME != "sift" && DATASET_NAME != "gist" && DATASET_NAME != "text2image") {
                std::cerr << "Unknown dataset: " << DATASET_NAME << " (use 'sift', 'gist', or 'text2image')\n";
                std::exit(1);
            }
        } else if (arg == "--nq") {
            NUM_QUERIES = next_val();
            if (NUM_QUERIES == 0)
                NUM_QUERIES = 1;
            NUM_QUERIES_SET = true;
        } else if (arg == "--search-threads") {
            SEARCH_THREADS = next_val();
            if (SEARCH_THREADS == 0)
                SEARCH_THREADS = 1;
            SEARCH_THREADS_SET = true;
        } else if (arg == "--2hop-threshold") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --2hop-threshold\n";
                std::exit(1);
            }
            ACORN_THRESHOLD = std::stod(argv[++i]);
        } else if (arg == "--bridge-budget-ratio") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --bridge-budget-ratio\n";
                std::exit(1);
            }
            BRIDGE_BUDGET_RATIO = std::stod(argv[++i]);
            if (BRIDGE_BUDGET_RATIO < 0.0) {
                std::cerr << "--bridge-budget-ratio must be >= 0\n";
                std::exit(1);
            }
        } else if (arg == "--fallback-threshold") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --fallback-threshold\n";
                std::exit(1);
            }
            FALLBACK_EXACT_THRESHOLD = std::stod(argv[++i]);
            if (FALLBACK_EXACT_THRESHOLD < 0.0) {
                std::cerr << "--fallback-threshold must be >= 0\n";
                std::exit(1);
            }
        } else if (arg == "--cluster-correlation") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --cluster-correlation (none|positive|negative)\n";
                std::exit(1);
            }
            std::string v = argv[++i];
            if (iequals(v, "none")) CLUSTER_CORRELATION = cluster_correlation_t::none;
            else if (iequals(v, "positive") || iequals(v, "pos")) CLUSTER_CORRELATION = cluster_correlation_t::positive;
            else if (iequals(v, "negative") || iequals(v, "neg")) CLUSTER_CORRELATION = cluster_correlation_t::negative;
            else {
                std::cerr << "--cluster-correlation expects none|positive|negative (got: " << v << ")\n";
                std::exit(1);
            }
        } else if (arg == "--cluster-k") {
            CLUSTER_K = next_val();
            if (CLUSTER_K == 0) {
                std::cerr << "--cluster-k must be > 0\n";
                std::exit(1);
            }
        } else if (arg == "--cluster-dir") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --cluster-dir\n";
                std::exit(1);
            }
            CLUSTER_DIR = argv[++i];
        } else if (arg == "--racorn-stride-sampling") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --racorn-stride-sampling (on|off)\n";
                std::exit(1);
            }
            std::string v = argv[++i];
            if (iequals(v, "on") || iequals(v, "true") || v == "1") {
                RACORN_STRIDE_SAMPLING = true;
            } else if (iequals(v, "off") || iequals(v, "false") || v == "0") {
                RACORN_STRIDE_SAMPLING = false;
            } else {
                std::cerr << "--racorn-stride-sampling expects on/off (got: " << v << ")\n";
                std::exit(1);
            }
        } else if (arg == "--3hop-threshold") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --3hop-threshold\n";
                std::exit(1);
            }
            THREE_HOP_THRESHOLD = std::stod(argv[++i]);
#if !USEARCH_ENABLE_3HOP
            if (THREE_HOP_THRESHOLD >= 0) {
                std::cerr << "Warning: --3hop-threshold=" << THREE_HOP_THRESHOLD
                          << " ignored — this binary was built without USEARCH_ENABLE_3HOP=1.\n"
                          << "         Use `bench_acorn_3hop` to enable 3-hop expansion.\n";
            }
#endif
        } else if (arg == "--runs") {
            BENCH_RUNS = next_val();
            if (BENCH_RUNS == 0)
                BENCH_RUNS = 1;
        } else if (arg == "--warmup") {
            BENCH_WARMUP = next_val();
        } else if (arg == "--selectivity") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --selectivity\n";
                std::exit(1);
            }
            std::string sel_list = argv[++i];
            FILTER_SELECTIVITIES.clear();
            std::istringstream ss(sel_list);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                double val = 0.0;
                if (token.back() == '%')
                    val = std::stod(token.substr(0, token.size() - 1)) / 100.0;
                else
                    val = std::stod(token);
                FILTER_SELECTIVITIES.push_back(val);
            }
            if (FILTER_SELECTIVITIES.empty()) {
                std::cerr << "No valid selectivity values in: " << sel_list << "\n";
                std::exit(1);
            }
        } else if (arg == "--build-index") {
            BUILD_INDEX_ONLY = true;
        } else if (arg == "--build-gt") {
            BUILD_GT_ONLY = true;
        } else if (arg == "--build-gt-k") {
            BUILD_GT_K = next_val();
            if (BUILD_GT_K == 0)
                BUILD_GT_K = 1;
        } else if (arg == "--data-dir") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --data-dir\n";
                std::exit(1);
            }
            DATA_DIR = argv[++i];
        } else if (arg == "--volume-dir") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --volume-dir\n";
                std::exit(1);
            }
            VOLUME_DIR = argv[++i];
        } else if (arg == "--gt-dir") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --gt-dir\n";
                std::exit(1);
            }
            GT_DIR = argv[++i];
        } else if (arg == "--qps") {
            QPS_MODE = true;
        } else if (arg == "--ef-values") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --ef-values\n";
                std::exit(1);
            }
            std::string ef_list = argv[++i];
            QPS_EF_VALUES.clear();
            std::istringstream ss(ef_list);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                QPS_EF_VALUES.push_back(static_cast<std::size_t>(std::stoull(token)));
            }
            if (QPS_EF_VALUES.empty()) {
                std::cerr << "No valid ef_search values in: " << ef_list << "\n";
                std::exit(1);
            }
        } else if (arg == "--skip-exact") {
            SKIP_EXACT = true;
        } else if (arg == "--modes") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --modes\n";
                std::exit(1);
            }
            std::string mode_list = argv[++i];
            FILTER_MODES.clear();
            std::istringstream ss(mode_list);
            std::string token;
            static char const* const valid[] = {"exact", "hnsw", "acorn-1", "racorn-1", "racorn-1+"};
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                bool ok = false;
                for (auto v : valid)
                    if (iequals(token, v)) {
                        ok = true;
                        break;
                    }
                if (!ok) {
                    std::cerr << "Unknown mode: " << token << " (valid: exact, hnsw, acorn-1, racorn-1, racorn-1+)\n";
                    std::exit(1);
                }
                FILTER_MODES.push_back(token);
            }
            if (FILTER_MODES.empty()) {
                std::cerr << "No valid modes in: " << mode_list << "\n";
                std::exit(1);
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: bench_acorn [options]\n"
                      << "\n"
                      << "Modes (mutually exclusive; default = latency benchmark):\n"
                      << "  (default)             Latency benchmark (Exact/HNSW/ACORN-1/RACORN-1 per selectivity).\n"
                      << "  --build-index         Build the index volume and exit. No queries are run.\n"
                      << "  --build-gt            Build ground-truth file for queries. Requires existing volume.\n"
                      << "  --qps                 QPS benchmark: sweep ef_search values, report recall/QPS/latency.\n"
                      << "\n"
                      << "Notes:\n"
                      << "  - GT files saved by --build-gt are auto-loaded by latency/qps modes when their\n"
                      << "    header is compatible (file_nq >= bench nq, file_gt_k >= K, magic 'GT01').\n"
                      << "  - Two binaries are produced: `bench_acorn` (default) and `bench_acorn_3hop`\n"
                      << "    (compiled with USEARCH_ENABLE_3HOP=1). Only the 3hop binary actually runs the\n"
                      << "    3-hop expansion code path; `--3hop-threshold` is parsed but ignored otherwise.\n"
                      << "\n"
                      << "── Common options (all modes) ──────────────────────────────────────\n"
                      << "  --dataset <name>      Dataset: sift, gist, text2image (default: sift)\n"
                      << "  --n <N>               Number of vectors to index (default: 1000000)\n"
                      << "                        sift max: " << SIFT_N_MAX << ", gist max: " << GIST_N_MAX
                      << ", text2image max: " << TEXT2IMAGE_N_MAX << "\n"
                      << "  --m <N>               Graph connectivity M (default: 16)\n"
                      << "  --ef-construction <N> EF construction expansion (default: 100)\n"
                      << "  --k <N>               Top-K results (default: 100)\n"
                      << "  --metric <name>       Distance metric: l2, cosine\n"
                      << "                        (default: l2 for sift/gist, cosine for text2image)\n"
                      << "  --nq <N>              Number of query vectors to use (default: 100)\n"
                      << "                        sift max: 10000, gist max: 1000, text2image max: 100000\n"
                      << "  --search-threads <N>  Parallel search threads (default: 1; --build-gt\n"
                      << "                        defaults to hardware_concurrency when unset)\n"
                      << "  --data-dir <path>     Directory for raw dataset files (default: data/raw)\n"
                      << "  --volume-dir <path>   Directory for index volume files (default: data/volumes)\n"
                      << "  --gt-dir <path>       Directory for ground truth files (default: data/gt)\n"
                      << "  --selectivity <list>  Comma-separated selectivity values to filter test cases\n"
                      << "                        e.g. --selectivity 10%,1%,0.1% or --selectivity 0.10,0.01\n"
                      << "                        Applies to --build-gt, latency, and --qps modes.\n"
                      << "                        Ignored by --build-index. In latency mode, also enables\n"
                      << "                        per-query recall breakdown (avg, p50, p95, p99).\n"
                      << "\n"
                      << "── Volume build mode (--build-index) ───────────────────────────────\n"
                      << "  Uses --m / --ef-construction to control build quality. --nq, --selectivity,\n"
                      << "  and all benchmark-only flags are ignored. Existing volume is overwritten.\n"
                      << "\n"
                      << "── Ground truth build mode (--build-gt) ────────────────────────────\n"
                      << "  --build-gt-k <N>      GT results per query (default: same as --k)\n"
                      << "  --nq <N>              Use this many queries for GT (default: dataset q_max\n"
                      << "                        when --nq is not specified). Smaller --nq → smaller GT.\n"
                      << "  --selectivity <list>  Build only the listed selectivities (default: all 21).\n"
                      << "\n"
                      << "── Latency benchmark mode (default) ────────────────────────────────\n"
                      << "  --ef-search <N>           EF search expansion (default: 200)\n"
                      << "  --2hop-threshold <F>      ACORN-1 two-hop expansion threshold (default: 0.3)\n"
                      << "  --bridge-budget-ratio <F> RACORN-1 bridge budget ratio (default: 1.0)\n"
                      << "  --fallback-threshold <F>  RACORN-1+ exact-fallback threshold (default: 0.003).\n"
                      << "                            Set to 0 to disable RACORN-1+ (degenerates to RACORN-1).\n"
                      << "  --3hop-threshold <F>      3-hop expansion threshold (default: -1, disabled).\n"
                      << "                            Mutually exclusive with racorn.\n"
                      << "                            Effective only in `bench_acorn_3hop` binary.\n"
                      << "  --racorn-stride-sampling <s>  RACORN-1 stride sampling (default: on; off disables\n"
                      << "                            stride down-sampling at both call sites). RACORN-1\n"
                      << "                            ablation tool — see run_racorn1_ablation.sh.\n"
                      << "  --runs <N>                Measurement runs per query (default: 10)\n"
                      << "  --warmup <N>              Warmup runs before measurement (default: 3)\n"
                      << "  --skip-exact              Skip exact search latency benchmark.\n"
                      << "                            (GT for recall is still computed/loaded.)\n"
                      << "  --modes <list>            Comma-separated modes to run.\n"
                      << "                            Valid: exact, hnsw, acorn-1, racorn-1, racorn-1+.\n"
                      << "                            e.g. --modes hnsw,acorn-1 or --modes acorn-1\n"
                      << "                            Baseline (no-predicate) row skipped when set.\n"
                      << "                            Speedup/Recall/Dist ratios shown only when HNSW is included.\n"
                      << "\n"
                      << "── QPS benchmark mode (--qps) ──────────────────────────────────────\n"
                      << "  --ef-values <list>        Comma-separated ef_search values to sweep\n"
                      << "                            (default: 128,256,384,512). Replaces --ef-search.\n"
                      << "  --2hop-threshold <F>      ACORN-1 two-hop expansion threshold (default: 0.3)\n"
                      << "  --bridge-budget-ratio <F> RACORN-1 bridge budget ratio (default: 1.0)\n"
                      << "  --3hop-threshold <F>      3-hop expansion threshold (default: -1).\n"
                      << "                            Effective only in `bench_acorn_3hop` binary.\n"
                      << "  --runs <N>                Measurement runs per configuration (default: 10)\n"
                      << "                            Each run = full nq batch; reported QPS/latency are medians.\n"
                      << "  --warmup <N>              Warmup runs (default: 3)\n"
                      << "  --modes <list>            Comma-separated modes to run.\n"
                      << "                            Valid: hnsw, acorn-1, racorn-1.\n"
                      << "                            'exact' is rejected — QPS mode never runs exact search.\n"
                      << "  --skip-exact has no effect (already implied).\n"
                      << "\n"
                      << "── Cluster correlation mode (overrides modulo cases) ───────────────\n"
                      << "  --cluster-correlation <s> Filter mode: none|positive|negative.\n"
                      << "                            positive: target = query.cluster_id\n"
                      << "                              → node.cluster_id == query.cluster_id\n"
                      << "                              (selectivity ~1/K, query-correlated).\n"
                      << "                            negative: target = a random non-self cluster id picked\n"
                      << "                              once per query (deterministic seed)\n"
                      << "                              → node.cluster_id == target\n"
                      << "                              (selectivity ~1/K, uncorrelated with the query).\n"
                      << "                            Replaces all 21 modulo cases with one cluster case.\n"
                      << "  --cluster-k <N>           K used for the cluster files (default: 100).\n"
                      << "                            Cluster files are produced by\n"
                      << "                            cpp/scripts/build_clusters.py. K >= 2 required for\n"
                      << "                            negative correlation.\n"
                      << "  --cluster-dir <path>      Cluster file directory (default: data/clusters).\n"
                      << "                            File pattern: cluster_<dataset>_n<N>_k<K>.bin and\n"
                      << "                            cluster_<dataset>_query_n<N>_k<K>.bin.\n"
                      << "  GT files for cluster mode are suffixed with `_cluster_pos_kK` /\n"
                      << "  `_cluster_neg_kK`, so they don't collide with modulo GT files.\n"
                      << "  Note: negative-mode semantics changed (now random non-self target,\n"
                      << "  was previously `!=`). Existing `_cluster_neg_kK` GT files must be\n"
                      << "  rebuilt with --build-gt before running benches.\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << " (use --help)\n";
            std::exit(1);
        }
    }

    // Post-parse: set per-dataset limits and dimensions
    std::size_t n_max = SIFT_N_MAX;
    std::size_t q_max = 10'000; // sift query file has 10K vectors
    if (DATASET_NAME == "gist") {
        n_max = GIST_N_MAX;
        q_max = 1'000; // gist query file has 1K vectors
        D_VAL = 960;
    } else if (DATASET_NAME == "text2image") {
        n_max = TEXT2IMAGE_N_MAX;
        q_max = 100'000; // text2image query file has 100K vectors
        D_VAL = 200;
    } else {
        D_VAL = 128;
    }
    // Default metric per dataset (if not explicitly set via --metric)
    if (METRIC_NAME.empty()) {
        if (DATASET_NAME == "text2image")
            METRIC_NAME = "cosine";
        else
            METRIC_NAME = "l2"; // sift, gist: L2 euclidean
    }
    if (N_VAL == 0 || N_VAL > n_max) {
        std::cerr << "--n must be between 1 and " << n_max << " for dataset '" << DATASET_NAME << "' (got " << N_VAL
                  << ")\n";
        std::exit(1);
    }
    if (NUM_QUERIES > q_max) {
        std::cerr << "--nq capped to " << q_max << " for dataset '" << DATASET_NAME << "'\n";
        NUM_QUERIES = q_max;
    }

    // --build-gt defaults: default GT_K to K; default nq to q_max only if not user-specified;
    // default --search-threads to hardware_concurrency (GT is exact NN — embarrassingly parallel,
    // so on shared boxes it is fine to use every available core for the one-time build).
    if (BUILD_GT_ONLY) {
        if (BUILD_GT_K == 0)
            BUILD_GT_K = K_VAL;
        if (!NUM_QUERIES_SET)
            NUM_QUERIES = q_max;
        if (!SEARCH_THREADS_SET) {
            std::size_t const hw = std::thread::hardware_concurrency();
            SEARCH_THREADS = hw == 0 ? 1 : hw;
            std::cout << "[--build-gt] Auto-set --search-threads to " << SEARCH_THREADS << " (hardware_concurrency)\n";
        }
        if (BUILD_INDEX_ONLY) {
            std::cerr << "--build-index and --build-gt are mutually exclusive\n";
            std::exit(1);
        }
    }
    if (QPS_MODE && (BUILD_INDEX_ONLY || BUILD_GT_ONLY)) {
        std::cerr << "--qps cannot be combined with --build-index or --build-gt\n";
        std::exit(1);
    }
}

// ── Benchmark runner (templated on metric type) ─────────────────────────────
template <typename metric_t>
int run_benchmark(vector_store_t<float>& store, float const* query_vecs_data, std::size_t num_queries,
                  char const* metric_label) {

    // For cosine metric: L2-normalize all vectors so cosine distance works on unit vectors
    if (METRIC_NAME == "cosine") {
        std::cout << "Normalizing vectors to unit length for cosine metric... " << std::flush;
        stopwatch_t norm_timer;
        store.normalize_l2();
        std::cout << "done (" << std::fixed << std::setprecision(1) << norm_timer.elapsed_ms() << " ms)\n";
    }

    // Prepare query vectors (up to NUM_QUERIES) — skip in build-index-only mode
    std::vector<float> query_data_owned; // storage for (possibly normalized) query vectors
    std::vector<float const*> queries;   // pointers into query_data_owned or store

    if (!BUILD_INDEX_ONLY || BUILD_GT_ONLY) {
        if (num_queries > 0 && query_vecs_data) {
            std::size_t nq = std::min(num_queries, NUM_QUERIES);
            query_data_owned.assign(query_vecs_data, query_vecs_data + nq * D);
            if (METRIC_NAME == "cosine") {
                for (std::size_t qi = 0; qi < nq; ++qi) {
                    float* v = query_data_owned.data() + qi * D;
                    float norm_sq = 0;
                    for (std::size_t d = 0; d < D; ++d)
                        norm_sq += v[d] * v[d];
                    float norm = std::sqrt(norm_sq);
                    if (norm > 0) {
                        float inv = 1.0f / norm;
                        for (std::size_t d = 0; d < D; ++d)
                            v[d] *= inv;
                    }
                }
                std::cout << "Query vectors normalized to unit length.\n";
            }
            queries.reserve(nq);
            for (std::size_t qi = 0; qi < nq; ++qi)
                queries.push_back(query_data_owned.data() + qi * D);
            std::cout << "Using " << nq << " " << DATASET_NAME << " query vectors\n";
        } else {
            std::cerr << "No query vectors available for dataset '" << DATASET_NAME << "'\n";
            return 1;
        }
    }

    // ── Cluster correlation files (loaded once when --cluster-correlation != none).
    if (CLUSTER_CORRELATION != cluster_correlation_t::none) {
        std::string base_path = cluster_base_path();
        std::string query_path = cluster_query_path();
        std::cout << "Loading cluster files (K=" << CLUSTER_K << "):\n  " << base_path << "\n  " << query_path << "\n";
        if (!load_cluster_file(base_path, NODE_CLUSTERS, N, CLUSTER_K, /*exact=*/true))
            return 1;
        // Query cluster file may have been built with the dataset's q_max
        // (e.g. 100K for text2image) while the bench uses fewer queries —
        // accept >= queries.size() (exact_count=false).
        if (!load_cluster_file(query_path, QUERY_CLUSTERS, queries.size(), CLUSTER_K, /*exact=*/false))
            return 1;
        // Trim query clusters to the bench-time nq so downstream code can
        // assume QUERY_CLUSTERS.size() == queries.size() (selectivity stats,
        // negative-target picker, predicate factory all key off the index q).
        if (QUERY_CLUSTERS.size() > queries.size())
            QUERY_CLUSTERS.resize(queries.size());
        std::cout << "  loaded base cluster IDs (" << NODE_CLUSTERS.size() << ") + query cluster IDs ("
                  << QUERY_CLUSTERS.size() << ")\n";
        if (CLUSTER_CORRELATION == cluster_correlation_t::negative) {
            if (!populate_negative_targets(QUERY_CLUSTERS, CLUSTER_K, NEGATIVE_TARGET_SEED, NEGATIVE_TARGET_CLUSTERS))
                return 1;
            std::cout << "  computed " << NEGATIVE_TARGET_CLUSTERS.size()
                      << " negative target cluster IDs (random non-self per query, seed="
                      << NEGATIVE_TARGET_SEED << ")\n";
        }
    }

    // Build or load index
    index_config_t config(M);
    aligned_index_t<index_t> aligned(config);
    auto& index = *aligned.ptr;

    std::size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;

    metric_t metric{&store, D};
    std::string vol_path = index_volume_path();
    bool index_loaded = false;

    // Try loading existing volume (skip in build-index-only mode to force rebuild)
    if (!BUILD_INDEX_ONLY) {
        std::ifstream probe(vol_path);
        if (probe.good()) {
            probe.close();
            std::cout << "Loading index from " << vol_path << "... " << std::flush;
            stopwatch_t load_timer;
            auto result = index.load(vol_path.c_str());
            if (result) {
                index_loaded = true;
                // Reserve thread contexts for parallel search
                std::size_t search_threads = std::max(SEARCH_THREADS, std::size_t(1));
                index.reserve(index_limits_t(index.size(), search_threads));
                std::cout << "done (" << std::fixed << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
                std::cout << "Index size: " << index.size() << "  [loaded from volume]\n" << std::endl;
            } else {
                std::cerr << "Warning: failed to load " << vol_path << ", building from scratch\n";
            }
        }
    }

    if (BUILD_GT_ONLY && !index_loaded) {
        std::cerr << "Error: --build-gt requires existing index volume: " << vol_path << "\n";
        return 1;
    }

    if (!index_loaded) {
        index.reserve(index_limits_t(N, std::max(num_threads, SEARCH_THREADS)));

        std::cout << "Building index with " << num_threads << " threads... " << std::flush;
        stopwatch_t build_timer;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        std::size_t chunk = (N + num_threads - 1) / num_threads;

        for (std::size_t t = 0; t < num_threads; ++t) {
            std::size_t begin = t * chunk;
            std::size_t end = std::min(begin + chunk, N);
            threads.emplace_back([&, begin, end, t]() {
                index_update_config_t ucfg;
                ucfg.expansion = EF_CONSTRUCTION;
                ucfg.thread = t;
                for (std::size_t i = begin; i < end; ++i)
                    index.add(static_cast<vec_key_t>(i), store.row(i), metric, ucfg);
            });
        }
        for (auto& th : threads)
            th.join();

        double build_ms = build_timer.elapsed_ms();
        std::cout << "done (" << std::fixed << std::setprecision(1) << build_ms << " ms, "
                  << static_cast<std::size_t>(N / (build_ms / 1000.0)) << " vec/s)\n";
        std::cout << "Index size: " << index.size() << "  [built from scratch]\n\n";
    }

    // --build-index: save and exit
    if (BUILD_INDEX_ONLY) {
        std::error_code ec;
        std::filesystem::create_directories(VOLUME_DIR, ec);  // no-op if exists
        std::cout << "Saving index to " << vol_path << "... " << std::flush;
        stopwatch_t save_timer;
        auto result = index.save(vol_path.c_str());
        if (!result) {
            std::cerr << "Failed to save index: " << result.error.release() << "\n";
            return 1;
        }
        std::cout << "done (" << std::fixed << std::setprecision(1) << save_timer.elapsed_ms() << " ms)\n";
        std::cout << "Volume saved: " << vol_path << "\n";
        return 0;
    }

    // Test configurations
    struct test_case_t {
        char const* label;
        char const* mode;
        double selectivity;
        std::size_t modulus;
        std::size_t pred_threshold;
        double two_hop;
        bool acorn;
        bool racorn;
        bool exact;
        double fallback_exact_threshold; // 0 = plain RACORN-1 / disabled
    };

    auto gt_key = [](std::size_t mod, std::size_t thr) { return (mod << 20) | thr; };

    // Cluster-mode selectivity statistics (computed when CLUSTER_CORRELATION != none).
    // Mean is used as the case selectivity / display value; per-query distribution
    // is reported alongside QPS / search summaries so we can see how cluster-size
    // imbalance translates into per-query filter ratios.
    cluster_sel_stats_t cluster_stats;
    bool cluster_mode = (CLUSTER_CORRELATION != cluster_correlation_t::none);
    bool cluster_pos = (CLUSTER_CORRELATION == cluster_correlation_t::positive);
    std::string cluster_label_owned;
    if (cluster_mode) {
        auto const& target_clusters = cluster_pos ? QUERY_CLUSTERS : NEGATIVE_TARGET_CLUSTERS;
        cluster_stats = compute_cluster_selectivity_stats(NODE_CLUSTERS, target_clusters, CLUSTER_K, N);
        std::ostringstream lbl;
        lbl << "Cluster " << (cluster_pos ? "Pos" : "Neg") << " K=" << CLUSTER_K;
        cluster_label_owned = lbl.str();
        std::cout << "\nCluster correlation: " << (cluster_pos ? "positive" : "negative") << " (K=" << CLUSTER_K
                  << ")\n";
        std::cout << "  per-query selectivity: mean=" << std::fixed << std::setprecision(4)
                  << (cluster_stats.mean * 100.0) << "%  p50=" << (cluster_stats.p50 * 100.0)
                  << "%  p90=" << (cluster_stats.p90 * 100.0) << "%  p95=" << (cluster_stats.p95 * 100.0)
                  << "%  p99=" << (cluster_stats.p99 * 100.0) << "%  max=" << (cluster_stats.max * 100.0) << "%\n";
    }

#define CASE(sel_label, sel, mod, thr)                                                                                 \
    {sel_label " | Exact", "exact", sel, mod, thr, -1.0, false, false, true, 0.0},                                     \
        {sel_label " | HNSW", "hnsw", sel, mod, thr, -1.0, false, false, false, 0.0},                                  \
        {sel_label " | ACORN-1", "acorn-1", sel, mod, thr, ACORN_THRESHOLD, true, false, false, 0.0},                  \
        {sel_label " | RACORN-1", "racorn-1", sel, mod, thr, ACORN_THRESHOLD, true, true, false, 0.0},                 \
        {sel_label " | RACORN-1+", "racorn-1+", sel,  mod,   thr,                                                      \
         ACORN_THRESHOLD,          true,        true, false, FALLBACK_EXACT_THRESHOLD}

    std::vector<test_case_t> cases;
    if (!cluster_mode) {
        cases = {
            CASE("Selectivity 100%", 1.000, 1, 1),       CASE("Selectivity  90%", 0.900, 10, 9),
            CASE("Selectivity  70%", 0.700, 10, 7),      CASE("Selectivity  50%", 0.500, 2, 1),
            CASE("Selectivity  30%", 0.300, 10, 3),      CASE("Selectivity  20%", 0.200, 10, 2),
            CASE("Selectivity  10%", 0.100, 10, 1),      CASE("Selectivity   5%", 0.050, 20, 1),
            CASE("Selectivity   4%", 0.040, 25, 1),      CASE("Selectivity   3%", 0.030, 100, 3),
            CASE("Selectivity   2%", 0.020, 50, 1),      CASE("Selectivity   1%", 0.010, 100, 1),
            CASE("Selectivity 0.8%", 0.0080, 125, 1),    CASE("Selectivity 0.6%", 0.0060, 500, 3),
            CASE("Selectivity 0.5%", 0.0050, 200, 1),    CASE("Selectivity 0.4%", 0.0040, 250, 1),
            CASE("Selectivity 0.3%", 0.0030, 1000, 3),   CASE("Selectivity 0.2%", 0.0020, 500, 1),
            CASE("Selectivity 0.1%", 0.0010, 1000, 1),   CASE("Selectivity 0.05%", 0.0005, 2000, 1),
            CASE("Selectivity 0.01%", 0.0001, 10000, 1),
        };
    } else {
        // Cluster mode: one selectivity (mean per-query), 5 modes. Sentinel
        // (modulus=0, threshold=CLUSTER_K) — modulo cases never use mod=0,
        // so gt_key is collision-free across modes.
        char const* base = cluster_label_owned.c_str();
        // Build label strings owned by static storage via a small helper.
        // Concatenate at allocation time — the labels live for the rest of the run.
        static std::string lbl_exact, lbl_hnsw, lbl_acorn, lbl_racorn, lbl_racorn_plus;
        lbl_exact = std::string(base) + " | Exact";
        lbl_hnsw = std::string(base) + " | HNSW";
        lbl_acorn = std::string(base) + " | ACORN-1";
        lbl_racorn = std::string(base) + " | RACORN-1";
        lbl_racorn_plus = std::string(base) + " | RACORN-1+";
        double s = cluster_stats.mean;
        std::size_t mod = 0;
        std::size_t thr = CLUSTER_K;
        cases = {
            {lbl_exact.c_str(), "exact", s, mod, thr, -1.0, false, false, true, 0.0},
            {lbl_hnsw.c_str(), "hnsw", s, mod, thr, -1.0, false, false, false, 0.0},
            {lbl_acorn.c_str(), "acorn-1", s, mod, thr, ACORN_THRESHOLD, true, false, false, 0.0},
            {lbl_racorn.c_str(), "racorn-1", s, mod, thr, ACORN_THRESHOLD, true, true, false, 0.0},
            {lbl_racorn_plus.c_str(), "racorn-1+", s, mod, thr, ACORN_THRESHOLD, true, true, false,
             FALLBACK_EXACT_THRESHOLD},
        };
    }
#undef CASE

    // Predicate-factory dispatch:
    //   * modulo mode  → factory returns a fixed selectivity_predicate_t
    //   * cluster mode → factory builds a per-query cluster_predicate_t
    // Both factories accept the query index; modulo factory ignores it.
    auto make_modulo_factory = [](std::size_t modulus, std::size_t pred_threshold) {
        selectivity_predicate_t pred{modulus, pred_threshold};
        return [pred](std::size_t /*q*/) { return pred; };
    };
    auto make_cluster_factory = [](bool positive) {
        uint32_t const* nc = NODE_CLUSTERS.data();
        // positive → query's own cluster id; negative → pre-computed random non-self target
        uint32_t const* targets = positive ? QUERY_CLUSTERS.data() : NEGATIVE_TARGET_CLUSTERS.data();
        return [nc, targets](std::size_t q) {
            return cluster_predicate_t{nc, targets[q]};
        };
    };

    // Filter cases to only those matching FILTER_SELECTIVITIES (if set)
    auto selectivity_matches = [](double sel) -> bool {
        if (FILTER_SELECTIVITIES.empty())
            return true;
        for (double f : FILTER_SELECTIVITIES)
            if (std::fabs(sel - f) < 1e-9)
                return true;
        return false;
    };

    // --build-gt: compute ground truth for all selectivities and save to file
    if (BUILD_GT_ONLY) {
        std::size_t gt_k = BUILD_GT_K;
        std::size_t nq = queries.size();
        std::string gt_path = gt_file_path(gt_k, nq);

        // Collect unique (modulus, threshold) pairs from test cases
        struct gt_sel_t {
            double selectivity;
            std::size_t modulus;
            std::size_t pred_threshold;
        };
        std::vector<gt_sel_t> gt_sels;
        std::unordered_set<std::size_t> seen;
        for (auto const& tc : cases) {
            if (!selectivity_matches(tc.selectivity))
                continue;
            std::size_t key = gt_key(tc.modulus, tc.pred_threshold);
            if (seen.count(key))
                continue;
            seen.insert(key);
            gt_sels.push_back({tc.selectivity, tc.modulus, tc.pred_threshold});
        }

        std::cout << "\n========================================================\n";
        std::cout << " Building ground truth: " << gt_sels.size() << " selectivities, " << nq << " queries, k=" << gt_k
                  << "\n";
        std::cout << "========================================================\n";

        // Compute GT for each selectivity (parallel over queries)
        std::size_t gt_threads = std::max(SEARCH_THREADS, std::size_t(1));

        // Binary format:
        //   [char x 4] magic "GT01"
        //   [uint32]   num_selectivities
        //   [uint32]   num_queries
        //   [uint32]   gt_k
        //   Per selectivity:
        //     [double]   selectivity
        //     [uint64]   modulus
        //     [uint64]   threshold
        //     Per query:
        //       [uint32]       count (actual results, <= gt_k)
        //       [uint64 x gt_k] keys (unused slots = UINT64_MAX)

        std::error_code ec;
        std::filesystem::create_directories(GT_DIR, ec);  // no-op if exists
        std::ofstream ofs(gt_path, std::ios::binary);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open " << gt_path << " for writing\n";
            return 1;
        }

        ofs.write("GT01", 4);
        uint32_t u32;
        u32 = static_cast<uint32_t>(gt_sels.size());
        ofs.write(reinterpret_cast<char const*>(&u32), sizeof(u32));
        u32 = static_cast<uint32_t>(nq);
        ofs.write(reinterpret_cast<char const*>(&u32), sizeof(u32));
        u32 = static_cast<uint32_t>(gt_k);
        ofs.write(reinterpret_cast<char const*>(&u32), sizeof(u32));

        for (auto const& sel : gt_sels) {
            std::cout << "  Computing GT for selectivity " << std::fixed << std::setprecision(4)
                      << (sel.selectivity * 100.0) << "% (mod=" << sel.modulus << ", thr=" << sel.pred_threshold
                      << ") ... " << std::flush;
            stopwatch_t gt_timer;

            // Per-query predicate factory: cluster mode uses a per-query
            // cluster_predicate_t (so each query filters by its own cluster id);
            // modulo mode returns the same fixed predicate for every query.
            auto run_one = [&](std::size_t qi, std::size_t tid, std::vector<std::vector<vec_key_t>>& gts) {
                index_search_config_t exact_cfg;
                exact_cfg.exact = true;
                exact_cfg.thread = tid;
                if (cluster_mode) {
                    uint32_t target = cluster_pos ? QUERY_CLUSTERS[qi] : NEGATIVE_TARGET_CLUSTERS[qi];
                    cluster_predicate_t pred{NODE_CLUSTERS.data(), target};
                    auto result = index.search(queries[qi], gt_k, metric, exact_cfg, pred);
                    gts[qi].reserve(result.size());
                    for (std::size_t ri = 0; ri < result.size(); ++ri)
                        gts[qi].push_back(result[ri].member.key);
                } else {
                    selectivity_predicate_t pred{sel.modulus, sel.pred_threshold};
                    auto result = index.search(queries[qi], gt_k, metric, exact_cfg, pred);
                    gts[qi].reserve(result.size());
                    for (std::size_t ri = 0; ri < result.size(); ++ri)
                        gts[qi].push_back(result[ri].member.key);
                }
            };

            // Parallel GT computation
            std::vector<std::vector<vec_key_t>> gts(nq);
            if (gt_threads <= 1) {
                for (std::size_t qi = 0; qi < nq; ++qi)
                    run_one(qi, 0, gts);
            } else {
                std::vector<std::thread> pool;
                pool.reserve(gt_threads);
                std::size_t chunk = (nq + gt_threads - 1) / gt_threads;
                for (std::size_t t = 0; t < gt_threads; ++t) {
                    std::size_t qbegin = t * chunk;
                    std::size_t qend = std::min(qbegin + chunk, nq);
                    pool.emplace_back([&, qbegin, qend, t]() {
                        for (std::size_t qi = qbegin; qi < qend; ++qi)
                            run_one(qi, t, gts);
                    });
                }
                for (auto& th : pool)
                    th.join();
            }

            std::cout << "done (" << std::fixed << std::setprecision(1) << gt_timer.elapsed_ms() << " ms)\n";

            // Write selectivity block
            ofs.write(reinterpret_cast<char const*>(&sel.selectivity), sizeof(double));
            uint64_t u64;
            u64 = static_cast<uint64_t>(sel.modulus);
            ofs.write(reinterpret_cast<char const*>(&u64), sizeof(u64));
            u64 = static_cast<uint64_t>(sel.pred_threshold);
            ofs.write(reinterpret_cast<char const*>(&u64), sizeof(u64));

            std::vector<uint64_t> key_buf(gt_k, UINT64_MAX);
            for (std::size_t qi = 0; qi < nq; ++qi) {
                uint32_t cnt = static_cast<uint32_t>(gts[qi].size());
                ofs.write(reinterpret_cast<char const*>(&cnt), sizeof(cnt));
                std::fill(key_buf.begin(), key_buf.end(), UINT64_MAX);
                for (std::size_t ri = 0; ri < gts[qi].size() && ri < gt_k; ++ri)
                    key_buf[ri] = static_cast<uint64_t>(gts[qi][ri]);
                ofs.write(reinterpret_cast<char const*>(key_buf.data()), gt_k * sizeof(uint64_t));
            }
        }

        ofs.close();
        std::cout << "\nGround truth saved: " << gt_path << "\n";
        std::cout << "  Selectivities: " << gt_sels.size() << ", queries: " << nq << ", k: " << gt_k << "\n";
        return 0;
    }

    std::cout << "\n========================================================\n";
    if (QPS_MODE) {
        std::cout << " QPS Benchmark (ef_search: ";
        for (std::size_t i = 0; i < QPS_EF_VALUES.size(); ++i) {
            if (i)
                std::cout << ", ";
            std::cout << QPS_EF_VALUES[i];
        }
        std::cout << ")\n";
    } else if (!FILTER_SELECTIVITIES.empty()) {
        std::cout << " Search Benchmarks (detail mode: ";
        for (std::size_t fi = 0; fi < FILTER_SELECTIVITIES.size(); ++fi) {
            if (fi)
                std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << (FILTER_SELECTIVITIES[fi] * 100.0) << "%";
        }
        std::cout << ")\n";
    } else {
        std::cout << " Search Benchmarks\n";
    }
    std::cout << "========================================================\n";

    // Compute ground truth per (modulus, threshold) combination, for all queries
    std::unordered_map<std::size_t, std::vector<std::vector<vec_key_t>>> gt_cache;

    // Helper: compute ground truth for all queries in parallel.
    // Accepts a per-query predicate factory `make_pred(qi)` so cluster mode can
    // build a query-specific cluster_predicate_t. Modulo callers wrap their
    // fixed predicate in a constant lambda.
    std::size_t gt_threads = std::min(SEARCH_THREADS, queries.size());
    auto compute_gt_parallel = [&](auto&& make_pred) {
        std::vector<std::vector<vec_key_t>> gts(queries.size());
        if (gt_threads <= 1) {
            for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                auto pred = make_pred(qi);
                gts[qi] = compute_ground_truth(index, metric, queries[qi], K, pred, 0);
            }
        } else {
            std::vector<std::thread> pool;
            pool.reserve(gt_threads);
            std::size_t chunk = (queries.size() + gt_threads - 1) / gt_threads;
            for (std::size_t t = 0; t < gt_threads; ++t) {
                std::size_t qbegin = t * chunk;
                std::size_t qend = std::min(qbegin + chunk, queries.size());
                pool.emplace_back([&, qbegin, qend, t]() {
                    for (std::size_t qi = qbegin; qi < qend; ++qi) {
                        auto pred = make_pred(qi);
                        gts[qi] = compute_ground_truth(index, metric, queries[qi], K, pred, t);
                    }
                });
            }
            for (auto& th : pool)
                th.join();
        }
        return gts;
    };

    // Count unique GT computations needed
    std::size_t gt_total = 0;
    {
        std::unordered_set<std::size_t> seen;
        for (auto const& tc : cases) {
            if (!selectivity_matches(tc.selectivity))
                continue;
            std::size_t key = gt_key(tc.modulus, tc.pred_threshold);
            if (seen.count(key))
                continue;
            seen.insert(key);
            ++gt_total;
        }
    }

    // Try to populate gt_cache from a previously saved GT file (--build-gt output).
    // File path: gt_file_path(K, file_nq) where file_nq = dataset's q_max (build-gt forces it).
    // Compatibility rules:
    //   - file num_queries >= bench queries.size() (we use first queries.size() entries)
    //   - file gt_k >= K                           (we use first K keys per query)
    //   - selectivity match keyed by (modulus, threshold)
    {
        // Try several common nq values (descending preference: exact match → q_max → 1000 → others).
        // The first existing file with header `nq >= queries.size()` is used.
        std::size_t qmax_for_dataset = (DATASET_NAME == "gist")         ? 1'000
                                       : (DATASET_NAME == "text2image") ? 100'000
                                                                        : 10'000; // sift
        std::vector<std::size_t> candidate_nqs;
        candidate_nqs.push_back(queries.size());   // exact bench-nq (preferred)
        candidate_nqs.push_back(1000);             // build_all_gt.sh default
        candidate_nqs.push_back(qmax_for_dataset); // legacy q_max files
        candidate_nqs.push_back(100);              // bench default
        // Keep only nq >= bench queries.size(), dedup preserving order
        std::vector<std::size_t> tried;
        std::string gt_path;
        for (std::size_t nq : candidate_nqs) {
            if (nq < queries.size())
                continue;
            if (std::find(tried.begin(), tried.end(), nq) != tried.end())
                continue;
            tried.push_back(nq);
            std::string p = gt_file_path(K, nq);
            std::ifstream test(p);
            if (test.good()) {
                gt_path = p;
                break;
            }
        }
        std::ifstream ifs(gt_path, std::ios::binary);
        if (ifs.is_open()) {
            char magic[4] = {};
            ifs.read(magic, 4);
            uint32_t num_sels = 0, file_nq_hdr = 0, file_gt_k = 0;
            ifs.read(reinterpret_cast<char*>(&num_sels), sizeof(num_sels));
            ifs.read(reinterpret_cast<char*>(&file_nq_hdr), sizeof(file_nq_hdr));
            ifs.read(reinterpret_cast<char*>(&file_gt_k), sizeof(file_gt_k));

            bool valid = ifs.good() && std::string(magic, 4) == "GT01" //
                         && file_nq_hdr >= queries.size() && file_gt_k >= K;
            if (!valid) {
                std::cout << "GT file " << gt_path << " skipped (";
                if (std::string(magic, 4) != "GT01")
                    std::cout << "bad magic";
                else if (file_nq_hdr < queries.size())
                    std::cout << "nq=" << file_nq_hdr << " < bench " << queries.size();
                else if (file_gt_k < K)
                    std::cout << "gt_k=" << file_gt_k << " < K=" << K;
                std::cout << "); recomputing.\n";
            } else {
                std::cout << "Loaded GT file: " << gt_path << " (" << num_sels << " selectivities, " << file_nq_hdr
                          << " queries, gt_k=" << file_gt_k << ")\n";
                std::vector<uint64_t> key_buf(file_gt_k);
                for (uint32_t si = 0; si < num_sels && ifs.good(); ++si) {
                    double sel_val = 0.0;
                    uint64_t mod = 0, thr = 0;
                    ifs.read(reinterpret_cast<char*>(&sel_val), sizeof(sel_val));
                    ifs.read(reinterpret_cast<char*>(&mod), sizeof(mod));
                    ifs.read(reinterpret_cast<char*>(&thr), sizeof(thr));
                    std::size_t key = gt_key(static_cast<std::size_t>(mod), static_cast<std::size_t>(thr));

                    std::vector<std::vector<vec_key_t>> gts(queries.size());
                    for (std::size_t qi = 0; qi < file_nq_hdr; ++qi) {
                        uint32_t cnt = 0;
                        ifs.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
                        ifs.read(reinterpret_cast<char*>(key_buf.data()), file_gt_k * sizeof(uint64_t));
                        if (qi < queries.size()) {
                            std::size_t use_cnt = std::min<std::size_t>(cnt, K);
                            gts[qi].reserve(use_cnt);
                            for (std::size_t ri = 0; ri < use_cnt; ++ri)
                                if (key_buf[ri] != UINT64_MAX)
                                    gts[qi].push_back(static_cast<vec_key_t>(key_buf[ri]));
                        }
                    }
                    // Only insert if needed by current run (selectivity filter)
                    bool needed = false;
                    for (auto const& tc : cases) {
                        if (!selectivity_matches(tc.selectivity))
                            continue;
                        if (gt_key(tc.modulus, tc.pred_threshold) == key) {
                            needed = true;
                            break;
                        }
                    }
                    if (needed)
                        gt_cache[key] = std::move(gts);
                }
            }
        }
    }

    std::size_t gt_done = 0;
    std::size_t gt_hits = gt_cache.size(); // selectivities populated from file
    std::size_t gt_misses = (gt_total > gt_hits) ? (gt_total - gt_hits) : 0;
    if (gt_misses > 0) {
        std::cout << "\nComputing exact ground truth per selectivity (" << queries.size() << " queries each, "
                  << gt_threads << " threads, " << gt_misses << " of " << gt_total << " selectivities, " << gt_hits
                  << " loaded from file)...\n";
    } else {
        std::cout << "\nAll " << gt_total << " selectivities served from GT file; skipping exact recompute.\n";
    }
    for (auto const& tc : cases) {
        if (!selectivity_matches(tc.selectivity))
            continue;
        std::size_t key = gt_key(tc.modulus, tc.pred_threshold);
        if (gt_cache.count(key))
            continue;
        stopwatch_t gt_timer;
        if (cluster_mode) {
            auto factory = make_cluster_factory(cluster_pos);
            gt_cache[key] = compute_gt_parallel(factory);
        } else {
            auto factory = make_modulo_factory(tc.modulus, tc.pred_threshold);
            gt_cache[key] = compute_gt_parallel(factory);
        }
        ++gt_done;
        std::cout << "  [" << gt_done << "/" << gt_misses << "] " << tc.label << ": " << queries.size() << " queries ("
                  << std::setprecision(1) << gt_timer.elapsed_ms() << " ms)\n";
    }

    // ── QPS mode: sweep ef_search values, report recall & QPS ───────────────
    if (QPS_MODE) {
        // Collect unique selectivities from test cases
        struct sel_info_t {
            double selectivity;
            std::size_t modulus;
            std::size_t pred_threshold;
        };
        std::vector<sel_info_t> unique_sels;
        {
            std::unordered_set<std::size_t> seen;
            for (auto const& tc : cases) {
                if (!selectivity_matches(tc.selectivity))
                    continue;
                std::size_t key = gt_key(tc.modulus, tc.pred_threshold);
                if (seen.count(key))
                    continue;
                seen.insert(key);
                unique_sels.push_back({tc.selectivity, tc.modulus, tc.pred_threshold});
            }
        }

        struct qps_mode_t {
            char const* name;
            double two_hop;
            bool racorn;
            double fallback_exact_threshold;
        };
        qps_mode_t all_modes[] = {
            {"HNSW", -1.0, false, 0.0},
            {"ACORN-1", ACORN_THRESHOLD, false, 0.0},
            {"RACORN-1", ACORN_THRESHOLD, true, 0.0},
            {"RACORN-1+", ACORN_THRESHOLD, true, FALLBACK_EXACT_THRESHOLD},
        };
        std::vector<qps_mode_t> modes;
        for (auto const& m : all_modes)
            if (mode_matches(m.name))
                modes.push_back(m);
        if (modes.empty()) {
            std::cerr << "Error: no QPS modes selected (QPS mode does not support --modes exact).\n";
            return 1;
        }
        if (unique_sels.empty()) {
            std::cerr << "Error: no selectivities to run. Check --selectivity filter.\n";
            return 1;
        }

        struct qps_row_t {
            double selectivity;
            char const* mode;
            std::size_t ef_search;
            double recall;
            double qps;
            double latency_ms;
        };
        std::vector<qps_row_t> rows;

        std::size_t total_configs = unique_sels.size() * QPS_EF_VALUES.size() * modes.size();
        std::size_t done = 0;

        for (auto const& sel : unique_sels) {
            std::size_t gk = gt_key(sel.modulus, sel.pred_threshold);
            auto& gt = gt_cache[gk];

            for (auto const& m : modes) {
                for (std::size_t ef : QPS_EF_VALUES) {
                    // RACORN-1+ dynamic threshold: scales with ef_search (so larger
                    // searches see proportionally more fallback samples before the
                    // heuristic fires) and shrinks 10× when N > 1M (where each
                    // fallback round explores a much smaller fraction of the index).
                    // Plain RACORN-1 / HNSW / ACORN-1 keep `m.fallback_exact_threshold = 0`
                    // and pass it through unchanged.
                    double dynamic_thr = m.fallback_exact_threshold;
                    if (dynamic_thr > 0) {
                        double ef_factor = static_cast<double>(ef) / 200.0;
                        double n_factor = (N <= 1'000'000) ? 1.0 : 0.1;
                        dynamic_thr *= ef_factor * n_factor;
                    }
                    auto run_with = [&](auto&& factory) {
                        return run_qps_search(index, metric, queries, K, ef, factory, m.two_hop, m.racorn, dynamic_thr,
                                              gt);
                    };
                    qps_result_t r = cluster_mode ? run_with(make_cluster_factory(cluster_pos))
                                                  : run_with(make_modulo_factory(sel.modulus, sel.pred_threshold));
                    rows.push_back({sel.selectivity, m.name, ef, r.recall, r.qps, r.latency_ms});
                    ++done;
                    std::cout << "\r  [" << done << "/" << total_configs << "] " << std::fixed << std::setprecision(1)
                              << (sel.selectivity * 100.0) << "% | " << m.name << " | ef=" << ef
                              << " => recall=" << std::setprecision(4) << r.recall << "  qps=" << std::setprecision(0)
                              << r.qps << "  lat=" << std::setprecision(3) << r.latency_ms << "ms"
                              << "        " << std::flush;
                }
            }
        }
        std::cout << "\n";

        // ── QPS Summary table ───────────────────────────────────────────────
        std::cout << "\n========================================================\n";
        std::cout << " QPS Benchmark Summary\n";
        std::cout << "========================================================\n\n";

        std::cout << "  Vectors     : " << N << "\n";
        std::cout << "  Dimensions  : " << D << "\n";
        std::cout << "  Connectivity: " << M << "\n";
        std::cout << "  EF construct: " << EF_CONSTRUCTION << "\n";
        std::cout << "  K           : " << K << "\n";
        std::cout << "  ACORN thresh: " << ACORN_THRESHOLD << "\n";
        std::cout << "  Bridge ratio: " << BRIDGE_BUDGET_RATIO << "\n";
        if (FALLBACK_EXACT_THRESHOLD > 0) {
            double ef_lo = static_cast<double>(*std::min_element(QPS_EF_VALUES.begin(), QPS_EF_VALUES.end()));
            double ef_hi = static_cast<double>(*std::max_element(QPS_EF_VALUES.begin(), QPS_EF_VALUES.end()));
            double nf = (N <= 1'000'000) ? 1.0 : 0.1;
            std::cout << std::fixed << std::setprecision(4)
                      << "  FB threshold: " << FALLBACK_EXACT_THRESHOLD
                      << " (QPS dynamic: base×ef/200"
                      << (nf < 1.0 ? "×0.1 (N>1M)" : "")
                      << " → " << (FALLBACK_EXACT_THRESHOLD * ef_lo / 200.0 * nf)
                      << ".." << (FALLBACK_EXACT_THRESHOLD * ef_hi / 200.0 * nf) << ")\n";
        }
#if USEARCH_ENABLE_3HOP
        std::cout << "  3hop thresh : " << THREE_HOP_THRESHOLD << "\n";
#endif
        std::cout << "  Metric      : " << metric_label << "\n";
        std::cout << "  Dataset     : " << DATASET_NAME << "\n";
        std::cout << "  Queries     : " << queries.size() << "\n";
        std::cout << "  Search thr  : " << SEARCH_THREADS << "\n";
        std::cout << "  Runs        : " << BENCH_RUNS << " (" << BENCH_WARMUP << " warmup)\n";
        std::cout << "  EF values   : ";
        for (std::size_t i = 0; i < QPS_EF_VALUES.size(); ++i) {
            if (i)
                std::cout << ", ";
            std::cout << QPS_EF_VALUES[i];
        }
        std::cout << "\n  Index source: " << (index_loaded ? "loaded (" + vol_path + ")" : "built from scratch")
                  << "\n";
        if (cluster_mode) {
            std::cout << "  Correlation : " << (cluster_pos ? "positive" : "negative") << " (cluster K=" << CLUSTER_K
                      << ")\n";
            std::cout << "  Per-q sel.  : mean=" << std::fixed << std::setprecision(4)
                      << (cluster_stats.mean * 100.0) << "%  p50=" << (cluster_stats.p50 * 100.0)
                      << "%  p90=" << (cluster_stats.p90 * 100.0) << "%  p95=" << (cluster_stats.p95 * 100.0)
                      << "%  p99=" << (cluster_stats.p99 * 100.0) << "%\n";
        }
        std::cout << "\n";

        std::cout << std::left << std::setw(14) << "Selectivity" << std::setw(12) << "Mode" << std::right
                  << std::setw(10) << "EF_Search" << std::setw(10) << "Recall" << std::setw(14) << "Latency(ms)"
                  << std::setw(12) << "QPS"
                  << "\n";
        std::cout << std::string(72, '-') << "\n";

        double prev_sel = -1.0;
        for (auto const& row : rows) {
            std::string sel_str;
            if (std::fabs(row.selectivity - prev_sel) > 1e-9) {
                std::ostringstream oss;
                if (row.selectivity >= 0.01)
                    oss << static_cast<int>(std::round(row.selectivity * 100)) << "%";
                else if (row.selectivity >= 0.001)
                    oss << std::fixed << std::setprecision(1) << (row.selectivity * 100) << "%";
                else
                    oss << std::fixed << std::setprecision(2) << (row.selectivity * 100) << "%";
                sel_str = oss.str();
                if (prev_sel >= 0)
                    std::cout << std::string(72, '-') << "\n";
                prev_sel = row.selectivity;
            }
            std::cout << std::left << std::setw(14) << sel_str << std::setw(12) << row.mode << std::right
                      << std::setw(10) << row.ef_search << std::fixed << std::setprecision(4) << std::setw(10)
                      << row.recall << std::setprecision(3) << std::setw(14) << row.latency_ms << std::setprecision(0)
                      << std::setw(12) << row.qps << "\n";
        }

        std::cout << "\n========================================================\n";
        std::cout << " QPS Benchmark complete.\n";
        std::cout << "========================================================\n";
        return 0;
    }

    std::vector<search_result_summary_t> summaries;

    // --modes filter applies to mode selection; --skip-exact is equivalent to removing "exact"
    auto case_enabled = [](test_case_t const& tc) {
        if (SKIP_EXACT && tc.exact)
            return false;
        return mode_matches(tc.mode);
    };
    // Baseline (no-predicate HNSW) only makes sense if no mode filter is active and exact is not skipped.
    // In cluster correlation mode the baseline is meaningless (every node passes), so skip it.
    bool show_baseline = FILTER_MODES.empty() && !SKIP_EXACT && !cluster_mode;

    // Count total search cases
    std::size_t search_total = 0;
    for (auto const& tc : cases) {
        if (!selectivity_matches(tc.selectivity))
            continue;
        if (!case_enabled(tc))
            continue;
        ++search_total;
    }
    if (show_baseline)
        ++search_total;
    if (search_total == 0) {
        std::cerr << "Error: no cases to run. "
                  << "Check --modes / --skip-exact / --selectivity combination "
                  << "(e.g. --modes exact with --skip-exact leaves nothing).\n";
        return 1;
    }
    std::size_t search_done = 0;

    // Baseline: unfiltered search (no predicate)
    if (show_baseline) {
        std::cout << "\n--- [" << (search_done + 1) << "/" << search_total << "] Baseline (no predicate) ---\n";
        dummy_predicate_t no_pred;
        auto baseline_factory = [no_pred](std::size_t /*q*/) { return no_pred; };
        auto gt_base = compute_gt_parallel(baseline_factory);
        summaries.push_back(run_search(index, metric, queries, K, EF_SEARCH, baseline_factory, -1.0, false, false,
                                       "Baseline (no predicate)", 1.0, false, 0.0, gt_base));
        ++search_done;
        std::cout.flush();
    }

    for (auto const& tc : cases) {
        if (!selectivity_matches(tc.selectivity))
            continue;
        if (!case_enabled(tc))
            continue;
        ++search_done;
        std::size_t eligible = static_cast<std::size_t>(N * tc.selectivity);
        std::cout << "\n--- [" << search_done << "/" << search_total << "] " << tc.label << " ---";
        std::cout << "\n  Eligible vectors : ~" << eligible << " / " << N << " (" << std::fixed << std::setprecision(1)
                  << (tc.selectivity * 100.0) << "%)\n";

        std::size_t key = gt_key(tc.modulus, tc.pred_threshold);
        auto run_with = [&](auto&& factory) {
            return run_search(index, metric, queries, K, EF_SEARCH, factory, tc.two_hop, tc.racorn, tc.exact, tc.label,
                              tc.selectivity, tc.acorn, tc.fallback_exact_threshold, gt_cache[key]);
        };
        summaries.push_back(cluster_mode ? run_with(make_cluster_factory(cluster_pos))
                                         : run_with(make_modulo_factory(tc.modulus, tc.pred_threshold)));
        // Persist log immediately after each (selectivity, mode) cell completes — so the
        // file reflects in-progress state if the process is interrupted or hangs later.
        std::cout.flush();
    }

    // ── Summary table ───────────────────────────────────────────────────────
    std::cout << "\n========================================================\n";
    std::cout << " Summary\n";
    std::cout << "========================================================\n\n";

    std::cout << "  Vectors     : " << N << "\n";
    std::cout << "  Dimensions  : " << D << "\n";
    std::cout << "  Connectivity: " << M << "\n";
    std::cout << "  EF construct: " << EF_CONSTRUCTION << "\n";
    std::cout << "  K           : " << K << "\n";
    std::cout << "  EF search   : " << EF_SEARCH << "\n";
    std::cout << "  ACORN thresh: " << ACORN_THRESHOLD << "\n";
    std::cout << "  Bridge ratio: " << BRIDGE_BUDGET_RATIO << "\n";
    if (FALLBACK_EXACT_THRESHOLD > 0)
        std::cout << std::fixed << std::setprecision(4)
                  << "  FB threshold: " << FALLBACK_EXACT_THRESHOLD << "\n";
#if USEARCH_ENABLE_3HOP
    std::cout << "  3hop thresh : " << THREE_HOP_THRESHOLD << "\n";
#endif
    std::cout << "  Metric      : " << metric_label << "\n";
    std::cout << "  Dataset     : " << DATASET_NAME << "\n";
    std::cout << "  Search thr  : " << SEARCH_THREADS << "\n";
    std::cout << "  Index source: " << (index_loaded ? "loaded (" + vol_path + ")" : "built from scratch") << "\n";
    if (cluster_mode) {
        std::cout << "  Correlation : " << (cluster_pos ? "positive" : "negative") << " (cluster K=" << CLUSTER_K
                  << ")\n";
        std::cout << "  Per-q sel.  : mean=" << std::fixed << std::setprecision(4) << (cluster_stats.mean * 100.0)
                  << "%  p50=" << (cluster_stats.p50 * 100.0) << "%  p90=" << (cluster_stats.p90 * 100.0)
                  << "%  p95=" << (cluster_stats.p95 * 100.0) << "%  p99=" << (cluster_stats.p99 * 100.0) << "%\n";
    }
    std::cout << "\n";

    std::cout << std::left << std::setw(14) << "Selectivity" << std::setw(11) << "Mode" << std::right << std::setw(12)
              << "Latency(ms)" << std::setw(14) << "Dist.Computed" << std::setw(14) << "NodesVisited" << std::setw(10)
              << "Recall(avg)" << std::setw(12) << "RecallRatio" << std::setw(10) << "Speedup" << std::setw(10)
              << "DistRatio" << std::setw(12) << "PredEvals" << std::setw(10) << "FBCount" << std::setw(10) << "FBSelec"
              << std::setw(9) << "FBExact"
              << "\n";
    std::cout << std::string(148, '-') << "\n";

    // Extract mode name from a label like "Selectivity 10% | ACORN-1". Falls back to empty.
    auto mode_from_label = [](std::string const& lbl) -> std::string {
        auto p = lbl.rfind(" | ");
        return (p == std::string::npos) ? std::string{} : lbl.substr(p + 3);
    };

    // Shared row printer
    auto print_row = [&](std::string const& sel, std::string const& mode, search_result_summary_t const& r,
                         search_result_summary_t const* hnsw_baseline) {
        std::cout << std::left << std::setw(14) << sel << std::setw(11) << mode << std::right << std::fixed
                  << std::setprecision(1) << std::setw(12) << r.latency_ms << std::setprecision(0) << std::setw(14)
                  << r.dist_computed << std::setw(14) << r.nodes_visited << std::setprecision(4) << std::setw(10)
                  << r.recall;
        if (hnsw_baseline && &r != hnsw_baseline && !r.exact) {
            double sp = hnsw_baseline->latency_ms / r.latency_ms;
            double dr = hnsw_baseline->dist_computed / r.dist_computed;
            double rr = (hnsw_baseline->recall > 0) ? r.recall / hnsw_baseline->recall : 0.0;
            std::cout << std::setprecision(2) << std::setw(11) << rr << "x" << std::setprecision(1) << std::setw(9)
                      << sp << "x" << std::setw(9) << dr << "x";
        } else {
            std::cout << std::setw(12) << "" << std::setw(10) << "" << std::setw(10) << "";
        }
        if (r.predicate_evals > 0)
            std::cout << std::setprecision(0) << std::setw(12) << r.predicate_evals;
        else
            std::cout << std::setw(12) << "";
        if (r.fallback_count > 0 && !std::isnan(r.fallback_selectivity))
            std::cout << std::setprecision(1) << std::setw(10) << r.fallback_count << std::setprecision(4)
                      << std::setw(10) << r.fallback_selectivity;
        else
            std::cout << std::setw(10) << "" << std::setw(10) << "";
        std::cout << std::setw(9) << (r.fallback_to_exact ? "YES" : "");
        std::cout << "\n";
    };

    // Baseline row (no-predicate HNSW), only present when show_baseline was true
    std::size_t group_start = 0;
    if (show_baseline) {
        auto const& b = summaries[0];
        print_row("Baseline", "NoPred", b, nullptr);
        std::cout << std::string(148, '-') << "\n";
        group_start = 1;
    }

    // Group remaining summaries by selectivity (contiguous with same selectivity value)
    for (std::size_t i = group_start; i < summaries.size();) {
        std::size_t j = i;
        double sel_val = summaries[i].selectivity;
        while (j < summaries.size() && std::fabs(summaries[j].selectivity - sel_val) < 1e-12)
            ++j;

        // Find HNSW within [i, j) as the ratio baseline
        search_result_summary_t const* hnsw_p = nullptr;
        for (std::size_t k = i; k < j; ++k) {
            if (iequals(mode_from_label(summaries[k].label), "HNSW")) {
                hnsw_p = &summaries[k];
                break;
            }
        }

        std::ostringstream sel_oss;
        if (sel_val >= 0.01)
            sel_oss << static_cast<int>(std::round(sel_val * 100)) << "%";
        else if (sel_val >= 0.001)
            sel_oss << std::fixed << std::setprecision(1) << (sel_val * 100) << "%";
        else
            sel_oss << std::fixed << std::setprecision(2) << (sel_val * 100) << "%";
        std::string sel_str = sel_oss.str();

        bool first = true;
        for (std::size_t k = i; k < j; ++k) {
            std::string mode = mode_from_label(summaries[k].label);
            print_row(first ? sel_str : std::string{}, mode, summaries[k], hnsw_p);
            first = false;
        }
        if (j < summaries.size())
            std::cout << std::string(148, '-') << "\n";

        i = j;
    }

    std::cout << "\n========================================================\n";
    std::cout << " Benchmark complete.\n";
    std::cout << "========================================================\n";
    return 0;
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Force line/unit buffering so logs are persisted incrementally even when stdout is
    // redirected to a file (default block-buffering hides progress until exit / SIGKILL).
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    parse_args(argc, argv);
    std::cout << "========================================================\n";
    std::cout << " index_gt ACORN-1 Performance Benchmark\n";
    std::cout << "========================================================\n";
    std::cout << "  Vectors     : " << N << "\n";
    std::cout << "  Dimensions  : " << D << "\n";
    std::cout << "  Connectivity: " << M << "\n";
    std::cout << "  EF construct: " << EF_CONSTRUCTION << "\n";
    std::cout << "  K           : " << K << "\n";
    std::cout << "  EF search   : " << EF_SEARCH << "\n";
    std::cout << "  ACORN thresh: " << ACORN_THRESHOLD << "\n";
    std::cout << "  Bridge ratio: " << BRIDGE_BUDGET_RATIO << "\n";
#if USEARCH_ENABLE_3HOP
    std::cout << "  3hop thresh : " << THREE_HOP_THRESHOLD << "\n";
#endif
    std::cout << "  Metric      : " << METRIC_NAME << "\n";
    std::cout << "  Dataset     : " << DATASET_NAME << "\n";
    std::cout << "  HW threads  : " << std::thread::hardware_concurrency() << "\n";
    std::cout << "  Search thr  : " << SEARCH_THREADS << "\n";
    if (QPS_MODE) {
        std::cout << "  Mode        : QPS (ef_search: ";
        for (std::size_t i = 0; i < QPS_EF_VALUES.size(); ++i) {
            if (i)
                std::cout << ",";
            std::cout << QPS_EF_VALUES[i];
        }
        std::cout << ")\n";
    }
    std::cout << "========================================================\n\n";

    // 1. Load or generate vectors
    std::unique_ptr<vector_store_t<float>> store_ptr;
    std::vector<float> query_vecs_data;
    std::size_t num_queries = 0;

    // Load dataset
    if (DATASET_NAME == "sift") {
        std::vector<float> base_data;
        std::size_t base_count = 0, base_dims = 0;
        bool loaded = false;

        std::string base_path = DATA_DIR + "/sift_base.fvecs";
        {
            stopwatch_t load_timer;
            if (load_fvecs(base_path.c_str(), base_data, base_count, base_dims)) {
                std::cout << "Loaded SIFT base vectors from " << base_path << ": " << base_count << " x " << base_dims
                          << "d (" << std::fixed << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
                loaded = true;
            }
        }

        if (loaded && base_dims == D && base_count >= N) {
            if (base_count > N)
                base_data.resize(N * D);
            store_ptr = std::make_unique<vector_store_t<float>>(std::move(base_data), D);
            std::cout << "Using SIFT1M dataset (" << N << " vectors, " << D << "d)\n";

            std::string query_path = DATA_DIR + "/sift_query.fvecs";
            std::size_t q_count = 0, q_dims = 0;
            stopwatch_t load_timer;
            if (load_fvecs(query_path.c_str(), query_vecs_data, q_count, q_dims) && q_dims == D) {
                num_queries = q_count;
                std::cout << "Loaded SIFT query vectors: " << q_count << " x " << q_dims << "d (" << std::fixed
                          << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
            }
        } else {
            if (loaded)
                std::cerr << "SIFT dimensions/count mismatch (got " << base_dims << "d, " << base_count << " vecs)\n";
            else
                std::cerr << "SIFT1M fvecs not found\n";
            return 1;
        }
    } else if (DATASET_NAME == "gist") {
        std::vector<float> base_data;
        std::size_t base_count = 0, base_dims = 0;
        bool loaded = false;

        std::string base_path = DATA_DIR + "/gist_base.fvecs";
        {
            stopwatch_t load_timer;
            if (load_fvecs(base_path.c_str(), base_data, base_count, base_dims)) {
                std::cout << "Loaded GIST base vectors from " << base_path << ": " << base_count << " x " << base_dims
                          << "d (" << std::fixed << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
                loaded = true;
            }
        }

        if (loaded && base_dims == D && base_count >= N) {
            if (base_count > N)
                base_data.resize(N * D);
            store_ptr = std::make_unique<vector_store_t<float>>(std::move(base_data), D);
            std::cout << "Using GIST1M dataset (" << N << " vectors, " << D << "d)\n";

            std::string query_path = DATA_DIR + "/gist_query.fvecs";
            std::size_t q_count = 0, q_dims = 0;
            stopwatch_t load_timer;
            if (load_fvecs(query_path.c_str(), query_vecs_data, q_count, q_dims) && q_dims == D) {
                num_queries = q_count;
                std::cout << "Loaded GIST query vectors: " << q_count << " x " << q_dims << "d (" << std::fixed
                          << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
            }
        } else {
            if (loaded)
                std::cerr << "GIST dimensions/count mismatch (got " << base_dims << "d, " << base_count << " vecs)\n";
            else
                std::cerr << "GIST1M fvecs not found\n";
            return 1;
        }
    } else if (DATASET_NAME == "text2image") {
        std::vector<float> base_data;
        std::size_t base_count = 0, base_dims = 0;
        bool loaded = false;

        std::string base_path = DATA_DIR + "/text2image-40M.fbin";
        {
            stopwatch_t load_timer;
            if (load_fbin(base_path.c_str(), base_data, base_count, base_dims, N)) {
                std::cout << "Loaded text2image base vectors from " << base_path << ": " << base_count << " x "
                          << base_dims << "d (" << std::fixed << std::setprecision(1) << load_timer.elapsed_ms()
                          << " ms)\n";
                loaded = true;
            }
        }

        if (loaded && base_dims == D && base_count >= N) {
            if (base_count > N)
                base_data.resize(N * D);
            store_ptr = std::make_unique<vector_store_t<float>>(std::move(base_data), D);
            std::cout << "Using text2image dataset (" << N << " vectors, " << D << "d)\n";

            std::string query_path = DATA_DIR + "/text2image-query-100K.fbin";
            std::size_t q_count = 0, q_dims = 0;
            stopwatch_t load_timer;
            if (load_fbin(query_path.c_str(), query_vecs_data, q_count, q_dims) && q_dims == D) {
                num_queries = q_count;
                std::cout << "Loaded text2image query vectors: " << q_count << " x " << q_dims << "d (" << std::fixed
                          << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
            }
        } else {
            if (loaded)
                std::cerr << "text2image dimensions/count mismatch (got " << base_dims << "d, " << base_count
                          << " vecs)\n";
            else
                std::cerr << "text2image fbin not found\n";
            return 1;
        }
    }

    if (!store_ptr) {
        std::cerr << "Failed to load dataset '" << DATASET_NAME << "'\n";
        return 1;
    }

    int ret;
    if (METRIC_NAME == "cosine")
        ret = run_benchmark<cosine_metric_t<float, vec_key_t, slot_t>>(
            *store_ptr, num_queries > 0 ? query_vecs_data.data() : nullptr, num_queries, "Cosine");
    else
        ret = run_benchmark<l2_metric_t<float, vec_key_t, slot_t>>(
            *store_ptr, num_queries > 0 ? query_vecs_data.data() : nullptr, num_queries, "L2sq");

    return ret;
}
