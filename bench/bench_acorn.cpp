// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 * @file bench_acorn.cpp
 * @brief Filtered ANN search benchmark over a selectivity grid: exact, prefilter,
 *        HNSW, ACORN-1, RACORN-1 and NaviX modes under one controlled predicate.
 *
 * Defaults:
 *   - Dataset: sift, gist, text2image, laion, yfcc (see --dataset; default sift)
 *   - m=16, ef_construction=100, ef_search=200, k=100
 *   - Selectivity sweep from 100% down to 0.01% (restrict with --selectivity)
 *   - filter_first_threshold default 0.3
 *   - bridge_budget_ratio default 1.0 (used by RACORN-1; ACORN-1 uses 0.0)
 *   - early_termination_time_threshold off by default (--early-termination-ms)
 *   - Modes: exact, prefilter, prefilter-scan, hnsw, acorn-1, acorn-1-3hop, racorn-1,
 *     navix-onehop-s, navix-blind, navix-directed, navix-local, navix-local-paper (see --modes)
 *   - Index construction: multi-threaded
 *   - --build-index writes the index volumes and --build-gt writes the ground-truth files.
 *     A search run loads them when they exist. Otherwise it builds or computes them in memory
 */
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <limits>
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
        if (!ifs.read(reinterpret_cast<char*>(&dim_check), sizeof(dim_check)))
            return false;
        if (static_cast<std::size_t>(dim_check) != dims)
            return false;
        if (!ifs.read(reinterpret_cast<char*>(data.data() + i * dims), dims * sizeof(float)))
            return false;
    }
    return true;
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
    if (!ifs.read(reinterpret_cast<char*>(data.data()), count * dims * sizeof(float)))
        return false;
    return true;
}

// ── u8bin loader (uint8 -> float): [int32 n][int32 d][uint8 n*d] ───────────
// The big-ann-benchmarks format (YFCC): an fbin header with a uint8 body. The values are
// promoted to float on load. They are kept as they are (0-255), so the L2 ranking equals the
// original's.
static bool load_u8bin(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims,
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

    std::vector<std::uint8_t> chunk(dims);
    for (std::size_t i = 0; i < count; ++i) {
        if (!ifs.read(reinterpret_cast<char*>(chunk.data()), dims))
            return false;
        for (std::size_t j = 0; j < dims; ++j)
            data[i * dims + j] = static_cast<float>(chunk[j]);
    }
    return true;
}

// ── L2-normalize a contiguous [n x dims] row-major buffer in-place ──────────
template <typename T> static void normalize_rows_l2(T* data, std::size_t n, std::size_t dims) {
    for (std::size_t i = 0; i < n; ++i) {
        T* v = data + i * dims;
        T norm_sq = 0;
        for (std::size_t d = 0; d < dims; ++d)
            norm_sq += v[d] * v[d];
        T norm = std::sqrt(norm_sq);
        if (norm > static_cast<T>(0)) {
            T inv = static_cast<T>(1) / norm;
            for (std::size_t d = 0; d < dims; ++d)
                v[d] *= inv;
        }
    }
}

// ── Vector store ────────────────────────────────────────────────────────────
template <typename T = float> class vector_store_t {
  public:
    vector_store_t(std::vector<T>&& data, std::size_t dims) : dims_(dims), data_(std::move(data)) {}

    std::size_t size() const noexcept { return data_.size() / dims_; }
    std::size_t dims() const noexcept { return dims_; }
    T const* row(std::size_t i) const noexcept { return data_.data() + i * dims_; }

    // L2-normalize all vectors in-place (unit length)
    void normalize_l2() { normalize_rows_l2(data_.data(), size(), dims_); }

  private:
    std::size_t dims_;
    std::vector<T> data_;
};

// ── Dataset metric (parameterized on kernel) ────────────────────────────────
// One metric adapter for both L2-squared and cosine. `kernel_at` is the actual
// distance functor (e.g. metric_l2sq_gt / metric_cos_gt). The l2_metric_t /
// cosine_metric_t alias templates below give source-compatible names.
template <typename kernel_at, typename T = float, typename key_t = std::size_t, typename slot_t = std::size_t>
struct dataset_metric_t {
    using index_typed_t = index_gt<T, key_t, slot_t>;
    using member_cref_t = typename index_typed_t::member_cref_t;
    using member_citerator_t = typename index_typed_t::member_citerator_t;
    using kernel_t = kernel_at; // raw distance kernel, for index-free (prefilter) scans

    vector_store_t<T> const* store = nullptr;
    std::size_t dimensions = 0;

    T const* row(std::size_t s) const noexcept { return store->row(s); }

    float operator()(member_cref_t const& a, member_cref_t const& b) const {
        return kernel_at{}(row(get_key(b)), row(get_key(a)), dimensions);
    }
    float operator()(T const* q, member_cref_t const& m) const { return kernel_at{}(q, row(get_key(m)), dimensions); }
    float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
        return kernel_at{}(row(get_key(*b)), row(get_key(*a)), dimensions);
    }
    float operator()(T const* q, member_citerator_t const& m) const {
        return kernel_at{}(q, row(get_key(*m)), dimensions);
    }
};

template <typename T = float, typename key_t = std::size_t, typename slot_t = std::size_t>
using l2_metric_t = dataset_metric_t<metric_l2sq_gt<T, float>, T, key_t, slot_t>;

template <typename T = float, typename key_t = std::size_t, typename slot_t = std::size_t>
using cosine_metric_t = dataset_metric_t<metric_cos_gt<T, float>, T, key_t, slot_t>;

// Unnormalized inner product (Text2Image's published metric). The distance is
// `1 - <a,b>`, as in metric_ip_gt. The vectors are NOT normalized: unlike
// cosine, the magnitude is part of the ranking.
template <typename T = float, typename key_t = std::size_t, typename slot_t = std::size_t>
using ip_metric_t = dataset_metric_t<metric_ip_gt<T, float>, T, key_t, slot_t>;

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
// sift covers both sift1m (--n 1000000) and sift100m (--n 100000000, with
// --data-dir pointing at the 100M fvecs). The volume and GT names embed N.
static constexpr std::size_t SIFT_N_MAX = 100'000'000;
static constexpr std::size_t GIST_N_MAX = 1'000'000;
static constexpr std::size_t YFCC_N_MAX = 10'000'000; // big-ann filtered track YFCC (192d, uint8)
static constexpr std::size_t TEXT2IMAGE_N_MAX = 40'000'000;
static constexpr std::size_t LAION_N_MAX = 25'000'000; // LAION-400M CLIP img_emb prefix (512d, pre-normalized)

static double ACORN_THRESHOLD = 0.3;

// Tuneable defaults (overridable via command-line)
static std::size_t N_VAL = 1'000'000;
static std::size_t D_VAL = 128; // set at runtime based on dataset
static std::size_t M_VAL = 16;
static std::size_t K_VAL = 100;
static std::size_t EF_CONSTRUCTION_VAL = 100;
static std::size_t EF_SEARCH_VAL = 200;
static std::string METRIC_NAME = "";      // "l2", "cosine" or "ip" (empty = auto per dataset)
static std::string DATASET_NAME = "sift"; // "sift", "gist", "text2image", "laion" or "yfcc"

// RACORN-1 bridge budget ratio: bridge budget = one_hop_count * BRIDGE_BUDGET_RATIO.
// Passed to index_search_config_t::bridge_budget_ratio for the racorn-1 mode
// (0.0 = plain ACORN-1). --bridge-budget-ratio overrides it (default 1.0).
static double BRIDGE_BUDGET_RATIO = 1.0;

// Stride sampling toggle for the two budget-bounded 2-hop selections: the 2-hop
// passed cap and the RACORN-1 bridge pool. Passed to
// index_search_config_t::stride_sampling. `false` selects prefix truncation,
// the ablation baseline of Section VI-F. --stride on|off overrides it (default on).
static bool STRIDE_SAMPLING = true;

// Base-level average out-degree of the loaded graph, measured with index.stats(0)
// (edges / nodes). The nominal cap is connectivity_base = 2*M. The RNG pruning
// heuristic leaves most nodes below it, so the cost model in the paper needs
// the measured value rather than 2*M. Reported in the JSON config as "mavg"
// and printed in the header.
static double MAVG_BASE = 0.0;

// Soft base-layer time budget in milliseconds. Passed to
// index_search_config_t::early_termination_time_threshold for every non-exact
// mode. Negative (the default) leaves the budget disabled, so the flag can only
// ever bound a search. --early-termination-ms overrides it.
static double EARLY_TERMINATION_MS = -1.0;

static bool early_termination_enabled() { return EARLY_TERMINATION_MS >= 0.0; }

// How full the result heap must be before the soft budget may stop a search. 0 keeps
// the engine default (ef_search). Setting it to k makes the budget the only
// stopping rule beyond k results. Every arm then spends the same wall-clock time
// and is judged on recall (iso-latency mode). --et-min-results.
static std::size_t ET_MIN_RESULTS = 0;

// With an absolute budget ladder (--early-termination-ms-values), the unbudgeted
// row of every series is only a reference. It is also by far the most expensive
// row at large ef. --et-skip-unbudgeted drops it. The flag is ignored when a
// fraction ladder is given, because the fractions are derived from that row's p50.
static bool ET_SKIP_UNBUDGETED = false;

// Soft-timeout ladders for the latency mode. Each (selectivity, mode, br, ef)
// cell first runs with no budget to obtain its own unlimited p50. It then runs
// once per ladder entry. --early-termination-fractions gives budgets as fractions
// of that p50, a relative ladder per mode. An absolute ladder would be a no-op
// for fast modes and a floor for slow ones. --early-termination-ms-values adds
// absolute budgets. Rows record the budget actually used (`early_termination_ms`,
// -1 = off).
static std::vector<double> ET_FRACTIONS;
static std::vector<double> ET_MS_VALUES;

// --et-anchor <mode>@<ef>:<fractions> (repeatable). The budgets are derived from
// the prefill p50 of another series, measured in this same process. That p50 is
// the time at which the series' W first held ef results. It is the earliest
// deadline at which the series could return a full result set (the reference
// deadline of Section VI-E). Every graph series of the same selectivity gets
// anchor_ms × fraction as an absolute budget. So all arms are compared at the same
// wall-clock ms, and the anchor is read in the same process as the rows it governs.
// The ladder rows of a series wait until every anchor of its selectivity has been
// measured. The anchor series must be enabled and must fill W; otherwise the run aborts.
struct et_anchor_t {
    std::string mode;              // matched case-insensitively against the series' display name
    std::size_t ef = 0;            // the anchor series' ef_search
    std::vector<double> fractions; // > 0, may exceed 1
    std::string name;              // "<mode>@<ef>" as given, written to each row's et_anchor
};
static std::vector<et_anchor_t> ET_ANCHORS;

// --mode-ef <mode>:<list> (repeatable): an ef sweep for one mode only. It overrides
// --ef-values / --ef-search for that mode. One process can then run HNSW at ef 100
// and 200 while RACORN-1 runs at ef 200 alone.
static std::vector<std::pair<std::string, std::vector<std::size_t>>> MODE_EF;

static bool et_ladder_enabled() {
    return !ET_FRACTIONS.empty() || !ET_MS_VALUES.empty() || !ET_ANCHORS.empty();
}

// Header-level provenance strings for the JSON config block.
static std::string et_anchors_spec() {
    std::ostringstream os;
    for (std::size_t i = 0; i < ET_ANCHORS.size(); ++i) {
        if (i)
            os << ";";
        os << ET_ANCHORS[i].name << ":";
        for (std::size_t j = 0; j < ET_ANCHORS[i].fractions.size(); ++j)
            os << (j ? "," : "") << std::setprecision(6) << ET_ANCHORS[i].fractions[j];
    }
    return os.str();
}
static std::string mode_ef_spec() {
    std::ostringstream os;
    for (std::size_t i = 0; i < MODE_EF.size(); ++i) {
        if (i)
            os << ";";
        os << MODE_EF[i].first << ":";
        for (std::size_t j = 0; j < MODE_EF[i].second.size(); ++j)
            os << (j ? "," : "") << MODE_EF[i].second[j];
    }
    return os.str();
}

// The engine takes seconds as a double, so a fractional millisecond budget
// reaches it unrounded. This matters for traversals that finish in single-digit
// milliseconds.
static std::chrono::duration<double> early_termination_threshold() {
    return std::chrono::duration<double>(EARLY_TERMINATION_MS / 1000.0);
}

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
static std::string JSON_PATH = "";               // --json <path>: write structured results as JSON
static double BUILD_TIME_MS = -1.0;              // index build time in ms (>=0 if built this run, -1 if loaded)
static bool SKIP_EXACT = false;                  // --skip-exact: skip exact latency benchmarking
static std::vector<std::string> FILTER_MODES;    // --modes: only run these modes (empty = all)
static std::vector<std::size_t> QPS_EF_VALUES = {128, 256, 384, 512};
static bool EF_VALUES_SET = false;               // --ef-values given: latency mode sweeps them too
static std::vector<double> BBR_VALUES;           // --bridge-budget-ratios: racorn-1 runs once per value
static double RECALL_TARGET = -1.0;              // --recall-target: stop an ef sweep once avg recall reaches it
static std::size_t BUILD_THREADS = 0;            // --build-threads: indexing threads (0 = hardware_concurrency)
static double THREE_HOP_TAU = 1.0;               // --three-hop-threshold: acorn-1-3hop trigger (a multiple of M0)
static bool PRUNING_FREE = false;                // --pruning-free: ACORN-paper-spec construction (no RNG pruning)

// ── Control predicate kind ──────────────────────────────────────────────────
// "modulo": key % modulus < threshold (periodic, exact selectivity).
// "random": every key gets a seed-fixed uniform uint32 attribute at load time
//           and passes when attr[key] < selectivity * 2^32. This is a uniform
//           random subset with no key structure (ACORN/NaviX-style random filter).
//           The realized selectivity is the nominal one ± sampling noise.
static std::string PREDICATE_KIND = "modulo";
static std::uint64_t PREDICATE_SEED = 12345;
static std::vector<std::uint32_t> RANDOM_ATTR; // N_VAL entries when PREDICATE_KIND == "random"
static bool predicate_random() { return PREDICATE_KIND == "random"; }
static std::uint32_t attr_threshold_for(double selectivity) {
    double t = selectivity * 4294967296.0;
    if (t >= 4294967295.0)
        return 0xFFFFFFFFu;
    return static_cast<std::uint32_t>(t);
}
// Tag embedded in GT file names so modulo and random GTs coexist.
static std::string predicate_tag() { return predicate_random() ? "_rnd" + std::to_string(PREDICATE_SEED) : ""; }

// Content fingerprint: the size plus FNV-1a of the first and last 1 MiB. It goes into
// the GT file name, so a cached GT is tied to the data it was computed from.
static std::string file_fingerprint(std::string const& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return "missing";
    f.seekg(0, std::ios::end);
    std::uint64_t size = (std::uint64_t)f.tellg();
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&](char const* p, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i)
            h = (h ^ (unsigned char)p[i]) * 1099511628211ULL;
    };
    mix(reinterpret_cast<char const*>(&size), sizeof(size));
    std::vector<char> buf(1u << 20);
    std::size_t n1 = (std::size_t)std::min<std::uint64_t>(size, buf.size());
    f.seekg(0, std::ios::beg);
    f.read(buf.data(), (std::streamsize)n1);
    mix(buf.data(), n1);
    if (size > buf.size()) {
        f.seekg((std::streamoff)(size - buf.size()), std::ios::beg);
        f.read(buf.data(), (std::streamsize)buf.size());
        mix(buf.data(), buf.size());
    }
    std::ostringstream o;
    o << std::hex << std::setfill('0') << std::setw(16) << h;
    return o.str();
}
// Fingerprint of the base vectors this run loads; main() sets it before any GT path is built.
static std::string BASE_FINGERPRINT = "unset";

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

using vec_key_t = std::size_t;
using slot_t = std::size_t;
using index_t = index_gt<float, vec_key_t, slot_t>;

// ── Selectivity-based predicate ─────────────────────────────────────────────
// Passes if key % modulus < threshold. e.g. modulus=10,threshold=1 => 10%, modulus=10,threshold=3 => 30%
struct selectivity_predicate_t {
    std::size_t modulus;
    std::size_t threshold;
    std::uint32_t const* attr = nullptr; // non-null → random-attribute mode
    std::uint32_t attr_threshold = 0;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        auto key = get_key(m);
        if (attr)
            return attr[key] < attr_threshold;
        return (key % modulus) < threshold;
    }
};

// Builds the predicate for one selectivity case under the active predicate kind.
static selectivity_predicate_t make_case_predicate(std::size_t modulus, std::size_t threshold, double selectivity) {
    selectivity_predicate_t p{modulus, threshold};
    if (predicate_random()) {
        p.attr = RANDOM_ATTR.data();
        p.attr_threshold = attr_threshold_for(selectivity);
    }
    return p;
}

// Key-level pass test for the prefilter's passing-key scan. Same rule as the predicate.
static inline bool key_passes(std::size_t key, std::size_t modulus, std::size_t threshold, double selectivity) {
    if (predicate_random())
        return RANDOM_ATTR[key] < attr_threshold_for(selectivity);
    return (key % modulus) < threshold;
}

// ── Key-based neighbor comparator ──────────────────────────────────────────
// Sorts neighbors by vector key (ascending). Passed to index.search() as
// neighbor_cmp, so the search path activates the sorted_data + sort_neighbors
// branches in usearch. The modulo predicate uses (key % modulus < threshold).
// A key-ascending sort therefore lays out the passing slots in contiguous runs
// within each candidate's neighbor list: the predicate-passing prefix tends to
// be denser up front.
struct key_cmp_t {
    template <typename member_like_at> bool operator()(member_like_at&& a, member_like_at&& b) const noexcept {
        return static_cast<vec_key_t>(a.key) < static_cast<vec_key_t>(b.key);
    }
};

// When true, searches pass `key_cmp_t` as neighbor_cmp (key-ascending neighbor
// sort). When false, neighbor_cmp defaults to dummy (no sort). Toggled by the
// `--cmpkey` CLI flag. Default: off.
static bool USE_CMPKEY = false;

// Single search entry point that honors USE_CMPKEY. Both branches return the
// same search_result_t (the return type is independent of neighbor_cmp).
template <typename index_at, typename query_at, typename metric_at, typename config_at, typename predicate_at>
static inline auto bench_search(index_at& index, query_at&& query, std::size_t k, metric_at&& metric, config_at&& cfg,
                                predicate_at&& pred) {
    if (USE_CMPKEY)
        return index.search(query, k, metric, cfg, pred, dummy_prefetch_t{}, dummy_preevaluate_t{}, key_cmp_t{});
    return index.search(query, k, metric, cfg, pred, dummy_prefetch_t{}, dummy_preevaluate_t{});
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

// Recall of the k best entries of `W` as it stood when it first filled, i.e. at the instant
// of `prefill_ms` (search_result_t::fill_at). Scored the same way compute_recall scores the
// final results. NaN when `W` never filled or the search kept no copy.
template <typename result_at>
double compute_fill_recall(result_at const& result, std::vector<vec_key_t> const& gt, std::size_t k) {
    std::size_t const n = result.fill_count;
    if (!n)
        return std::numeric_limits<double>::quiet_NaN();
    std::vector<std::pair<double, vec_key_t>> w;
    w.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto m = result.fill_at(i);
        w.emplace_back(static_cast<double>(m.distance), m.member.key);
    }
    std::size_t const take = std::min(n, k);
    std::partial_sort(w.begin(), w.begin() + take, w.end(),
                      [](std::pair<double, vec_key_t> const& a, std::pair<double, vec_key_t> const& b) {
                          return a.first < b.first;
                      });
    std::size_t limit = std::min({gt.size(), k});
    std::unordered_set<vec_key_t> gt_set(gt.begin(), gt.begin() + limit);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < take; ++i)
        if (gt_set.count(w[i].second))
            ++hits;
    return limit > 0 ? static_cast<double>(hits) / static_cast<double>(limit) : 0.0;
}

// ── Search result summary ───────────────────────────────────────────────────
struct search_result_summary_t {
    std::string label;
    std::string mode; // display name, e.g. "HNSW"
    double selectivity;
    std::size_t ef_search;      // 0 for ef-independent modes (Exact, Prefilter)
    double bridge_budget_ratio; // >0 only for RACORN-1 rows
    bool exact;
    double latency_ms; // median
    double latency_p50;
    double latency_p90;
    double latency_p99;
    double qps; // batch throughput: median over RUNS of nq / batch_wall_time
    double dist_computed;
    double nodes_visited;
    // Per-query visit percentiles (higher = more cycles). The mean alone hides
    // whether the ef/s relation holds query by query (Section IV-D).
    double visited_p01; // 1% of queries below this
    double visited_p10;
    double visited_p50;
    double visited_p90;
    double visited_p99;
    double recall; // avg
    double recall_p50;
    double recall_p90;
    double recall_p99;
    double predicate_evals; // ACORN-1/RACORN-1 only; 0 otherwise
    double fallback_count;  // RACORN-1 only; 0 otherwise
    // Expansion size |expand(u)|: what one EXPAND returns, averaged over a
    // query's cycles and then summarized across queries. `bridge_size` is the
    // RACORN-1 bridge spend on the same cycles (0 for ACORN-1). It is reported
    // apart because bridges do not contribute results. `promoted` is the share
    // that survived the visited-set dedup. All 0 on the HNSW/prefilter/exact paths.
    double expand_size_mean;
    double expand_size_p10;
    double expand_size_p50;
    double expand_size_p90;
    double bridge_size_mean;
    double bridge_size_p10;
    double bridge_size_p50;
    double bridge_size_p90;
    // Bridges the fallback sampled and measured. The radius test rejects the gap
    // to bridge_size_mean, so this is the figure that distance accounting needs.
    double bridge_sampled_mean;
    double promoted_mean;
    // Base-loop cycles alone. `nodes_visited` also carries the upper-layer
    // descent, so only this figure composes with the per-cycle means above.
    double expand_cycles;
    // Share of base-loop cycles whose EXPAND returned nothing.
    double expand_empty_rate;
    // Phase split at the moment W filled, and why the base loop stopped.
    // `prefill` is the stretch while W still held fewer than ef entries.
    double cycles_prefill;
    double cycles_postfill;
    double prefill_expand_mean;
    double prefill_bridge_mean;
    double prefill_fire_rate;
    double postfill_expand_mean;
    double postfill_bridge_mean;
    double postfill_fire_rate;
    // Shares across queries; exactly one stop reason fires per query.
    double stop_radius_rate;
    double stop_queue_rate;
    double stop_budget_rate;
    double never_full_rate;
    // Time from base-layer entry to the moment `W` first held `ef` results.
    // This is the earliest deadline at which this arm could return a full result
    // set. Queries that never filled are excluded; they are `never_full_rate`.
    double prefill_ms_mean;
    double prefill_ms_p50;
    double prefill_ms_p90;
    double prefill_ms_p99;
    // Distance computations and predicate evaluations per base cycle, by phase,
    // plus the entry descent's distance computations (outside the loop).
    double dist_descent;
    double prefill_dist_mean;
    double postfill_dist_mean;
    double prefill_pred_mean;
    double postfill_pred_mean;
    // Share of queries whose base-layer traversal hit the soft time budget.
    // Always 0 while --early-termination-ms is unset.
    double early_term_rate;
    double results_returned;
    std::size_t threads;
    std::size_t num_queries;
    double early_termination_ms = -1.0; // soft budget used for this row; -1 = none
    // Wall time to build the prefilter's passing-key list, i.e. the cost this
    // baseline is granted for free. The control predicate is one attribute
    // column shared by every query, so the list is built once per cell and
    // reused. Reporting it lets a reader amortize it over the query batch.
    // Prefilter rows only; -1 elsewhere.
    double passing_build_ms = -1.0;
    // Provenance of an --et-anchor budget: the anchor series ("hnsw@200") and the
    // fraction of its prefill p50 this row's budget was. Empty / NaN elsewhere.
    // Appended last so the positional initializers above stay valid.
    std::string et_anchor{};
    double et_fraction = std::numeric_limits<double>::quiet_NaN();
    // Measured at the instant `W` first held `ef` entries, the instant of prefill_ms.
    // `recall_at_fill` is the recall of its k best entries (mean over queries). The two p50
    // fields are the per-query totals of base-layer distance computations and predicate
    // evaluations up to then (median over queries). Queries whose `W` never filled are
    // excluded, as for prefill_ms. NaN for rows without a base-layer loop. Appended last for
    // the same reason as et_anchor.
    double recall_at_fill = std::numeric_limits<double>::quiet_NaN();
    double prefill_dist_p50 = std::numeric_limits<double>::quiet_NaN();
    double prefill_pred_p50 = std::numeric_limits<double>::quiet_NaN();
};

// ── Median helper ───────────────────────────────────────────────────────────
// Strips NaN entries first. Callers can then pass per-query stats that include
// NaN sentinels without breaking the strict-weak-ordering requirement of
// nth_element. Returns NaN when every entry is NaN (i.e. nothing to summarize).
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

// ── JSON output ─────────────────────────────────────────────────────────────
static std::string json_escape(std::string const& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o.push_back('\\');
            o.push_back(c);
        } else if (c == '\n')
            o += "\\n";
        else
            o.push_back(c);
    }
    return o;
}

static void write_json_results(std::string const& path, std::vector<search_result_summary_t> const& summaries) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "Failed to open --json path: " << path << "\n";
        return;
    }
    auto num = [](double v) -> std::string {
        if (std::isnan(v))
            return "null";
        std::ostringstream os;
        os << std::setprecision(6) << v;
        return os.str();
    };
    ofs << "{\n  \"config\": {\n";
    ofs << "    \"dataset\": \"" << json_escape(DATASET_NAME) << "\",\n";
    ofs << "    \"n\": " << N_VAL << ", \"d\": " << D_VAL << ", \"m\": " << M_VAL
        << ", \"ef_construction\": " << EF_CONSTRUCTION_VAL << ",\n";
    ofs << "    \"k\": " << K_VAL << ", \"ef_search\": " << EF_SEARCH_VAL << ", \"metric\": \""
        << json_escape(METRIC_NAME) << "\",\n";
    ofs << "    \"filter_first_threshold\": " << num(ACORN_THRESHOLD)
        << ", \"bridge_budget_ratio\": " << num(BRIDGE_BUDGET_RATIO)
        << ", \"stride_sampling\": " << (STRIDE_SAMPLING ? "true" : "false")
        << ", \"mavg\": " << std::fixed << std::setprecision(3) << MAVG_BASE
        << ", \"early_termination_ms\": " << num(EARLY_TERMINATION_MS)
        << ", \"et_min_results\": " << ET_MIN_RESULTS << ",\n";
    ofs << "    \"et_anchors\": \"" << json_escape(et_anchors_spec()) << "\", \"mode_ef\": \""
        << json_escape(mode_ef_spec()) << "\",\n";
    ofs << "    \"warmup\": " << BENCH_WARMUP << ", \"runs\": " << BENCH_RUNS
        << ", \"search_threads\": " << SEARCH_THREADS << ",\n";
    ofs << "    \"cmpkey\": " << (USE_CMPKEY ? "true" : "false") << ", \"build_time_ms\": " << num(BUILD_TIME_MS)
        << ", \"pruning_free\": " << (PRUNING_FREE ? "true" : "false") << ",\n";
    ofs << "    \"predicate\": \"" << json_escape(PREDICATE_KIND) << "\", \"seed\": " << PREDICATE_SEED
        << ", \"three_hop_threshold\": " << num(THREE_HOP_TAU)
        << "\n  },\n  \"results\": [\n";
    for (std::size_t i = 0; i < summaries.size(); ++i) {
        auto const& r = summaries[i];
        ofs << "    {\"label\": \"" << json_escape(r.label) << "\", \"mode\": \"" << json_escape(r.mode)
            << "\", \"selectivity\": " << num(r.selectivity) << ", \"ef_search\": " << r.ef_search
            << ", \"bridge_budget_ratio\": " << num(r.bridge_budget_ratio)
            << ", \"exact\": " << (r.exact ? "true" : "false")
            << ", \"threads\": " << r.threads << ", \"num_queries\": " << r.num_queries
            << ", \"latency_ms\": " << num(r.latency_ms) << ", \"latency_p50\": " << num(r.latency_p50)
            << ", \"latency_p90\": " << num(r.latency_p90) << ", \"latency_p99\": " << num(r.latency_p99)
            << ", \"qps\": " << num(r.qps)
            << ", \"recall\": " << num(r.recall) << ", \"recall_p50\": " << num(r.recall_p50)
            << ", \"recall_p90\": " << num(r.recall_p90) << ", \"recall_p99\": " << num(r.recall_p99)
            << ", \"dist_computed\": " << num(r.dist_computed) << ", \"nodes_visited\": " << num(r.nodes_visited)
            << ", \"visited_p01\": " << num(r.visited_p01) << ", \"visited_p10\": " << num(r.visited_p10)
            << ", \"visited_p50\": " << num(r.visited_p50) << ", \"visited_p90\": " << num(r.visited_p90)
            << ", \"visited_p99\": " << num(r.visited_p99)
            << ", \"predicate_evals\": " << num(r.predicate_evals) << ", \"fallback_count\": " << num(r.fallback_count)
            << ", \"expand_size_mean\": " << num(r.expand_size_mean) << ", \"expand_size_p10\": " << num(r.expand_size_p10)
            << ", \"expand_size_p50\": " << num(r.expand_size_p50) << ", \"expand_size_p90\": " << num(r.expand_size_p90)
            << ", \"bridge_size_mean\": " << num(r.bridge_size_mean) << ", \"bridge_size_p10\": " << num(r.bridge_size_p10)
            << ", \"bridge_size_p50\": " << num(r.bridge_size_p50) << ", \"bridge_size_p90\": " << num(r.bridge_size_p90)
            << ", \"bridge_sampled_mean\": " << num(r.bridge_sampled_mean)
            << ", \"promoted_mean\": " << num(r.promoted_mean)
            << ", \"expand_cycles\": " << num(r.expand_cycles)
            << ", \"expand_empty_rate\": " << num(r.expand_empty_rate)
            << ", \"cycles_prefill\": " << num(r.cycles_prefill)
            << ", \"cycles_postfill\": " << num(r.cycles_postfill)
            << ", \"prefill_expand_mean\": " << num(r.prefill_expand_mean)
            << ", \"prefill_bridge_mean\": " << num(r.prefill_bridge_mean)
            << ", \"prefill_fire_rate\": " << num(r.prefill_fire_rate)
            << ", \"postfill_expand_mean\": " << num(r.postfill_expand_mean)
            << ", \"postfill_bridge_mean\": " << num(r.postfill_bridge_mean)
            << ", \"postfill_fire_rate\": " << num(r.postfill_fire_rate)
            << ", \"stop_radius_rate\": " << num(r.stop_radius_rate)
            << ", \"stop_queue_rate\": " << num(r.stop_queue_rate)
            << ", \"stop_budget_rate\": " << num(r.stop_budget_rate)
            << ", \"never_full_rate\": " << num(r.never_full_rate)
            << ", \"prefill_ms_mean\": " << num(r.prefill_ms_mean)
            << ", \"prefill_ms_p50\": " << num(r.prefill_ms_p50)
            << ", \"prefill_ms_p90\": " << num(r.prefill_ms_p90)
            << ", \"prefill_ms_p99\": " << num(r.prefill_ms_p99)
            << ", \"recall_at_fill\": " << num(r.recall_at_fill)
            << ", \"prefill_dist_p50\": " << num(r.prefill_dist_p50)
            << ", \"prefill_pred_p50\": " << num(r.prefill_pred_p50)
            << ", \"dist_descent\": " << num(r.dist_descent)
            << ", \"prefill_dist_mean\": " << num(r.prefill_dist_mean)
            << ", \"postfill_dist_mean\": " << num(r.postfill_dist_mean)
            << ", \"prefill_pred_mean\": " << num(r.prefill_pred_mean)
            << ", \"postfill_pred_mean\": " << num(r.postfill_pred_mean)
            << ", \"early_term_rate\": " << num(r.early_term_rate)
            << ", \"early_termination_ms\": " << num(r.early_termination_ms)
            << ", \"et_anchor\": " << (r.et_anchor.empty() ? std::string("null") : "\"" + json_escape(r.et_anchor) + "\"")
            << ", \"et_fraction\": " << num(r.et_fraction)
            << ", \"passing_build_ms\": " << num(r.passing_build_ms)
            << ", \"results_returned\": " << num(r.results_returned) << "}" << (i + 1 < summaries.size() ? "," : "")
            << "\n";
    }
    ofs << "  ]\n}\n";
    std::cout << "Wrote JSON results to " << path << " (" << summaries.size() << " rows)\n";
}

// ── Persistent worker pool ──────────────────────────────────────────────────
// Pre-spawns `st` worker threads. They wait on a condition variable for each
// new batch. Each batch dispatches `fn(q, tid)` over the query indices
// [chunk*tid, chunk*tid+chunk) for tid in [0, st). So no thread is spawned
// or joined inside a timed batch.
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
// `make_predicate(query_idx)` is invoked per query. The caller's factory returns
// the same fixed predicate object for every query (see make_modulo_factory).
// `bridge_budget_ratio` selects ACORN-1 (0.0) vs RACORN-1 (>0.0).
template <typename metric_at, typename make_predicate_at>
search_result_summary_t run_search(index_t& index, metric_at& metric, std::vector<float const*> const& queries,
                                   std::size_t k, std::size_t ef_search, make_predicate_at&& make_predicate,
                                   double filter_first_threshold, bool is_exact, char const* label, char const* mode,
                                   double selectivity, double bridge_budget_ratio,
                                   std::vector<std::vector<vec_key_t>> const& ground_truths,
                                   std::size_t navix_heuristic = 0, double early_ms = EARLY_TERMINATION_MS,
                                   double three_hop = 0.0) {

    index_search_config_t cfg;
    cfg.expansion = ef_search;
    cfg.filter_first_threshold = filter_first_threshold;
    cfg.exact = is_exact;
    cfg.bridge_budget_ratio = bridge_budget_ratio;
    cfg.stride_sampling = STRIDE_SAMPLING;
    cfg.navix_heuristic = navix_heuristic;
    cfg.three_hop_threshold = three_hop;
    cfg.early_termination_time_threshold = std::chrono::duration<double>(early_ms >= 0.0 ? early_ms / 1000.0 : -1.0);
    cfg.early_termination_min_results = ET_MIN_RESULTS;
    // Keep a copy of W at the instant it fills, for recall_at_fill: one copy of ef entries per
    // query, taken right after the prefill_ms clock read.
    cfg.fill_snapshot = true;

    std::size_t nq = queries.size();
    std::size_t st = std::min(SEARCH_THREADS, nq);
    int WARMUP = static_cast<int>(BENCH_WARMUP);
    int RUNS = static_cast<int>(BENCH_RUNS);

    // Persistent worker pool: spawned once per run_search and reused across all
    // WARMUP/RUNS iterations. This avoids per-iteration spawn/join overhead and
    // keeps thread/CPU placement more stable between iterations.
    query_pool_t pool(st, nq);
    auto parallel_queries = [&](auto&& fn) { pool.run(std::forward<decltype(fn)>(fn)); };

    for (int i = 0; i < WARMUP; ++i) {
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            auto r = bench_search(index, queries[q], k, metric, c, pred);
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
    std::vector<std::vector<double>> per_q_pred_evals(nq);
    std::vector<std::vector<double>> per_q_fallback(nq);
    std::vector<std::vector<double>> per_q_early_term(nq);
    // Per-cycle means for this query: sum / cycles. NaN when the query ran no
    // ACORN cycle (HNSW path, exact). median() then drops it instead of pulling
    // the summary toward 0.
    std::vector<std::vector<double>> per_q_expand(nq);
    std::vector<std::vector<double>> per_q_bridge(nq);
    std::vector<std::vector<double>> per_q_bridge_sampled(nq);
    std::vector<std::vector<double>> per_q_expand_cycles(nq);
    std::vector<std::vector<double>> per_q_expand_empty(nq);
    std::vector<std::vector<double>> per_q_cyc_pre(nq), per_q_cyc_post(nq);
    std::vector<std::vector<double>> per_q_pre_exp(nq), per_q_pre_br(nq), per_q_pre_fire(nq);
    std::vector<std::vector<double>> per_q_post_exp(nq), per_q_post_br(nq), per_q_post_fire(nq);
    std::vector<std::vector<double>> per_q_stop_r(nq), per_q_stop_q(nq), per_q_stop_b(nq), per_q_nfull(nq);
    std::vector<std::vector<double>> per_q_prefill_ms(nq);
    std::vector<std::vector<double>> per_q_fill_recall(nq), per_q_fill_dist(nq), per_q_fill_pred(nq);
    std::vector<std::vector<double>> per_q_desc(nq), per_q_pre_dist(nq), per_q_post_dist(nq), per_q_pre_pred(nq), per_q_post_pred(nq);
    std::vector<std::vector<double>> per_q_promoted(nq);

    for (std::size_t q = 0; q < nq; ++q) {
        per_q_ms[q].reserve(RUNS);
        per_q_distances[q].reserve(RUNS);
        per_q_visited[q].reserve(RUNS);
        per_q_recall[q].reserve(RUNS);
        per_q_result_count[q].reserve(RUNS);
        per_q_pred_evals[q].reserve(RUNS);
        per_q_fallback[q].reserve(RUNS);
        per_q_early_term[q].reserve(RUNS);
        per_q_expand[q].reserve(RUNS);
        per_q_bridge[q].reserve(RUNS);
        per_q_bridge_sampled[q].reserve(RUNS);
        per_q_expand_cycles[q].reserve(RUNS);
        per_q_expand_empty[q].reserve(RUNS);
        for (auto* v : {&per_q_cyc_pre, &per_q_cyc_post, &per_q_pre_exp, &per_q_pre_br, &per_q_pre_fire,
                        &per_q_post_exp, &per_q_post_br, &per_q_post_fire, &per_q_stop_r, &per_q_stop_q,
                        &per_q_stop_b, &per_q_nfull, &per_q_prefill_ms, &per_q_desc, &per_q_pre_dist, &per_q_post_dist,
                        &per_q_pre_pred, &per_q_post_pred, &per_q_fill_recall, &per_q_fill_dist, &per_q_fill_pred})
            (*v)[q].reserve(RUNS);
        per_q_promoted[q].reserve(RUNS);
    }

    // Each measurement run is a full nq batch, so timing the whole batch here
    // yields the QPS figure for free. The per-query stopwatch bookkeeping adds
    // sub-microsecond overhead per query, negligible against ms-scale queries.
    std::vector<double> batch_qps;
    batch_qps.reserve(RUNS);
    for (int i = 0; i < RUNS; ++i) {
        stopwatch_t batch_timer;
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            stopwatch_t t;
            auto r = bench_search(index, queries[q], k, metric, c, pred);
            per_q_ms[q].push_back(t.elapsed_ms());
            per_q_result_count[q].push_back(static_cast<double>(r.size()));
            per_q_distances[q].push_back(static_cast<double>(r.computed_distances));
            per_q_visited[q].push_back(static_cast<double>(r.visited_members));
            per_q_pred_evals[q].push_back(static_cast<double>(r.predicate_evaluations));
            per_q_fallback[q].push_back(static_cast<double>(r.fallback_count));
            double const cycles = static_cast<double>(r.expansion_cycles);
            per_q_expand_cycles[q].push_back(cycles);
            per_q_expand_empty[q].push_back(cycles > 0 ? static_cast<double>(r.expansion_empty_cycles) / cycles : NAN);
            double const cpre = static_cast<double>(r.cycles_prefill);
            double const cpost = cycles - cpre;
            per_q_cyc_pre[q].push_back(cpre);
            per_q_cyc_post[q].push_back(cpost);
            per_q_pre_exp[q].push_back(cpre > 0 ? static_cast<double>(r.expand_prefill) / cpre : NAN);
            per_q_pre_br[q].push_back(cpre > 0 ? static_cast<double>(r.bridge_prefill) / cpre : NAN);
            per_q_pre_fire[q].push_back(cpre > 0 ? static_cast<double>(r.fallback_prefill) / cpre : NAN);
            per_q_post_exp[q].push_back(cpost > 0 ? static_cast<double>(r.expansion_size_sum - r.expand_prefill) / cpost : NAN);
            per_q_post_br[q].push_back(cpost > 0 ? static_cast<double>(r.bridge_size_sum - r.bridge_prefill) / cpost : NAN);
            per_q_post_fire[q].push_back(cpost > 0 ? static_cast<double>(r.fallback_count - r.fallback_prefill) / cpost : NAN);
            per_q_stop_r[q].push_back(static_cast<double>(r.stop_radius));
            per_q_stop_q[q].push_back(static_cast<double>(r.stop_queue_empty));
            per_q_stop_b[q].push_back(static_cast<double>(r.stop_budget));
            per_q_nfull[q].push_back(static_cast<double>(r.never_full));
            per_q_prefill_ms[q].push_back(r.prefill_ms); // NaN when W never filled
            per_q_fill_recall[q].push_back(compute_fill_recall(r, ground_truths[q], k));
            per_q_fill_dist[q].push_back(r.never_full ? NAN : static_cast<double>(r.dist_prefill));
            per_q_fill_pred[q].push_back(r.never_full ? NAN : static_cast<double>(r.pred_prefill));
            per_q_desc[q].push_back(static_cast<double>(r.computed_distances - r.dist_base));
            per_q_pre_dist[q].push_back(cpre > 0 ? static_cast<double>(r.dist_prefill) / cpre : NAN);
            per_q_post_dist[q].push_back(cpost > 0 ? static_cast<double>(r.dist_base - r.dist_prefill) / cpost : NAN);
            per_q_pre_pred[q].push_back(cpre > 0 ? static_cast<double>(r.pred_prefill) / cpre : NAN);
            per_q_post_pred[q].push_back(cpost > 0 ? static_cast<double>(r.pred_base - r.pred_prefill) / cpost : NAN);
            per_q_expand[q].push_back(cycles > 0 ? static_cast<double>(r.expansion_size_sum) / cycles : NAN);
            per_q_bridge[q].push_back(cycles > 0 ? static_cast<double>(r.bridge_size_sum) / cycles : NAN);
            per_q_bridge_sampled[q].push_back(cycles > 0 ? static_cast<double>(r.bridge_sampled_sum) / cycles : NAN);
            per_q_promoted[q].push_back(cycles > 0 ? static_cast<double>(r.promoted_count) / cycles : NAN);
            per_q_early_term[q].push_back(r.early_terminated ? 1.0 : 0.0);
            per_q_recall[q].push_back(r ? compute_recall(r, ground_truths[q], k) : 0.0);
        });
        batch_qps.push_back(static_cast<double>(nq) / (batch_timer.elapsed_ms() / 1000.0));
    }
    double med_qps = median(batch_qps);

    // For each query, compute its median across RUNS, then take median of those across queries
    std::vector<double> q_medians_ms(nq), q_medians_dist(nq), q_medians_visit(nq);
    std::vector<double> q_medians_recall(nq), q_medians_result(nq);
    std::vector<double> q_medians_pred_evals(nq), q_medians_fallback(nq), q_medians_early_term(nq);
    std::vector<double> q_medians_expand(nq), q_medians_bridge(nq), q_medians_promoted(nq);
    std::vector<double> q_medians_bridge_sampled(nq), q_medians_expand_cycles(nq), q_medians_expand_empty(nq);
    std::vector<double> qm_cyc_pre(nq), qm_cyc_post(nq), qm_pre_exp(nq), qm_pre_br(nq), qm_pre_fire(nq),
        qm_post_exp(nq), qm_post_br(nq), qm_post_fire(nq), qm_stop_r(nq), qm_stop_q(nq), qm_stop_b(nq), qm_nfull(nq),
        qm_desc(nq), qm_pre_dist(nq), qm_post_dist(nq), qm_pre_pred(nq), qm_post_pred(nq), qm_prefill_ms(nq),
        qm_fill_recall(nq), qm_fill_dist(nq), qm_fill_pred(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        q_medians_ms[q] = median(per_q_ms[q]);
        q_medians_dist[q] = median(per_q_distances[q]);
        q_medians_visit[q] = median(per_q_visited[q]);
        q_medians_recall[q] = median(per_q_recall[q]);
        q_medians_result[q] = median(per_q_result_count[q]);
        q_medians_pred_evals[q] = median(per_q_pred_evals[q]);
        q_medians_fallback[q] = median(per_q_fallback[q]);
        q_medians_early_term[q] = median(per_q_early_term[q]);
        q_medians_expand[q] = median(per_q_expand[q]);
        q_medians_bridge[q] = median(per_q_bridge[q]);
        q_medians_bridge_sampled[q] = median(per_q_bridge_sampled[q]);
        q_medians_expand_cycles[q] = median(per_q_expand_cycles[q]);
        q_medians_expand_empty[q] = median(per_q_expand_empty[q]);
        qm_cyc_pre[q] = median(per_q_cyc_pre[q]);
        qm_cyc_post[q] = median(per_q_cyc_post[q]);
        qm_pre_exp[q] = median(per_q_pre_exp[q]);
        qm_pre_br[q] = median(per_q_pre_br[q]);
        qm_pre_fire[q] = median(per_q_pre_fire[q]);
        qm_post_exp[q] = median(per_q_post_exp[q]);
        qm_post_br[q] = median(per_q_post_br[q]);
        qm_post_fire[q] = median(per_q_post_fire[q]);
        qm_stop_r[q] = median(per_q_stop_r[q]);
        qm_stop_q[q] = median(per_q_stop_q[q]);
        qm_stop_b[q] = median(per_q_stop_b[q]);
        qm_nfull[q] = median(per_q_nfull[q]);
        qm_prefill_ms[q] = median(per_q_prefill_ms[q]);
        qm_fill_recall[q] = median(per_q_fill_recall[q]);
        qm_fill_dist[q] = median(per_q_fill_dist[q]);
        qm_fill_pred[q] = median(per_q_fill_pred[q]);
        qm_desc[q] = median(per_q_desc[q]);
        qm_pre_dist[q] = median(per_q_pre_dist[q]);
        qm_post_dist[q] = median(per_q_post_dist[q]);
        qm_pre_pred[q] = median(per_q_pre_pred[q]);
        qm_post_pred[q] = median(per_q_post_pred[q]);
        q_medians_promoted[q] = median(per_q_promoted[q]);
    }

    double med_ms = median(q_medians_ms);
    double med_distances = median(q_medians_dist);
    double med_visited = median(q_medians_visit);
    double med_result_count = median(q_medians_result);
    double med_pred_evals = median(q_medians_pred_evals);
    double med_fallback = median(q_medians_fallback);
    // Mean, not median. The interesting figure is the share of queries that
    // stopped early, and a median over 0/1 values would only ever read 0 or 1.
    double early_term_rate = 0.0;
    for (double e : q_medians_early_term)
        early_term_rate += e;
    early_term_rate /= static_cast<double>(q_medians_early_term.size());

    // Latency percentiles across per-query medians (higher = slower).
    std::vector<double> ms_sorted(q_medians_ms);
    std::sort(ms_sorted.begin(), ms_sorted.end());
    double p50_ms = percentile_sorted(ms_sorted, 0.50);
    double p90_ms = percentile_sorted(ms_sorted, 0.90);
    double p99_ms = percentile_sorted(ms_sorted, 0.99);

    // Visit percentiles across per-query medians (higher = more cycles).
    std::vector<double> visit_sorted(q_medians_visit);
    std::sort(visit_sorted.begin(), visit_sorted.end());
    double p01_visit = percentile_sorted(visit_sorted, 0.01);
    double p10_visit = percentile_sorted(visit_sorted, 0.10);
    double p50_visit = percentile_sorted(visit_sorted, 0.50);
    double p90_visit = percentile_sorted(visit_sorted, 0.90);
    double p99_visit = percentile_sorted(visit_sorted, 0.99);

    // Expansion size across per-query means. NaN entries (queries that ran no
    // ACORN cycle) are dropped rather than counted as 0. A row with no ACORN
    // cycle at all reports 0 for every field.
    auto summarize = [&](std::vector<double> const& per_query, double& mean, double& p10, double& p50, double& p90) {
        std::vector<double> v;
        v.reserve(per_query.size());
        for (double x : per_query)
            if (!std::isnan(x))
                v.push_back(x);
        if (v.empty()) {
            mean = p10 = p50 = p90 = 0.0;
            return;
        }
        double sum = 0.0;
        for (double x : v)
            sum += x;
        mean = sum / static_cast<double>(v.size());
        std::sort(v.begin(), v.end());
        p10 = percentile_sorted(v, 0.10);
        p50 = percentile_sorted(v, 0.50);
        p90 = percentile_sorted(v, 0.90);
    };
    double expand_mean, expand_p10, expand_p50, expand_p90;
    double bridge_mean, bridge_p10, bridge_p50, bridge_p90;
    double promoted_mean, promoted_p10, promoted_p50, promoted_p90;
    double bridge_sampled_mean, bs_p10, bs_p50, bs_p90;
    summarize(q_medians_bridge_sampled, bridge_sampled_mean, bs_p10, bs_p50, bs_p90);
    double med_expand_cycles = median(q_medians_expand_cycles);
    double med_cyc_pre = median(qm_cyc_pre), med_cyc_post = median(qm_cyc_post);
    double pre_exp_mean, pre_br_mean, post_exp_mean, post_br_mean, pre_fire, post_fire, d10, d50, d90;
    summarize(qm_pre_exp, pre_exp_mean, d10, d50, d90);
    summarize(qm_pre_br, pre_br_mean, d10, d50, d90);
    summarize(qm_pre_fire, pre_fire, d10, d50, d90);
    summarize(qm_post_exp, post_exp_mean, d10, d50, d90);
    summarize(qm_post_br, post_br_mean, d10, d50, d90);
    summarize(qm_post_fire, post_fire, d10, d50, d90);
    double stop_r_rate, stop_q_rate, stop_b_rate, nfull_rate;
    summarize(qm_stop_r, stop_r_rate, d10, d50, d90);
    summarize(qm_stop_q, stop_q_rate, d10, d50, d90);
    summarize(qm_stop_b, stop_b_rate, d10, d50, d90);
    summarize(qm_nfull, nfull_rate, d10, d50, d90);
    // Prefill time over the queries that actually reached `ef`. The rest are
    // NaN and dropped; their share is `never_full_rate`.
    // NaN, not 0: a cell where `W` never filled has an empty population, and 0 would
    // read as "filled instantly". The JSON writer turns NaN into null.
    double prefill_ms_mean = std::numeric_limits<double>::quiet_NaN(), prefill_ms_p50 = prefill_ms_mean,
           prefill_ms_p90 = prefill_ms_mean, prefill_ms_p99 = prefill_ms_mean;
    {
        std::vector<double> v;
        v.reserve(qm_prefill_ms.size());
        for (double x : qm_prefill_ms)
            if (!std::isnan(x))
                v.push_back(x);
        if (!v.empty()) {
            double sum = 0.0;
            for (double x : v)
                sum += x;
            prefill_ms_mean = sum / static_cast<double>(v.size());
            std::sort(v.begin(), v.end());
            prefill_ms_p50 = percentile_sorted(v, 0.50);
            prefill_ms_p90 = percentile_sorted(v, 0.90);
            prefill_ms_p99 = percentile_sorted(v, 0.99);
        }
    }
    double med_desc = median(qm_desc);
    double pre_dist_mean, post_dist_mean, pre_pred_mean, post_pred_mean;
    summarize(qm_pre_dist, pre_dist_mean, d10, d50, d90);
    summarize(qm_post_dist, post_dist_mean, d10, d50, d90);
    summarize(qm_pre_pred, pre_pred_mean, d10, d50, d90);
    summarize(qm_post_pred, post_pred_mean, d10, d50, d90);
    double expand_empty_rate, ee_p10, ee_p50, ee_p90;
    summarize(q_medians_expand_empty, expand_empty_rate, ee_p10, ee_p50, ee_p90);
    (void)ee_p10;
    (void)ee_p50;
    (void)ee_p90;
    (void)bs_p10;
    (void)bs_p50;
    (void)bs_p90;
    summarize(q_medians_expand, expand_mean, expand_p10, expand_p50, expand_p90);
    summarize(q_medians_bridge, bridge_mean, bridge_p10, bridge_p50, bridge_p90);
    summarize(q_medians_promoted, promoted_mean, promoted_p10, promoted_p50, promoted_p90);
    (void)promoted_p10;
    (void)promoted_p50;
    (void)promoted_p90;

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
    if (med_pred_evals > 0)
        std::cout << "  Med predicate evals : " << std::fixed << std::setprecision(0) << med_pred_evals << "\n";
    if (early_ms >= 0.0)
        std::cout << "  Early-terminated    : " << std::fixed << std::setprecision(1) << (early_term_rate * 100.0)
                  << "% of queries (budget " << std::setprecision(3) << early_ms << " ms)\n";

    double recall_at_fill = std::numeric_limits<double>::quiet_NaN();
    {
        double sum = 0.0;
        std::size_t n = 0;
        for (double x : qm_fill_recall)
            if (!std::isnan(x)) {
                sum += x;
                ++n;
            }
        if (n)
            recall_at_fill = sum / static_cast<double>(n);
    }
    if (!std::isnan(recall_at_fill))
        std::cout << "  Recall at fill      : avg=" << std::fixed << std::setprecision(4) << recall_at_fill
                  << " (k best of W when it first held ef entries)\n";

    search_result_summary_t out = {label,      mode,        selectivity,    is_exact ? std::size_t(0) : ef_search,
            bridge_budget_ratio,     is_exact,
            med_ms,     p50_ms,      p90_ms,         p99_ms,     med_qps,             med_distances,
            med_visited,             p01_visit,      p10_visit,  p50_visit,  p90_visit, p99_visit,
            avg_recall,              p50_recall,     p90_recall, p99_recall,
            med_pred_evals,          med_fallback,
            expand_mean,             expand_p10,     expand_p50, expand_p90,
            bridge_mean,             bridge_p10,     bridge_p50, bridge_p90,
            bridge_sampled_mean,     promoted_mean,  med_expand_cycles, expand_empty_rate,
            med_cyc_pre,             med_cyc_post,
            pre_exp_mean,            pre_br_mean,    pre_fire,
            post_exp_mean,           post_br_mean,   post_fire,
            stop_r_rate,             stop_q_rate,    stop_b_rate,    nfull_rate,
            prefill_ms_mean,         prefill_ms_p50, prefill_ms_p90, prefill_ms_p99,
            med_desc,                pre_dist_mean,  post_dist_mean, pre_pred_mean, post_pred_mean,
            early_term_rate,         med_result_count,
            st,         nq};
    out.recall_at_fill = recall_at_fill;
    out.prefill_dist_p50 = median(qm_fill_dist);
    out.prefill_pred_p50 = median(qm_fill_pred);
    return out;
}

// ── Prefilter (pre-filtering + exact scan) baseline ─────────────────────────
// Models the classic pre-filtering strategy. The set of predicate-passing keys
// is assumed to come from an attribute index, so it is collected once outside
// the timed region. The timed portion is the exact top-k scan over that set.
// Recall is 1.0 by construction (up to distance ties at the k-th boundary).
// Latency is independent of ef_search, so there is one row per selectivity.
template <typename metric_at>
search_result_summary_t run_prefilter_search(metric_at& metric, std::vector<float const*> const& queries,
                                             std::size_t k, std::size_t modulus, std::size_t pred_threshold,
                                             char const* label, char const* mode, double selectivity,
                                             std::vector<std::vector<vec_key_t>> const& ground_truths,
                                             std::size_t ef_limit = 0, double early_ms = -1.0,
                                             bool per_query_scan = false) {
    using kernel_t = typename metric_at::kernel_t;
    std::size_t const dims = metric.dimensions;
    // The soft timeout for the exact scan mirrors the graph gate. The scan stops
    // only once the budget is spent AND the heap holds `ef_limit` (= ef_search)
    // candidates. So the returned top-k always comes from at least ef_search
    // scanned matches. The clock is polled every 64 keys to keep it off the inner loop.
    std::size_t const heap_cap = std::max(k, ef_limit);
    bool const soft_timeout = early_ms >= 0.0 && ef_limit > 0;
    std::chrono::duration<double, std::milli> const budget{early_ms};

    // Passing-key list: one pass over the key space, outside the timed region.
    // It is timed separately and reported as `passing_build_ms`; see the field comment.
    // `per_query_scan` (mode prefilter-scan) rebuilds the list inside every query
    // instead, so the attribute pass lands in the measured region. The shared list
    // is then left empty and `passing_build_ms` reports -1.
    stopwatch_t passing_timer;
    std::vector<vec_key_t> passing;
    double passing_build_ms = -1.0;
    if (!per_query_scan) {
        passing.reserve(static_cast<std::size_t>(static_cast<double>(N_VAL) * selectivity * 1.05) + 16);
        for (std::size_t i = 0; i < N_VAL; ++i)
            if (key_passes(i, modulus, pred_threshold, selectivity))
                passing.push_back(static_cast<vec_key_t>(i));
        passing_build_ms = passing_timer.elapsed_ms();
    }
    std::size_t const expected_passing =
        per_query_scan ? static_cast<std::size_t>(static_cast<double>(N_VAL) * selectivity) : passing.size();

    std::size_t nq = queries.size();
    std::size_t st = std::min(SEARCH_THREADS, nq);
    int WARMUP = static_cast<int>(BENCH_WARMUP);
    int RUNS = static_cast<int>(BENCH_RUNS);

    // Exact top-k over the passing set: max-heap of (distance, key), then
    // ascending-by-distance key list for recall.
    // Returns the top-k keys (ascending distance) and whether the soft timeout
    // cut the scan short.
    auto scan_one = [&](std::size_t q) {
        using pair_t = std::pair<float, vec_key_t>;
        std::vector<pair_t> heap;
        heap.reserve(heap_cap + 1);
        float const* query = queries[q];
        bool terminated = false;
        auto const started = std::chrono::steady_clock::now();
        std::size_t scanned = 0;
        // prefilter-scan: one pass over the attribute column per query, then the
        // exact scan over what it collected. This is the cost a system pays when
        // the predicate is not known ahead of the query.
        std::vector<vec_key_t> local;
        if (per_query_scan) {
            local.reserve(static_cast<std::size_t>(static_cast<double>(N_VAL) * selectivity * 1.05) + 16);
            for (std::size_t i = 0; i < N_VAL; ++i)
                if (key_passes(i, modulus, pred_threshold, selectivity))
                    local.push_back(static_cast<vec_key_t>(i));
        }
        std::vector<vec_key_t> const& scan_set = per_query_scan ? local : passing;
        for (vec_key_t key : scan_set) {
            float dist = kernel_t{}(query, metric.row(key), dims);
            if (heap.size() < heap_cap) {
                heap.emplace_back(dist, key);
                std::push_heap(heap.begin(), heap.end());
            } else if (dist < heap.front().first) {
                std::pop_heap(heap.begin(), heap.end());
                heap.back() = pair_t(dist, key);
                std::push_heap(heap.begin(), heap.end());
            }
            if (soft_timeout && (++scanned & 63) == 0 && heap.size() >= ef_limit &&
                std::chrono::steady_clock::now() - started >= budget) {
                terminated = true;
                break;
            }
        }
        std::sort_heap(heap.begin(), heap.end());
        std::vector<vec_key_t> keys;
        keys.reserve(std::min(heap.size(), k));
        for (std::size_t i = 0; i < heap.size() && i < k; ++i)
            keys.push_back(heap[i].second);
        return std::make_pair(std::move(keys), terminated);
    };

    query_pool_t pool(st, nq);
    auto parallel_queries = [&](auto&& fn) { pool.run(std::forward<decltype(fn)>(fn)); };

    for (int i = 0; i < WARMUP; ++i)
        parallel_queries([&](std::size_t q, std::size_t) { (void)scan_one(q); });

    std::vector<std::vector<double>> per_q_ms(nq);
    std::vector<double> q_recall(nq, 0.0);
    std::vector<double> q_early(nq, 0.0);
    for (std::size_t q = 0; q < nq; ++q)
        per_q_ms[q].reserve(RUNS);

    std::vector<double> batch_qps;
    batch_qps.reserve(RUNS);
    for (int i = 0; i < RUNS; ++i) {
        stopwatch_t batch_timer;
        parallel_queries([&](std::size_t q, std::size_t) {
            stopwatch_t t;
            auto scanned = scan_one(q);
            auto const& keys = scanned.first;
            per_q_ms[q].push_back(t.elapsed_ms());
            if (i == 0) {
                q_early[q] = scanned.second ? 1.0 : 0.0;
                auto const& gt = ground_truths[q];
                std::size_t limit = std::min(gt.size(), k);
                std::unordered_set<vec_key_t> gt_set(gt.begin(), gt.begin() + limit);
                std::size_t hits = 0;
                for (std::size_t r = 0; r < keys.size() && r < k; ++r)
                    if (gt_set.count(keys[r]))
                        ++hits;
                q_recall[q] = limit > 0 ? static_cast<double>(hits) / static_cast<double>(limit) : 0.0;
            }
        });
        batch_qps.push_back(static_cast<double>(nq) / (batch_timer.elapsed_ms() / 1000.0));
    }
    double med_qps = median(batch_qps);

    std::vector<double> q_medians_ms(nq);
    for (std::size_t q = 0; q < nq; ++q)
        q_medians_ms[q] = median(per_q_ms[q]);
    double med_ms = median(q_medians_ms);
    std::vector<double> ms_sorted(q_medians_ms);
    std::sort(ms_sorted.begin(), ms_sorted.end());
    double p50_ms = percentile_sorted(ms_sorted, 0.50);
    double p90_ms = percentile_sorted(ms_sorted, 0.90);
    double p99_ms = percentile_sorted(ms_sorted, 0.99);

    double avg_recall = 0.0;
    for (double r : q_recall)
        avg_recall += r;
    avg_recall /= static_cast<double>(nq);
    std::vector<double> recall_sorted(q_recall);
    std::sort(recall_sorted.begin(), recall_sorted.end());
    double p50_recall = percentile_sorted(recall_sorted, 0.50);
    double p90_recall = percentile_sorted(recall_sorted, 0.10);
    double p99_recall = percentile_sorted(recall_sorted, 0.01);

    double results_returned = static_cast<double>(std::min<std::size_t>(k, expected_passing));
    double early_term_rate = 0.0;
    for (double e : q_early)
        early_term_rate += e;
    early_term_rate /= static_cast<double>(nq);

    std::cout << "\n[" << label << "]\n";
    std::cout << "  Queries             : " << nq << " x " << RUNS << " runs (" << WARMUP << " warmup, " << st
              << " threads)\n";
    std::cout << "  Passing keys        : " << passing.size() << " / " << N_VAL << "\n";
    if (soft_timeout)
        std::cout << "  Early-terminated    : " << std::fixed << std::setprecision(1) << (early_term_rate * 100.0)
                  << "% of queries (budget " << std::setprecision(3) << early_ms << " ms, heap " << heap_cap << ")\n";
    std::cout << "  Recall@" << k << std::string(13 - std::to_string(k).size(), ' ') << ": "
              << "avg=" << std::fixed << std::setprecision(4) << avg_recall << "\n";
    std::cout << "  Med latency         : " << std::fixed << std::setprecision(3) << med_ms << " ms\n";
    std::cout << "  Distances per query : " << expected_passing << "\n";
    if (per_query_scan)
        std::cout << "  Passing-set build   : inside every query (prefilter-scan)\n";
    else
        std::cout << "  Passing-set build   : " << std::fixed << std::setprecision(3) << passing_build_ms
                  << " ms (once per cell; " << std::setprecision(4) << passing_build_ms / static_cast<double>(nq)
                  << " ms per query over " << nq << ")\n";

    search_result_summary_t out = {label,      mode,       selectivity, std::size_t(0), 0.0,        false,
            med_ms,     p50_ms,     p90_ms,      p99_ms,         med_qps,
            static_cast<double>(expected_passing), static_cast<double>(expected_passing),
            // Prefilter scans every passing node for every query, so the visit
            // count has no per-query spread. The percentiles equal the mean.
            static_cast<double>(expected_passing), static_cast<double>(expected_passing),
            static_cast<double>(expected_passing), static_cast<double>(expected_passing),
            static_cast<double>(expected_passing),
            avg_recall, p50_recall, p90_recall,  p99_recall,     0.0,
            0.0,
            // Prefilter runs no ACORN cycle, so expansion accounting is empty.
            0.0,        0.0,        0.0,         0.0,
            0.0,        0.0,        0.0,         0.0,
            0.0,        0.0,        0.0,        0.0,
            // Prefilter runs no base-layer loop: no phase split, no stop reason.
            0.0,        0.0,        0.0,         0.0,        0.0,
            0.0,        0.0,        0.0,         0.0,        0.0,
            0.0,        0.0,
            // ... and no `W` to fill, so no prefill time either.
            0.0,        0.0,        0.0,         0.0,
            0.0,        0.0,        0.0,         0.0,        0.0,
            early_term_rate,        results_returned,            st,
            nq};
    out.passing_build_ms = passing_build_ms;
    return out;
}

// ── QPS search result ───────────────────────────────────────────────────────
struct qps_result_t {
    double recall;
    double qps;
    double latency_ms;
    // Share of queries that hit the soft time budget, sampled on the recall
    // pass. Always 0 while --early-termination-ms is unset.
    double early_term_rate;
};

// ── Run a single QPS measurement (all queries batched, median QPS) ──────────
template <typename metric_at, typename make_predicate_at>
qps_result_t run_qps_search(index_t& index, metric_at& metric, std::vector<float const*> const& queries, std::size_t k,
                            std::size_t ef_search, make_predicate_at&& make_predicate, double filter_first_threshold,
                            double bridge_budget_ratio, std::vector<std::vector<vec_key_t>> const& ground_truths,
                            std::size_t navix_heuristic = 0) {

    index_search_config_t cfg;
    cfg.expansion = ef_search;
    cfg.filter_first_threshold = filter_first_threshold;
    cfg.bridge_budget_ratio = bridge_budget_ratio;
    cfg.stride_sampling = STRIDE_SAMPLING;
    cfg.navix_heuristic = navix_heuristic;
    cfg.exact = false;
    cfg.early_termination_time_threshold = early_termination_threshold();
    cfg.early_termination_min_results = ET_MIN_RESULTS;

    std::size_t nq = queries.size();
    std::size_t st = std::min(SEARCH_THREADS, nq);

    // Persistent worker pool: spawned once per run_qps_search. It is reused across
    // warmup, the recall pass, and all RUNS measurement batches, so the batch
    // timer contains no thread spawn/join.
    query_pool_t pool(st, nq);
    auto parallel_queries = [&](auto&& fn) { pool.run(std::forward<decltype(fn)>(fn)); };

    // Warmup
    for (std::size_t i = 0; i < BENCH_WARMUP; ++i) {
        parallel_queries([&](std::size_t q, std::size_t tid) {
            auto c = cfg;
            c.thread = tid;
            auto pred = make_predicate(q);
            auto r = bench_search(index, queries[q], k, metric, c, pred);
            (void)r;
        });
    }

    // Recall: a single pass. It also serves as the early-termination sample,
    // because the timed batches below must not pay for extra bookkeeping.
    std::vector<double> recalls(nq);
    std::vector<double> early_terms(nq);
    parallel_queries([&](std::size_t q, std::size_t tid) {
        auto c = cfg;
        c.thread = tid;
        auto pred = make_predicate(q);
        auto r = bench_search(index, queries[q], k, metric, c, pred);
        recalls[q] = r ? compute_recall(r, ground_truths[q], k) : 0.0;
        early_terms[q] = r.early_terminated ? 1.0 : 0.0;
    });
    double avg_recall = 0.0;
    for (double r : recalls)
        avg_recall += r;
    avg_recall /= static_cast<double>(nq);
    double early_term_rate = 0.0;
    for (double e : early_terms)
        early_term_rate += e;
    early_term_rate /= static_cast<double>(nq);

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
            auto r = bench_search(index, queries[q], k, metric, c, pred);
            (void)r;
        });
        double elapsed_ms = timer.elapsed_ms();
        double elapsed_s = elapsed_ms / 1000.0;
        qps_values.push_back(static_cast<double>(nq) / elapsed_s);
        latency_values.push_back(elapsed_ms * static_cast<double>(st) / static_cast<double>(nq));
    }
    double med_qps = median(qps_values);
    double med_latency = median(latency_values);

    return {avg_recall, med_qps, med_latency, early_term_rate};
}

// ── Compute exact ground truth for a given predicate ────────────────────────
template <typename metric_at, typename predicate_at>
std::vector<vec_key_t> compute_ground_truth(index_t& index, metric_at& metric, float const* query, std::size_t k,
                                            predicate_at&& predicate, std::size_t thread_id = 0) {
    index_search_config_t exact_cfg;
    exact_cfg.exact = true;
    exact_cfg.thread = thread_id;
    auto result = bench_search(index, query, k, metric, exact_cfg, predicate);
    std::vector<vec_key_t> gt;
    gt.reserve(result.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        gt.push_back(result[i].member.key);
    return gt;
}

// ── Index volume file name ───────────────────────────────────────────────────
// Pruning-free volumes carry a `_pf` suffix so both constructions coexist.
static std::string index_volume_path() {
    std::ostringstream oss;
    oss << VOLUME_DIR << "/" << DATASET_NAME << "_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef"
        << EF_CONSTRUCTION_VAL << (PRUNING_FREE ? "_pf" : "") << ".usearch";
    return oss.str();
}

// ── Ground truth file name ───────────────────────────────────────────────────
// {dataset}_{N}_{D}_m{M}_ef{EFC}{predicate_tag}_d{base fingerprint}_gt_k{k}_nq{nq}.bin
// The fingerprint ties the cache to the base vectors it was computed from (see
// file_fingerprint).
static std::string gt_file_path(std::size_t gt_k, std::size_t nq) {
    std::ostringstream oss;
    oss << GT_DIR << "/" << DATASET_NAME << "_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef"
        << EF_CONSTRUCTION_VAL << predicate_tag() << "_d" << BASE_FINGERPRINT << "_gt_k" << gt_k << "_nq" << nq
        << ".bin";
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
            char const* val = argv[++i];
            try {
                return static_cast<std::size_t>(std::stoull(val));
            } catch (std::exception const&) {
                std::cerr << "Invalid numeric value for " << arg << ": " << val << "\n";
                std::exit(1);
            }
        };
        auto next_double = [&]() -> double {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            char const* val = argv[++i];
            try {
                return std::stod(val);
            } catch (std::exception const&) {
                std::cerr << "Invalid numeric value for " << arg << ": " << val << "\n";
                std::exit(1);
            }
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
            if (METRIC_NAME != "l2" && METRIC_NAME != "cosine" && METRIC_NAME != "ip") {
                std::cerr << "Unknown metric: " << METRIC_NAME << " (use 'l2', 'cosine', or 'ip')\n";
                std::exit(1);
            }
        } else if (arg == "--dataset") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --dataset\n";
                std::exit(1);
            }
            DATASET_NAME = argv[++i];
            if (DATASET_NAME != "sift" && DATASET_NAME != "gist" && DATASET_NAME != "text2image" &&
                DATASET_NAME != "laion" && DATASET_NAME != "yfcc") {
                std::cerr << "Unknown dataset: " << DATASET_NAME
                          << " (use 'sift', 'gist', 'text2image', 'laion', or 'yfcc')\n";
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
        } else if (arg == "--filter-first-threshold") {
            ACORN_THRESHOLD = next_double();
        } else if (arg == "--cmpkey") {
            USE_CMPKEY = true;
        } else if (arg == "--stride") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --stride (use 'on' or 'off')\n";
                std::exit(1);
            }
            std::string v = argv[++i];
            if (iequals(v, "on"))
                STRIDE_SAMPLING = true;
            else if (iequals(v, "off"))
                STRIDE_SAMPLING = false;
            else {
                std::cerr << "Invalid --stride value: " << v << " (use 'on' or 'off')\n";
                std::exit(1);
            }
        } else if (arg == "--bridge-budget-ratio") {
            // RACORN-1 bridge budget ratio (used by the racorn-1 mode).
            BRIDGE_BUDGET_RATIO = next_double();
            if (BRIDGE_BUDGET_RATIO < 0.0) {
                std::cerr << "--bridge-budget-ratio must be >= 0\n";
                std::exit(1);
            }
        } else if (arg == "--bridge-budget-ratios") {
            // Sweep list for the racorn-1 mode: one RACORN-1 row per ratio.
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --bridge-budget-ratios\n";
                std::exit(1);
            }
            std::string list = argv[++i];
            BBR_VALUES.clear();
            std::istringstream ss(list);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                double v = 0.0;
                try {
                    v = std::stod(token);
                } catch (std::exception const&) {
                    std::cerr << "Invalid bridge budget ratio: " << token << "\n";
                    std::exit(1);
                }
                if (v < 0.0) {
                    std::cerr << "--bridge-budget-ratios values must be >= 0\n";
                    std::exit(1);
                }
                BBR_VALUES.push_back(v);
            }
            if (BBR_VALUES.empty()) {
                std::cerr << "No valid values in --bridge-budget-ratios: " << list << "\n";
                std::exit(1);
            }
        } else if (arg == "--recall-target") {
            // Stop a latency-mode ef sweep for a (selectivity, mode, br) series
            // once avg recall reaches this value. Only meaningful with --ef-values.
            RECALL_TARGET = next_double();
        } else if (arg == "--build-threads") {
            BUILD_THREADS = next_val();
        } else if (arg == "--three-hop-threshold") {
            // acorn-1-3hop trigger: a third hop when the 2-hop passing candidates < connectivity_base x tau.
            THREE_HOP_TAU = next_double();
            if (THREE_HOP_TAU < 0.0) {
                std::cerr << "--three-hop-threshold must be >= 0\n";
                std::exit(1);
            }
        } else if (arg == "--predicate") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --predicate\n";
                std::exit(1);
            }
            PREDICATE_KIND = argv[++i];
            if (PREDICATE_KIND != "modulo" && PREDICATE_KIND != "random") {
                std::cerr << "Unknown predicate: " << PREDICATE_KIND << " (use 'modulo' or 'random')\n";
                std::exit(1);
            }
        } else if (arg == "--seed") {
            PREDICATE_SEED = static_cast<std::uint64_t>(next_val());
        } else if (arg == "--pruning-free") {
            // ACORN-paper-spec construction (Section 5.3): skip the RNG pruning
            // heuristic and keep the nearest candidates. Applies to --build-index
            // and selects the `_pf` volume in the benchmark modes.
            PRUNING_FREE = true;
        } else if (arg == "--early-termination-fractions" || arg == "--early-termination-ms-values") {
            // Soft-timeout ladder for the latency mode (see ET_FRACTIONS/ET_MS_VALUES).
            bool const fractions = arg == "--early-termination-fractions";
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << "\n";
                std::exit(1);
            }
            std::string list = argv[++i];
            std::vector<double>& dst = fractions ? ET_FRACTIONS : ET_MS_VALUES;
            dst.clear();
            std::istringstream ss(list);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                double v = 0.0;
                try {
                    v = std::stod(token);
                } catch (std::exception const&) {
                    std::cerr << "Invalid value for " << arg << ": " << token << "\n";
                    std::exit(1);
                }
                if (v < 0.0 || (fractions && v > 1.0)) {
                    std::cerr << arg << " values must be " << (fractions ? "within [0, 1]" : ">= 0") << "\n";
                    std::exit(1);
                }
                dst.push_back(v);
            }
            if (dst.empty()) {
                std::cerr << "No valid values in " << arg << ": " << list << "\n";
                std::exit(1);
            }
        } else if (arg == "--early-termination-ms") {
            // Soft base-layer time budget; negative disables it.
            EARLY_TERMINATION_MS = next_double();
        } else if (arg == "--et-min-results") {
            // Heap fill the budget waits for (0 = ef_search; k = budget-only stopping).
            ET_MIN_RESULTS = next_val();
        } else if (arg == "--et-skip-unbudgeted") {
            ET_SKIP_UNBUDGETED = true;
        } else if (arg == "--et-anchor") {
            // <mode>@<ef>:<f1,f2,...> — see ET_ANCHORS.
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --et-anchor\n";
                std::exit(1);
            }
            std::string const spec = argv[++i];
            std::size_t const at = spec.find('@');
            std::size_t const colon = at == std::string::npos ? std::string::npos : spec.find(':', at);
            if (at == std::string::npos || at == 0 || colon == std::string::npos || colon <= at + 1 ||
                colon + 1 >= spec.size()) {
                std::cerr << "--et-anchor expects <mode>@<ef>:<fractions>, got: " << spec << "\n";
                std::exit(1);
            }
            et_anchor_t a;
            a.mode = spec.substr(0, at);
            a.name = spec.substr(0, colon);
            try {
                a.ef = static_cast<std::size_t>(std::stoul(spec.substr(at + 1, colon - at - 1)));
            } catch (std::exception const&) {
                std::cerr << "--et-anchor: invalid ef in " << spec << "\n";
                std::exit(1);
            }
            std::istringstream ss(spec.substr(colon + 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                double v = 0.0;
                try {
                    v = std::stod(token);
                } catch (std::exception const&) {
                    std::cerr << "--et-anchor: invalid fraction " << token << " in " << spec << "\n";
                    std::exit(1);
                }
                if (!(v > 0.0)) {
                    std::cerr << "--et-anchor fractions must be > 0: " << spec << "\n";
                    std::exit(1);
                }
                a.fractions.push_back(v);
            }
            if (a.fractions.empty()) {
                std::cerr << "--et-anchor: no fractions in " << spec << "\n";
                std::exit(1);
            }
            ET_ANCHORS.push_back(a);
        } else if (arg == "--mode-ef") {
            // <mode>:<ef1,ef2,...> — see MODE_EF.
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --mode-ef\n";
                std::exit(1);
            }
            std::string const spec = argv[++i];
            std::size_t const colon = spec.find(':');
            if (colon == std::string::npos || colon == 0 || colon + 1 >= spec.size()) {
                std::cerr << "--mode-ef expects <mode>:<ef list>, got: " << spec << "\n";
                std::exit(1);
            }
            std::vector<std::size_t> efs;
            std::istringstream ss(spec.substr(colon + 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                try {
                    efs.push_back(static_cast<std::size_t>(std::stoul(token)));
                } catch (std::exception const&) {
                    std::cerr << "--mode-ef: invalid ef " << token << " in " << spec << "\n";
                    std::exit(1);
                }
                if (efs.back() == 0) {
                    std::cerr << "--mode-ef: ef must be > 0: " << spec << "\n";
                    std::exit(1);
                }
            }
            if (efs.empty()) {
                std::cerr << "--mode-ef: no ef values in " << spec << "\n";
                std::exit(1);
            }
            MODE_EF.emplace_back(spec.substr(0, colon), efs);
        } else if (arg == "--runs") {
            BENCH_RUNS = next_val();
            if (BENCH_RUNS == 0)
                BENCH_RUNS = 1;
        } else if (arg == "--warmup") {
            BENCH_WARMUP = next_val();
        } else if (arg == "--json") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for --json\n";
                std::exit(1);
            }
            JSON_PATH = argv[++i];
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
                try {
                    if (token.back() == '%')
                        val = std::stod(token.substr(0, token.size() - 1)) / 100.0;
                    else
                        val = std::stod(token);
                } catch (std::exception const&) {
                    std::cerr << "Invalid selectivity value: " << token << "\n";
                    std::exit(1);
                }
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
            EF_VALUES_SET = true;
            QPS_EF_VALUES.clear();
            std::istringstream ss(ef_list);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (token.empty())
                    continue;
                try {
                    QPS_EF_VALUES.push_back(static_cast<std::size_t>(std::stoull(token)));
                } catch (std::exception const&) {
                    std::cerr << "Invalid ef_search value: " << token << "\n";
                    std::exit(1);
                }
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
            static char const* const valid[] = {"exact",          "prefilter",   "prefilter-scan", "hnsw",
                                                "acorn-1",        "acorn-1-3hop", "racorn-1",   "navix-onehop-s",
                                                "navix-blind",    "navix-directed", "navix-local",  "navix-local-paper"};
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
                    std::cerr << "Unknown mode: " << token
                              << " (valid: exact, prefilter, prefilter-scan, hnsw, acorn-1, acorn-1-3hop, racorn-1, "
                                 "navix-onehop-s, navix-blind, navix-directed, navix-local, navix-local-paper)\n";
                    std::exit(1);
                }
                FILTER_MODES.push_back(token);
            }
            if (FILTER_MODES.empty()) {
                std::cerr << "No valid modes in: " << mode_list << "\n";
                std::exit(1);
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: bench_acorn [options]\n"
                << "\n"
                << "Modes (mutually exclusive; default = latency benchmark):\n"
                << "  (default)             Latency benchmark: the modes of --modes at every selectivity.\n"
                << "  --build-index         Build the index volume and exit. No queries are run.\n"
                << "  --build-gt            Build ground-truth file for queries. Requires existing volume.\n"
                << "  --qps                 QPS benchmark: sweep ef_search values, report recall/QPS/latency.\n"
                << "\n"
                << "Notes:\n"
                << "  - GT files saved by --build-gt are auto-loaded by latency/qps modes when their\n"
                << "    header is compatible (file_nq >= bench nq, file_gt_k >= K, magic 'GT01').\n"
                << "\n"
                << "-- Common options (all modes) --------------------------------------\n"
                << "  --dataset <name>      Dataset: sift, gist, text2image, laion, yfcc (default: sift)\n"
                << "  --n <N>               Number of vectors to index (default: 1000000)\n"
                << "                        sift max: " << SIFT_N_MAX << ", gist max: " << GIST_N_MAX
                << ", text2image max: " << TEXT2IMAGE_N_MAX << ", laion max: " << LAION_N_MAX
                << ", yfcc max: " << YFCC_N_MAX << "\n"
                << "  --m <N>               Graph connectivity M (default: 16)\n"
                << "  --ef-construction <N> EF construction expansion (default: 100)\n"
                << "  --k <N>               Top-K results (default: 100)\n"
                << "  --metric <name>       Distance metric: l2, cosine, ip (default: per dataset)\n"
                << "                        cosine L2-normalizes base+query; ip uses raw magnitudes.\n"
                << "                        NOTE: volume/GT filenames do NOT encode the metric --\n"
                << "                        use a separate --volume-dir/--gt-dir per metric.\n"
                << "                        (default: l2 for sift, gist and yfcc; cosine for text2image and laion)\n"
                << "  --nq <N>              Number of query vectors to use (default: 100)\n"
                << "                        sift max: 10000, gist and laion max: 1000, text2image and yfcc max: 100000\n"
                << "  --search-threads <N>  Parallel search threads (default: 1; --build-gt\n"
                << "                        defaults to hardware_concurrency when unset)\n"
                << "  --data-dir <path>     Directory for raw dataset files (default: data/raw)\n"
                << "  --volume-dir <path>   Directory for index volume files (default: data/volumes)\n"
                << "  --gt-dir <path>       Directory for ground truth files (default: data/gt)\n"
                << "  --selectivity <list>  Comma-separated selectivity values to filter test cases\n"
                << "                        e.g. --selectivity 10%,1%,0.1% or --selectivity 0.10,0.01\n"
                << "                        Applies to --build-gt, latency, and --qps modes.\n"
                << "                        Ignored by --build-index.\n"
                << "  --help, -h            This text\n"
                << "\n"
                << "-- Predicate (--build-gt, latency and --qps modes) -----------------\n"
                << "  --predicate <kind>    Control predicate: modulo (default, key % m < t) or random\n"
                << "                        (seed-fixed uniform uint32 attribute per key, pass when\n"
                << "                        attr < selectivity*2^32). GT files get a _rnd<seed> tag.\n"
                << "  --seed <N>            Seed for --predicate random (default 12345).\n"
                << "\n"
                << "-- Volume build mode (--build-index) -------------------------------\n"
                << "  Uses --m / --ef-construction to control build quality. --nq, --selectivity,\n"
                << "  and all benchmark-only flags are ignored. Existing volume is overwritten.\n"
                << "  --build-threads <N>   Indexing threads (default: hardware_concurrency).\n"
                << "  --pruning-free        Skip the RNG pruning heuristic (ACORN paper Sec. 5.3\n"
                << "                        construction). Volume gets a `_pf` suffix; pass the\n"
                << "                        same flag to benchmark modes to load that volume.\n"
                << "\n"
                << "-- Ground truth build mode (--build-gt) ----------------------------\n"
                << "  --build-gt-k <N>      GT results per query (default: same as --k)\n"
                << "  --nq <N>              Use this many queries for GT (default: dataset q_max\n"
                << "                        when --nq is not specified). Smaller --nq -> smaller GT.\n"
                << "  --selectivity <list>  Build only the listed selectivities (default: all 25).\n"
                << "\n"
                << "-- Latency benchmark mode (default) --------------------------------\n"
                << "  --ef-search <N>           EF search expansion (default: 200)\n"
                << "  --ef-values <list>        Sweep ef_search over these values (one row per ef).\n"
                << "                            Replaces --ef-search; combine with --recall-target\n"
                << "                            to stop each (selectivity, mode, br) series early.\n"
                << "  --recall-target <F>       With --ef-values: stop a series once avg recall >= F.\n"
                << "  --bridge-budget-ratios <list>  Run racorn-1 once per ratio (e.g. 1.0,1.5,2.0).\n"
                << "  --early-termination-fractions <list>  Soft-timeout ladder: each series first runs\n"
                << "                            without a budget, then once per fraction of its own\n"
                << "                            unlimited p50 (e.g. 0,0.1,0.25,0.5,0.75,0.9). Also applies\n"
                << "                            to prefilter (exact scan stops once ef_search matches are\n"
                << "                            held and the budget is spent). Rows carry early_termination_ms.\n"
                << "  --early-termination-ms-values <list>  Absolute soft-timeout ladder (ms), same semantics.\n"
                << "  --et-skip-unbudgeted      With --early-termination-ms-values, skip each series' unbudgeted row.\n"
                << "  --et-min-results <N>      Heap fill the soft budget waits for before it may stop a search\n"
                << "                            (default 0 = ef_search). N = k makes the budget the only stopping\n"
                << "                            rule beyond k results (iso-latency mode).\n"
                << "  --et-anchor <mode>@<ef>:<fractions>  Budgets = fractions x that series' prefill p50 measured\n"
                << "                            in this process (the time its W first held ef results), applied\n"
                << "                            to every graph series of the same selectivity. Repeatable;\n"
                << "                            fractions may exceed 1. Rows carry et_anchor / et_fraction.\n"
                << "  --mode-ef <mode>:<list>   ef sweep for one mode only (overrides --ef-values / --ef-search for it). Repeatable.\n"
                << "  --filter-first-threshold <F>      Filter-first threshold (probe ratio at entry) (default: 0.3)\n"
                << "  --three-hop-threshold <F>  acorn-1-3hop trigger (default 1.0): expand through\n"
                << "                        predicate-failing 2-hop nodes to 3-hop when the 2-hop harvest is\n"
                << "                        below connectivity_base x F. Only passing nodes are queued.\n"
                << "  --bridge-budget-ratio <F> RACORN-1 bridge budget ratio (default: 1.0).\n"
                << "                            Bridge budget = one_hop_count * ratio; used by the\n"
                << "                            racorn-1 mode. acorn-1 mode always uses 0.0.\n"
                << "  --early-termination-ms <N> Soft base-layer time budget in ms, fractional values\n"
                << "                            kept (default: -1 = off).\n"
                << "                            A traversal stops once the budget is spent AND the\n"
                << "                            result heap holds ef_search candidates, so recall is\n"
                << "                            traded for latency. Applies to every graph mode; exact\n"
                << "                            search and the prefilter arms ignore it. Adds an EarlyTerm column\n"
                << "                            (share of queries that stopped early).\n"
                << "  --cmpkey                  Pass key-ascending neighbor_cmp to search (default: off)\n"
                << "  --stride on|off           Stride-sample the 2-hop passed cap and the RACORN-1\n"
                << "                            bridge pool (default: on). 'off' keeps the head of each\n"
                << "                            pool instead (prefix truncation) - stride ablation.\n"
                << "  --runs <N>                Measurement runs per query (default: 10)\n"
                << "  --warmup <N>              Warmup runs before measurement (default: 3)\n"
                << "  --json <path>             Write structured results (config + per-row metrics) as JSON.\n"
                << "  --skip-exact              Skip the exact-search rows and the baseline row.\n"
                << "                            (GT for recall is still computed/loaded.)\n"
                << "  --modes <list>            Comma-separated modes to run.\n"
                << "                            Valid: exact, prefilter, prefilter-scan, hnsw, acorn-1,\n"
                << "                            acorn-1-3hop, racorn-1, navix-onehop-s, navix-blind,\n"
                << "                            navix-directed, navix-local, navix-local-paper.\n"
                << "                            prefilter = passing keys gathered upfront (assumed\n"
                << "                            attribute index, untimed) + timed exact scan over them.\n"
                << "                            prefilter-scan = no attribute index: every query scans the\n"
                << "                            attribute column itself, then exact-scans what it collected.\n"
                << "                            navix-* = the NaviX (VLDB 2025) heuristics on the same graph;\n"
                << "                            navix-local follows the authors' code, navix-local-paper the paper's text.\n"
                << "                            e.g. --modes hnsw,acorn-1 or --modes acorn-1,racorn-1\n"
                << "                            Baseline (no-predicate) row skipped when set.\n"
                << "                            Speedup/Recall/Dist ratios shown only when HNSW is included\n"
                << "                            and --ef-values is not given.\n"
                << "\n"
                << "-- QPS benchmark mode (--qps) --------------------------------------\n"
                << "  --ef-values <list>        Comma-separated ef_search values to sweep\n"
                << "                            (default: 128,256,384,512). Replaces --ef-search.\n"
                << "  --filter-first-threshold, --bridge-budget-ratio, --bridge-budget-ratios, --recall-target,\n"
                << "  --early-termination-ms, --et-min-results, --cmpkey, --stride, --json\n"
                << "                            As in the latency benchmark mode.\n"
                << "  --runs <N>                Measurement runs per configuration (default: 10)\n"
                << "                            Each run = full nq batch; reported QPS/latency are medians.\n"
                << "  --warmup <N>              Warmup runs (default: 3)\n"
                << "  --modes <list>            Comma-separated modes to run.\n"
                << "                            Valid: hnsw, acorn-1, racorn-1, navix-*; exact, acorn-1-3hop\n"
                << "                            and the prefilter modes are not run in QPS mode.\n"
                << "  --skip-exact has no effect (already implied).\n";
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
    } else if (DATASET_NAME == "laion") {
        n_max = LAION_N_MAX;
        q_max = 1'000; // laion_query.fvecs holds the 1000 rows after the base prefix
        D_VAL = 512;
    } else if (DATASET_NAME == "yfcc") {
        n_max = YFCC_N_MAX;
        q_max = 100'000; // query.public.100K.u8bin
        D_VAL = 192;
    } else {
        D_VAL = 128;
    }
    // Default metric per dataset (if not explicitly set via --metric).
    // laion vectors are pre-normalized CLIP embeddings: cosine (re-normalization is a no-op).
    if (METRIC_NAME.empty()) {
        if (DATASET_NAME == "text2image" || DATASET_NAME == "laion")
            METRIC_NAME = "cosine";
        else
            METRIC_NAME = "l2"; // sift, gist, yfcc: L2 euclidean
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

    // --build-gt defaults: GT_K defaults to K; nq defaults to q_max unless the user set it;
    // --search-threads defaults to hardware_concurrency. GT is exact NN and embarrassingly
    // parallel, so on shared boxes it is fine to use every available core for the one-time build.
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

    if (!BUILD_INDEX_ONLY) {
        if (num_queries > 0 && query_vecs_data) {
            std::size_t nq = std::min(num_queries, NUM_QUERIES);
            query_data_owned.assign(query_vecs_data, query_vecs_data + nq * D_VAL);
            if (METRIC_NAME == "cosine") {
                normalize_rows_l2(query_data_owned.data(), nq, D_VAL);
                std::cout << "Query vectors normalized to unit length.\n";
            }
            queries.reserve(nq);
            for (std::size_t qi = 0; qi < nq; ++qi)
                queries.push_back(query_data_owned.data() + qi * D_VAL);
            std::cout << "Using " << nq << " " << DATASET_NAME << " query vectors\n";
        } else {
            std::cerr << "No query vectors available for dataset '" << DATASET_NAME << "'\n";
            return 1;
        }
    }

    // Build or load index
    index_config_t config(M_VAL);
    config.pruning_free = PRUNING_FREE;
    aligned_index_t<index_t> aligned(config);
    auto& index = *aligned.ptr;

    std::size_t num_threads = BUILD_THREADS ? BUILD_THREADS : std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;

    metric_t metric{&store, D_VAL};
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
        index.reserve(index_limits_t(N_VAL, std::max(num_threads, SEARCH_THREADS)));

        std::cout << "Building index with " << num_threads << " threads... " << std::flush;
        stopwatch_t build_timer;

        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        std::size_t chunk = (N_VAL + num_threads - 1) / num_threads;

        for (std::size_t t = 0; t < num_threads; ++t) {
            std::size_t begin = t * chunk;
            std::size_t end = std::min(begin + chunk, N_VAL);
            threads.emplace_back([&, begin, end, t]() {
                index_update_config_t ucfg;
                ucfg.expansion = EF_CONSTRUCTION_VAL;
                ucfg.thread = t;
                for (std::size_t i = begin; i < end; ++i)
                    index.add(static_cast<vec_key_t>(i), store.row(i), metric, ucfg);
            });
        }
        for (auto& th : threads)
            th.join();

        double build_ms = build_timer.elapsed_ms();
        BUILD_TIME_MS = build_ms;
        std::cout << "done (" << std::fixed << std::setprecision(1) << build_ms << " ms, "
                  << static_cast<std::size_t>(N_VAL / (build_ms / 1000.0)) << " vec/s)\n";
        std::cout << "Index size: " << index.size() << "  [built from scratch]\n\n";
    }

    {
        // Measured base-level average degree. Must come after both the load and
        // the build-from-scratch paths: an empty index reports 0 here.
        auto st = index.stats(0);
        MAVG_BASE = st.nodes ? static_cast<double>(st.edges) / static_cast<double>(st.nodes) : 0.0;
    }

    // --build-index: save and exit
    if (BUILD_INDEX_ONLY) {
        std::error_code ec;
        std::filesystem::create_directories(VOLUME_DIR, ec); // no-op if exists
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
        double filter_first_threshold; // passed to search (-1 for exact/hnsw)
        bool exact;
        double bridge_budget_ratio; // 0.0 = ACORN-1 / non-RACORN; >0.0 = RACORN-1
        bool prefilter;             // pre-filtering + exact-scan baseline (index-free)
        std::size_t navix;          // NaviX heuristic (0 off, 1 onehop-s, 2 blind, 3 directed, adaptive-local: 8 Kuzu, 5 paper)
        double three_hop;           // ACORN-1 3-hop trigger (0.0 = two hops only)
    };

    auto gt_key = [](std::size_t mod, std::size_t thr) { return (mod << 20) | thr; };

    // RACORN-1 mode's bridge budget ratio (0.0 for all other modes). Set from
    // --bridge-budget-ratio (default 1.0).
    double const racorn_ratio = BRIDGE_BUDGET_RATIO;
    double const three_hop_tau = THREE_HOP_TAU;

#define CASE(sel_label, sel, mod, thr)                                                                                 \
    {sel_label " | Exact", "Exact", sel, mod, thr, -1.0, true, 0.0, false, 0, 0.0},                                    \
        {sel_label " | Prefilter", "Prefilter", sel, mod, thr, -1.0, false, 0.0, true, 0, 0.0},                        \
        {sel_label " | Prefilter-scan", "Prefilter-scan", sel, mod, thr, -1.0, false, 0.0, true, 0, 0.0},               \
        {sel_label " | HNSW", "HNSW", sel, mod, thr, -1.0, false, 0.0, false, 0, 0.0},                                 \
        {sel_label " | ACORN-1", "ACORN-1", sel, mod, thr, ACORN_THRESHOLD, false, 0.0, false, 0, 0.0},                \
        {sel_label " | ACORN-1-3hop", "ACORN-1-3hop", sel, mod, thr, ACORN_THRESHOLD, false, 0.0, false, 0,            \
         three_hop_tau},                                                                                               \
        {sel_label " | RACORN-1", "RACORN-1", sel, mod, thr, ACORN_THRESHOLD, false, racorn_ratio, false, 0, 0.0},     \
        {sel_label " | NaviX-onehop-s", "NaviX-onehop-s", sel, mod, thr, -1.0, false, 0.0, false, 1, 0.0},             \
        {sel_label " | NaviX-blind", "NaviX-blind", sel, mod, thr, -1.0, false, 0.0, false, 2, 0.0},                   \
        {sel_label " | NaviX-directed", "NaviX-directed", sel, mod, thr, -1.0, false, 0.0, false, 3, 0.0},             \
        {sel_label " | NaviX-local", "NaviX-local", sel, mod, thr, -1.0, false, 0.0, false, 8, 0.0},                   \
        {sel_label " | NaviX-local-paper", "NaviX-local-paper", sel, mod, thr, -1.0, false, 0.0, false, 5, 0.0}

    std::vector<test_case_t> cases = {
        CASE("Selectivity 100%", 1.000, 1, 1),       CASE("Selectivity  90%", 0.900, 10, 9),
        CASE("Selectivity  70%", 0.700, 10, 7),      CASE("Selectivity  50%", 0.500, 2, 1),
        CASE("Selectivity  30%", 0.300, 10, 3),      CASE("Selectivity  20%", 0.200, 10, 2),
        CASE("Selectivity   9%", 0.090, 100, 9),     CASE("Selectivity   7%", 0.070, 100, 7),
        CASE("Selectivity  10%", 0.100, 10, 1),      CASE("Selectivity   5%", 0.050, 20, 1),
        CASE("Selectivity   4%", 0.040, 25, 1),      CASE("Selectivity   3%", 0.030, 100, 3),
        CASE("Selectivity   2%", 0.020, 50, 1),      CASE("Selectivity   1%", 0.010, 100, 1),
        CASE("Selectivity 0.9%", 0.0090, 1000, 9),   CASE("Selectivity 0.7%", 0.0070, 1000, 7),
        CASE("Selectivity 0.8%", 0.0080, 125, 1),    CASE("Selectivity 0.6%", 0.0060, 500, 3),
        CASE("Selectivity 0.5%", 0.0050, 200, 1),    CASE("Selectivity 0.4%", 0.0040, 250, 1),
        CASE("Selectivity 0.3%", 0.0030, 1000, 3),   CASE("Selectivity 0.2%", 0.0020, 500, 1),
        CASE("Selectivity 0.1%", 0.0010, 1000, 1),   CASE("Selectivity 0.05%", 0.0005, 2000, 1),
        CASE("Selectivity 0.01%", 0.0001, 10000, 1),
    };
#undef CASE

    // Predicate factory: returns a fixed selectivity_predicate_t for every query.
    // The query index is ignored; it is kept for a uniform per-query factory signature.
    auto make_modulo_factory = [&](std::size_t modulus, std::size_t pred_threshold) {
        // Resolve the selectivity of this (modulus, threshold) case so the
        // random-attribute predicate gets the matching attribute threshold.
        double sel = static_cast<double>(pred_threshold) / static_cast<double>(modulus);
        selectivity_predicate_t pred = make_case_predicate(modulus, pred_threshold, sel);
        return [pred](std::size_t /*q*/) { return pred; };
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
        std::filesystem::create_directories(GT_DIR, ec); // no-op if exists
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

            // Per-query GT: exact filtered search under this case's predicate
            // (modulo or random attribute), the same for every query.
            auto run_one = [&](std::size_t qi, std::size_t tid, std::vector<std::vector<vec_key_t>>& gts) {
                index_search_config_t exact_cfg;
                exact_cfg.exact = true;
                exact_cfg.thread = tid;
                selectivity_predicate_t pred = make_case_predicate(sel.modulus, sel.pred_threshold, sel.selectivity);
                auto result = index.search(queries[qi], gt_k, metric, exact_cfg, pred);
                gts[qi].reserve(result.size());
                for (std::size_t ri = 0; ri < result.size(); ++ri)
                    gts[qi].push_back(result[ri].member.key);
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
    // Accepts a per-query predicate factory `make_pred(qi)`; modulo callers wrap
    // their fixed predicate in a constant lambda.
    std::size_t gt_threads = std::min(SEARCH_THREADS, queries.size());
    auto compute_gt_parallel = [&](auto&& make_pred) {
        std::vector<std::vector<vec_key_t>> gts(queries.size());
        if (gt_threads <= 1) {
            for (std::size_t qi = 0; qi < queries.size(); ++qi) {
                auto pred = make_pred(qi);
                gts[qi] = compute_ground_truth(index, metric, queries[qi], K_VAL, pred, 0);
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
                        gts[qi] = compute_ground_truth(index, metric, queries[qi], K_VAL, pred, t);
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
    // Candidates are found by scanning GT_DIR for files whose name starts with the
    // dataset/index prefix "{dataset}_{N}_{D}_m{M}_ef{EFC}{tag}_d{fingerprint}_gt_k" and
    // ends in ".bin". These files differ only by their k/nq suffix. The exact-nq file for
    // this run is tried first; the rest follow in sorted order.
    // Compatibility rules per candidate:
    //   - file num_queries >= bench queries.size() (we use first queries.size() entries)
    //   - file gt_k >= K                           (we use first K keys per query)
    //   - selectivity match keyed by (modulus, threshold)
    {
        std::ostringstream prefix_oss;
        prefix_oss << DATASET_NAME << "_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef" << EF_CONSTRUCTION_VAL
                   << predicate_tag() << "_d" << BASE_FINGERPRINT << "_gt_k";
        std::string const prefix = prefix_oss.str();

        // Filename (not full path) of the ideal exact-nq GT file for this run.
        std::string const exact_name = std::filesystem::path(gt_file_path(K_VAL, queries.size())).filename().string();

        std::vector<std::string> candidates; // filenames only
        std::error_code scan_ec;
        for (auto const& entry : std::filesystem::directory_iterator(GT_DIR, scan_ec)) {
            if (!entry.is_regular_file())
                continue;
            std::string fname = entry.path().filename().string();
            if (fname.size() >= 4 && fname.compare(0, prefix.size(), prefix) == 0 &&
                fname.compare(fname.size() - 4, 4, ".bin") == 0)
                candidates.push_back(fname);
        }
        std::sort(candidates.begin(), candidates.end());
        // Move the exact-nq file to the front if present.
        auto exact_it = std::find(candidates.begin(), candidates.end(), exact_name);
        if (exact_it != candidates.end())
            std::rotate(candidates.begin(), exact_it, exact_it + 1);

        for (std::string const& fname : candidates) {
            std::string gt_path = GT_DIR + "/" + fname;
            std::ifstream ifs(gt_path, std::ios::binary);
            if (!ifs.is_open())
                continue;
            char magic[4] = {};
            ifs.read(magic, 4);
            uint32_t num_sels = 0, file_nq_hdr = 0, file_gt_k = 0;
            ifs.read(reinterpret_cast<char*>(&num_sels), sizeof(num_sels));
            ifs.read(reinterpret_cast<char*>(&file_nq_hdr), sizeof(file_nq_hdr));
            ifs.read(reinterpret_cast<char*>(&file_gt_k), sizeof(file_gt_k));

            bool valid = ifs.good() && std::string(magic, 4) == "GT01" //
                         && file_nq_hdr >= queries.size() && file_gt_k >= K_VAL;
            if (!valid) {
                std::cout << "GT file " << gt_path << " skipped (";
                if (std::string(magic, 4) != "GT01")
                    std::cout << "bad magic";
                else if (file_nq_hdr < queries.size())
                    std::cout << "nq=" << file_nq_hdr << " < bench " << queries.size();
                else if (file_gt_k < K_VAL)
                    std::cout << "gt_k=" << file_gt_k << " < K=" << K_VAL;
                std::cout << ")\n";
                continue;
            }

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
                        std::size_t use_cnt = std::min<std::size_t>(cnt, K_VAL);
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
            break;
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
        auto factory = make_modulo_factory(tc.modulus, tc.pred_threshold);
        gt_cache[key] = compute_gt_parallel(factory);
        ++gt_done;
        std::cout << "  [" << gt_done << "/" << gt_misses << "] " << tc.label << ": " << queries.size() << " queries ("
                  << std::setprecision(1) << gt_timer.elapsed_ms() << " ms)\n";
    }

    // Shared summary config header (Vectors/Dimensions/... ) for both the QPS and
    // latency summary tables. `qps` toggles the few mode-specific lines.
    auto print_summary_config = [&](bool qps) {
        std::cout << "  Vectors     : " << N_VAL << "\n";
        std::cout << "  Dimensions  : " << D_VAL << "\n";
        std::cout << "  Connectivity: " << M_VAL << "\n";
        std::cout << "  EF construct: " << EF_CONSTRUCTION_VAL << "\n";
        std::cout << "  Construction: " << (PRUNING_FREE ? "pruning-free" : "pruned (default)") << "\n";
        std::cout << "  Predicate   : " << PREDICATE_KIND
                  << (predicate_random() ? " (seed " + std::to_string(PREDICATE_SEED) + ")" : "") << "\n";
        std::cout << "  K           : " << K_VAL << "\n";
        if (!qps)
            std::cout << "  EF search   : " << EF_SEARCH_VAL << "\n";
        std::cout << "  ACORN thresh: " << ACORN_THRESHOLD << "\n";
        if (early_termination_enabled())
            std::cout << "  Early-term  : " << EARLY_TERMINATION_MS << " ms (soft)\n";
        std::cout << "  Metric      : " << metric_label << "\n";
        std::cout << "  Dataset     : " << DATASET_NAME << "\n";
        if (qps) {
            std::cout << "  Queries     : " << queries.size() << "\n";
            std::cout << "  Runs        : " << BENCH_RUNS << " (" << BENCH_WARMUP << " warmup)\n";
            std::cout << "  EF values   : ";
            for (std::size_t i = 0; i < QPS_EF_VALUES.size(); ++i) {
                if (i)
                    std::cout << ", ";
                std::cout << QPS_EF_VALUES[i];
            }
            std::cout << "\n";
        }
        std::cout << "  Search thr  : " << SEARCH_THREADS << "\n";
        std::cout << "  Index source: " << (index_loaded ? "loaded (" + vol_path + ")" : "built from scratch") << "\n";
        std::cout << "  Mavg (base) : " << std::fixed << std::setprecision(2) << MAVG_BASE << " (cap "
                  << (2 * M_VAL) << ")\n";
    };

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
            double filter_first_threshold;
            double bridge_budget_ratio;
            std::size_t navix;
        };
        qps_mode_t all_modes[] = {
            {"HNSW", -1.0, 0.0, 0},
            {"ACORN-1", ACORN_THRESHOLD, 0.0, 0},
            {"RACORN-1", ACORN_THRESHOLD, racorn_ratio, 0},
            {"NaviX-onehop-s", -1.0, 0.0, 1},
            {"NaviX-blind", -1.0, 0.0, 2},
            {"NaviX-directed", -1.0, 0.0, 3},
            {"NaviX-local", -1.0, 0.0, 8},
            {"NaviX-local-paper", -1.0, 0.0, 5},
        };
        // racorn-1 expands into one entry per --bridge-budget-ratios value.
        std::vector<double> const qps_bbr =
            BBR_VALUES.empty() ? std::vector<double>{racorn_ratio} : BBR_VALUES;
        std::vector<qps_mode_t> modes;
        for (auto const& m : all_modes) {
            if (!mode_matches(m.name))
                continue;
            if (iequals(m.name, "RACORN-1")) {
                for (double br : qps_bbr)
                    modes.push_back({m.name, m.filter_first_threshold, br, 0});
            } else {
                modes.push_back(m);
            }
        }
        if (modes.empty()) {
            std::cerr << "Error: no QPS modes selected (valid QPS modes: hnsw, acorn-1, racorn-1, navix-*; 'exact' and "
                         "'prefilter' are not supported).\n";
            return 1;
        }
        if (unique_sels.empty()) {
            std::cerr << "Error: no selectivities to run. Check --selectivity filter.\n";
            return 1;
        }

        struct qps_row_t {
            double selectivity;
            char const* mode;
            double bridge_budget_ratio;
            std::size_t ef_search;
            double recall;
            double qps;
            double latency_ms;
            double early_term_rate;
        };
        std::vector<qps_row_t> rows;

        std::size_t total_configs = unique_sels.size() * QPS_EF_VALUES.size() * modes.size();
        std::size_t done = 0;

        for (auto const& sel : unique_sels) {
            std::size_t gk = gt_key(sel.modulus, sel.pred_threshold);
            auto& gt = gt_cache[gk];

            for (auto const& m : modes) {
                for (std::size_t ef : QPS_EF_VALUES) {
                    auto run_with = [&](auto&& factory) {
                        return run_qps_search(index, metric, queries, K_VAL, ef, factory, m.filter_first_threshold,
                                              m.bridge_budget_ratio, gt, m.navix);
                    };
                    qps_result_t r = run_with(make_modulo_factory(sel.modulus, sel.pred_threshold));
                    rows.push_back({sel.selectivity, m.name, m.bridge_budget_ratio, ef, r.recall, r.qps, r.latency_ms,
                                    r.early_term_rate});
                    ++done;
                    std::cout << "\r  [" << done << "/" << total_configs << "] " << std::fixed << std::setprecision(1)
                              << (sel.selectivity * 100.0) << "% | " << m.name << " | ef=" << ef
                              << " => recall=" << std::setprecision(4) << r.recall << "  qps=" << std::setprecision(0)
                              << r.qps << "  lat=" << std::setprecision(3) << r.latency_ms << "ms";
                    if (early_termination_enabled())
                        std::cout << "  early=" << std::setprecision(1) << (r.early_term_rate * 100.0) << "%";
                    std::cout << "        " << std::flush;
                    // Same early-stop rule as the latency-mode sweep: once this
                    // (selectivity, mode, br) series reaches the recall target,
                    // larger ef values add no curve information.
                    if (RECALL_TARGET > 0 && r.recall >= RECALL_TARGET)
                        break;
                }
            }
        }
        std::cout << "\n";

        // ── QPS Summary table ───────────────────────────────────────────────
        std::cout << "\n========================================================\n";
        std::cout << " QPS Benchmark Summary\n";
        std::cout << "========================================================\n\n";

        print_summary_config(/*qps=*/true);
        std::cout << "\n";

        std::cout << std::left << std::setw(14) << "Selectivity" << std::setw(16) << "Mode" << std::right
                  << std::setw(7) << "BR" << std::setw(10) << "EF_Search" << std::setw(10) << "Recall" << std::setw(14)
                  << "Latency(ms)" << std::setw(12) << "QPS";
        if (early_termination_enabled())
            std::cout << std::setw(12) << "EarlyTerm";
        std::cout << "\n";
        // Rule widens with the optional early-termination column.
        std::size_t const qps_table_width = early_termination_enabled() ? 95 : 83;
        std::cout << std::string(qps_table_width, '-') << "\n";

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
                    std::cout << std::string(qps_table_width, '-') << "\n";
                prev_sel = row.selectivity;
            }
            std::cout << std::left << std::setw(14) << sel_str << std::setw(16) << row.mode << std::right;
            if (row.bridge_budget_ratio > 0)
                std::cout << std::fixed << std::setprecision(1) << std::setw(7) << row.bridge_budget_ratio;
            else
                std::cout << std::setw(7) << "-";
            std::cout << std::setw(10) << row.ef_search << std::fixed << std::setprecision(4) << std::setw(10)
                      << row.recall << std::setprecision(3) << std::setw(14) << row.latency_ms << std::setprecision(0)
                      << std::setw(12) << row.qps;
            if (early_termination_enabled())
                std::cout << std::setprecision(1) << std::setw(11) << (row.early_term_rate * 100.0) << "%";
            std::cout << "\n";
        }

        // Structured output for chart generation — same field names as the
        // latency-mode JSON where the semantics coincide, plus "qps".
        if (!JSON_PATH.empty()) {
            std::ofstream ofs(JSON_PATH);
            if (!ofs) {
                std::cerr << "Failed to open --json path: " << JSON_PATH << "\n";
            } else {
                ofs << "{\n  \"config\": {\n";
                ofs << "    \"dataset\": \"" << json_escape(DATASET_NAME) << "\",\n";
                ofs << "    \"n\": " << N_VAL << ", \"d\": " << D_VAL << ", \"m\": " << M_VAL
                    << ", \"ef_construction\": " << EF_CONSTRUCTION_VAL << ",\n";
                ofs << "    \"k\": " << K_VAL << ", \"metric\": \"" << json_escape(METRIC_NAME)
                    << "\", \"mode\": \"qps\",\n";
                ofs << "    \"warmup\": " << BENCH_WARMUP << ", \"runs\": " << BENCH_RUNS
                    << ", \"search_threads\": " << SEARCH_THREADS << ",\n";
                ofs << "    \"cmpkey\": " << (USE_CMPKEY ? "true" : "false")
                    << ", \"stride_sampling\": " << (STRIDE_SAMPLING ? "true" : "false")
                    << ", \"mavg\": " << MAVG_BASE
                    << ", \"pruning_free\": " << (PRUNING_FREE ? "true" : "false") << "\n  },\n  \"results\": [\n";
                for (std::size_t i = 0; i < rows.size(); ++i) {
                    auto const& r = rows[i];
                    ofs << "    {\"mode\": \"" << json_escape(r.mode) << "\", \"selectivity\": " << std::setprecision(6)
                        << r.selectivity << ", \"ef_search\": " << r.ef_search
                        << ", \"bridge_budget_ratio\": " << r.bridge_budget_ratio << ", \"recall\": " << r.recall
                        << ", \"qps\": " << r.qps << ", \"latency_ms\": " << r.latency_ms
                        << ", \"early_term_rate\": " << r.early_term_rate << "}" << (i + 1 < rows.size() ? "," : "")
                        << "\n";
                }
                ofs << "  ]\n}\n";
                std::cout << "\nWrote JSON results to " << JSON_PATH << " (" << rows.size() << " rows)\n";
            }
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
    bool show_baseline = FILTER_MODES.empty() && !SKIP_EXACT;

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
        summaries.push_back(run_search(index, metric, queries, K_VAL, EF_SEARCH_VAL, baseline_factory, -1.0, false,
                                       "Baseline (no predicate)", "NoPred", 1.0, 0.0, gt_base));
        ++search_done;
        std::cout.flush();
    }

    // Latency-mode sweep lists. --ef-values turns the single-ef run into an ef
    // sweep. --bridge-budget-ratios expands racorn-1 into one series per ratio.
    std::vector<std::size_t> const ef_sweep =
        EF_VALUES_SET ? QPS_EF_VALUES : std::vector<std::size_t>{EF_SEARCH_VAL};
    std::vector<double> const bbr_sweep = BBR_VALUES.empty() ? std::vector<double>{racorn_ratio} : BBR_VALUES;

    auto format_br = [](double br) {
        std::ostringstream oss;
        oss << std::setprecision(3) << br;
        return oss.str();
    };
    auto format_ms = [](double ms) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << ms;
        return oss.str();
    };
    auto format_frac = [](double f) {
        std::ostringstream oss;
        oss << std::setprecision(4) << f;
        return oss.str();
    };

    // Per-mode ef override (--mode-ef). Labels carry ef whenever any sweep may
    // hold more than one value, so rows of different ef never share a label.
    auto ef_list_for = [&](std::string const& mode) -> std::vector<std::size_t> const& {
        for (auto const& me : MODE_EF)
            if (iequals(me.first, mode))
                return me.second;
        return ef_sweep;
    };
    bool const label_ef = EF_VALUES_SET || !MODE_EF.empty();

    // --et-anchor bookkeeping. Per selectivity key: the measured prefill p50 of
    // each anchor (NaN until its unbudgeted row ran), and the series whose ladder
    // rows wait for them. Cases are grouped by selectivity. So if the key changes
    // while ladders are still waiting, an anchor series never ran. We then abort
    // loudly rather than silently drop the budget rows.
    double const nan_v = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::size_t, std::vector<double>> anchor_ms;
    struct pending_ladder_t {
        test_case_t const* tc;
        double br;
        std::size_t ef;
        std::string label;
        double p50_off;
    };
    std::vector<pending_ladder_t> pending_ladders;
    std::size_t pending_key = 0;
    bool have_pending_key = false;
    auto anchors_ready = [&](std::size_t key) {
        if (ET_ANCHORS.empty())
            return true;
        auto it = anchor_ms.find(key);
        if (it == anchor_ms.end())
            return false;
        for (double v : it->second)
            if (std::isnan(v))
                return false;
        return true;
    };
    auto fail_if_pending = [&]() {
        if (pending_ladders.empty())
            return;
        std::cerr << "Error: " << pending_ladders.size() << " series at selectivity "
                  << pending_ladders.front().tc->selectivity
                  << " are waiting for an --et-anchor series that never ran. Enable the anchor mode "
                     "(--modes) and its ef (--mode-ef / --ef-values), and keep --recall-target off.\n";
        std::exit(1);
    };

    // Soft-timeout ladder for one series. Its budgets come from three sources:
    // fractions of the series' own unlimited p50 (--early-termination-fractions),
    // absolute values (--early-termination-ms-values), and anchor fractions
    // (--et-anchor). Anchor fractions apply to graph series only, once the anchors
    // of this selectivity are measured.
    struct budget_t {
        double ms;
        std::string anchor;
        double fraction;
    };
    auto budget_ladder = [&](std::size_t key, double p50_off, bool with_anchors) {
        std::vector<budget_t> budgets;
        for (double f : ET_FRACTIONS)
            budgets.push_back({f * p50_off, "", nan_v});
        for (double ms : ET_MS_VALUES)
            budgets.push_back({ms, "", nan_v});
        if (with_anchors && !ET_ANCHORS.empty()) {
            auto it = anchor_ms.find(key);
            for (std::size_t ai = 0; ai < ET_ANCHORS.size(); ++ai) {
                double const base = it == anchor_ms.end() ? nan_v : it->second[ai];
                for (double f : ET_ANCHORS[ai].fractions)
                    budgets.push_back({f * base, ET_ANCHORS[ai].name, f});
            }
        }
        return budgets;
    };
    // Budget rows of one graph series. Called right after the series' unbudgeted
    // row, or later from the pending list once the anchors exist.
    auto run_ladder = [&](test_case_t const& tc, double br, std::size_t ef, std::string const& label,
                          double p50_off, std::size_t key) {
        auto factory = make_modulo_factory(tc.modulus, tc.pred_threshold);
        for (auto const& b : budget_ladder(key, p50_off, true)) {
            std::string lbl = label + " et=" + format_ms(b.ms) + "ms";
            if (!b.anchor.empty())
                lbl += " (" + b.anchor + " x" + format_frac(b.fraction) + ")";
            auto t = run_search(index, metric, queries, K_VAL, ef, factory, tc.filter_first_threshold, false,
                                lbl.c_str(), tc.mode, tc.selectivity, br, gt_cache[key], tc.navix, b.ms,
                                tc.three_hop);
            t.early_termination_ms = b.ms;
            t.et_anchor = b.anchor;
            t.et_fraction = b.fraction;
            summaries.push_back(t);
            std::cout.flush();
        }
    };

    for (auto const& tc : cases) {
        if (!selectivity_matches(tc.selectivity))
            continue;
        if (!case_enabled(tc))
            continue;
        ++search_done;
        std::size_t eligible = static_cast<std::size_t>(N_VAL * tc.selectivity);
        std::cout << "\n--- [" << search_done << "/" << search_total << "] " << tc.label << " ---";
        std::cout << "\n  Eligible vectors : ~" << eligible << " / " << N_VAL << " (" << std::fixed
                  << std::setprecision(1) << (tc.selectivity * 100.0) << "%)\n";

        std::size_t key = gt_key(tc.modulus, tc.pred_threshold);
        if (have_pending_key && key != pending_key)
            fail_if_pending();
        pending_key = key;
        have_pending_key = true;

        if (tc.prefilter) {
            bool const per_query_scan = iequals(tc.mode, "Prefilter-scan");
            std::string label = tc.label;
            if (et_ladder_enabled())
                label += " et=off";
            auto s = run_prefilter_search(metric, queries, K_VAL, tc.modulus, tc.pred_threshold, label.c_str(),
                                          tc.mode, tc.selectivity, gt_cache[key], 0, -1.0, per_query_scan);
            summaries.push_back(s);
            std::cout.flush();
            // Prefilter rows take the self-relative and absolute ladders only. Anchor
            // budgets are a graph-series comparison, and prefilter runs before the anchors.
            for (auto const& b : budget_ladder(key, s.latency_p50, false)) {
                std::string lbl = tc.label + std::string(" et=") + format_ms(b.ms) + "ms";
                auto t = run_prefilter_search(metric, queries, K_VAL, tc.modulus, tc.pred_threshold, lbl.c_str(),
                                              tc.mode, tc.selectivity, gt_cache[key], EF_SEARCH_VAL, b.ms,
                                              per_query_scan);
                t.early_termination_ms = b.ms;
                summaries.push_back(t);
                std::cout.flush();
            }
            continue;
        }

        bool const is_racorn = iequals(tc.mode, "RACORN-1");
        std::vector<double> const br_list = is_racorn ? bbr_sweep : std::vector<double>{tc.bridge_budget_ratio};
        // Exact search ignores ef; run it once regardless of the sweep.
        std::vector<std::size_t> const ef_list =
            tc.exact ? std::vector<std::size_t>{EF_SEARCH_VAL} : ef_list_for(tc.mode);

        for (double br : br_list) {
            for (std::size_t ef : ef_list) {
                std::string label = tc.label;
                if (label_ef && !tc.exact)
                    label += " ef=" + std::to_string(ef);
                if (is_racorn && bbr_sweep.size() > 1)
                    label += " br=" + format_br(br);
                auto factory = make_modulo_factory(tc.modulus, tc.pred_threshold);
                bool const skip_off = ET_SKIP_UNBUDGETED && !tc.exact && ET_FRACTIONS.empty() && !ET_MS_VALUES.empty();
                search_result_summary_t s{};
                if (!skip_off) {
                    std::string off_label = label + (et_ladder_enabled() && !tc.exact ? " et=off" : "");
                    s = run_search(index, metric, queries, K_VAL, ef, factory, tc.filter_first_threshold, tc.exact,
                                   off_label.c_str(), tc.mode, tc.selectivity, br, gt_cache[key], tc.navix,
                                   EARLY_TERMINATION_MS, tc.three_hop);
                    s.early_termination_ms = EARLY_TERMINATION_MS;
                    summaries.push_back(s);
                    // Persist the log immediately after each cell completes, so the file
                    // reflects the in-progress state if the process is interrupted later.
                    std::cout.flush();
                }
                // Is this unbudgeted row an --et-anchor? Record its prefill p50 for
                // the whole selectivity. A br sweep makes the anchor ambiguous.
                if (!skip_off && !tc.exact) {
                    for (std::size_t ai = 0; ai < ET_ANCHORS.size(); ++ai) {
                        auto const& a = ET_ANCHORS[ai];
                        if (!iequals(a.mode, tc.mode) || a.ef != ef)
                            continue;
                        if (is_racorn && bbr_sweep.size() > 1) {
                            std::cerr << "Error: --et-anchor " << a.name << " on a mode with a br sweep is ambiguous\n";
                            std::exit(1);
                        }
                        if (std::isnan(s.prefill_ms_p50)) {
                            std::cerr << "Error: anchor series " << a.name << " never filled W at selectivity "
                                      << tc.selectivity << " — no deadline reference for this cell\n";
                            std::exit(1);
                        }
                        auto& slot = anchor_ms[key];
                        if (slot.empty())
                            slot.assign(ET_ANCHORS.size(), nan_v);
                        slot[ai] = s.prefill_ms_p50;
                        std::cout << "  [et-anchor] " << a.name << " prefill p50 = " << format_ms(s.prefill_ms_p50)
                                  << " ms\n";
                    }
                }
                // Timeout ladder (graph modes only). With anchors, the rows wait until
                // every anchor of this selectivity exists. The waiting series then run
                // first, in their original order, so the JSON keeps the series order.
                if (!tc.exact) {
                    if (anchors_ready(key)) {
                        for (auto const& p : pending_ladders)
                            run_ladder(*p.tc, p.br, p.ef, p.label, p.p50_off, key);
                        pending_ladders.clear();
                        run_ladder(tc, br, ef, label, s.latency_p50, key);
                    } else {
                        pending_ladders.push_back({&tc, br, ef, label, s.latency_p50});
                    }
                }
                if (!tc.exact && !skip_off && RECALL_TARGET > 0 && s.recall >= RECALL_TARGET)
                    break; // target recall reached; larger ef adds no curve information
            }
        }
    }
    if (have_pending_key)
        fail_if_pending();

    // ── Summary table ───────────────────────────────────────────────────────
    std::cout << "\n========================================================\n";
    std::cout << " Summary\n";
    std::cout << "========================================================\n\n";

    print_summary_config(/*qps=*/false);
    std::cout << "\n";

    std::cout << std::left << std::setw(14) << "Selectivity" << std::setw(11) << "Mode" << std::right << std::setw(8)
              << "EF" << std::setw(7) << "BR" << std::setw(9) << "ET(ms)" << std::setw(12) << "Latency(ms)"
              << std::setw(10) << "QPS"
              << std::setw(14) << "Dist.Computed"
              << std::setw(14) << "NodesVisited" << std::setw(10) << "Recall(avg)" << std::setw(12) << "RecallRatio"
              << std::setw(10) << "Speedup" << std::setw(10) << "DistRatio" << std::setw(12) << "PredEvals";
    if (early_termination_enabled())
        std::cout << std::setw(12) << "EarlyTerm";
    std::cout << "\n";
    // Rule widens with the optional early-termination column.
    std::size_t const table_width = early_termination_enabled() ? 165 : 153;
    std::cout << std::string(table_width, '-') << "\n";

    // Shared row printer
    auto print_row = [&](std::string const& sel, std::string const& mode, search_result_summary_t const& r,
                         search_result_summary_t const* hnsw_baseline) {
        std::cout << std::left << std::setw(14) << sel << std::setw(11) << mode << std::right;
        if (r.ef_search > 0)
            std::cout << std::setw(8) << r.ef_search;
        else
            std::cout << std::setw(8) << "-";
        if (r.bridge_budget_ratio > 0)
            std::cout << std::fixed << std::setprecision(1) << std::setw(7) << r.bridge_budget_ratio;
        else
            std::cout << std::setw(7) << "-";
        if (r.early_termination_ms >= 0)
            std::cout << std::fixed << std::setprecision(2) << std::setw(9) << r.early_termination_ms;
        else
            std::cout << std::setw(9) << "-";
        std::cout << std::fixed << std::setprecision(1) << std::setw(12) << r.latency_ms << std::setprecision(0)
                  << std::setw(10) << r.qps << std::setw(14) << r.dist_computed << std::setw(14) << r.nodes_visited
                  << std::setprecision(4) << std::setw(10) << r.recall;
        if (hnsw_baseline && &r != hnsw_baseline && !r.exact && r.latency_ms > 0 && r.dist_computed > 0 &&
            hnsw_baseline->recall > 0 && std::isfinite(r.latency_ms) && std::isfinite(r.dist_computed)) {
            double sp = hnsw_baseline->latency_ms / r.latency_ms;
            double dr = hnsw_baseline->dist_computed / r.dist_computed;
            double rr = r.recall / hnsw_baseline->recall;
            std::cout << std::setprecision(2) << std::setw(11) << rr << "x" << std::setprecision(1) << std::setw(9)
                      << sp << "x" << std::setw(9) << dr << "x";
        } else {
            std::cout << std::setw(12) << "" << std::setw(10) << "" << std::setw(10) << "";
        }
        if (r.predicate_evals > 0)
            std::cout << std::setprecision(0) << std::setw(12) << r.predicate_evals;
        else
            std::cout << std::setw(12) << "";
        if (early_termination_enabled())
            std::cout << std::setprecision(1) << std::setw(11) << (r.early_term_rate * 100.0) << "%";
        std::cout << "\n";
    };

    // Baseline row (no-predicate HNSW), only present when show_baseline was true
    std::size_t group_start = 0;
    if (show_baseline) {
        auto const& b = summaries[0];
        print_row("Baseline", b.mode, b, nullptr);
        std::cout << std::string(table_width, '-') << "\n";
        group_start = 1;
    }

    // Group remaining summaries by selectivity (contiguous with same selectivity value)
    for (std::size_t i = group_start; i < summaries.size();) {
        std::size_t j = i;
        double sel_val = summaries[i].selectivity;
        while (j < summaries.size() && std::fabs(summaries[j].selectivity - sel_val) < 1e-12)
            ++j;

        // Find HNSW within [i, j) as the ratio baseline. With an ef sweep the
        // group holds many HNSW rows at different ef, so ratios are suppressed.
        search_result_summary_t const* hnsw_p = nullptr;
        for (std::size_t k = i; k < j && !EF_VALUES_SET; ++k) {
            if (iequals(summaries[k].mode, "HNSW")) {
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
            print_row(first ? sel_str : std::string{}, summaries[k].mode, summaries[k], hnsw_p);
            first = false;
        }
        if (j < summaries.size())
            std::cout << std::string(table_width, '-') << "\n";

        i = j;
    }

    if (!JSON_PATH.empty())
        write_json_results(JSON_PATH, summaries);

    std::cout << "\n========================================================\n";
    std::cout << " Benchmark complete.\n";
    std::cout << "========================================================\n";
    return 0;
}

// ── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Force line/unit buffering so logs are persisted incrementally even when stdout is
    // redirected to a file. The default block buffering hides progress until exit / SIGKILL.
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    parse_args(argc, argv);
    std::cout << "========================================================\n";
    std::cout << " index_gt ACORN-1 Performance Benchmark\n";
    std::cout << "========================================================\n";
    std::cout << "  Vectors     : " << N_VAL << "\n";
    std::cout << "  Dimensions  : " << D_VAL << "\n";
    std::cout << "  Connectivity: " << M_VAL << "\n";
    std::cout << "  EF construct: " << EF_CONSTRUCTION_VAL << "\n";
    std::cout << "  K           : " << K_VAL << "\n";
    std::cout << "  EF search   : " << EF_SEARCH_VAL << "\n";
    std::cout << "  ACORN thresh: " << ACORN_THRESHOLD << "\n";
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

    // Load dataset. One common path driven by a per-dataset spec table; DATASET_NAME
    // is already validated by parse_args.
    struct dataset_spec_t {
        char const* name;
        char const* base_file;
        char const* query_file;
        bool fbin;
        bool u8bin;
    };
    static dataset_spec_t const dataset_specs[] = {
        {"sift", "sift_base.fvecs", "sift_query.fvecs", false, false},
        {"gist", "gist_base.fvecs", "gist_query.fvecs", false, false},
        {"text2image", "text2image-40M.fbin", "text2image-query-100K.fbin", true, false},
        {"laion", "laion_base.fvecs", "laion_query.fvecs", false, false},
        {"yfcc", "base.10M.u8bin", "query.public.100K.u8bin", false, true},
    };

    {
        dataset_spec_t const* spec = nullptr;
        for (auto const& s : dataset_specs)
            if (DATASET_NAME == s.name) {
                spec = &s;
                break;
            }

        std::vector<float> base_data;
        std::size_t base_count = 0, base_dims = 0;
        bool loaded = false;

        std::string base_path = DATA_DIR + "/" + spec->base_file;
        // Taken before the load, so that every GT path built later carries it. This also
        // holds when the vectors come from the volume and the base file is only fingerprinted.
        BASE_FINGERPRINT = file_fingerprint(base_path);
        {
            stopwatch_t load_timer;
            bool ok = spec->u8bin ? load_u8bin(base_path.c_str(), base_data, base_count, base_dims, N_VAL)
                      : spec->fbin ? load_fbin(base_path.c_str(), base_data, base_count, base_dims, N_VAL)
                                   : load_fvecs(base_path.c_str(), base_data, base_count, base_dims);
            if (ok) {
                std::cout << "Loaded " << spec->name << " base vectors from " << base_path << ": " << base_count
                          << " x " << base_dims << "d (" << std::fixed << std::setprecision(1)
                          << load_timer.elapsed_ms() << " ms)\n";
                loaded = true;
            }
        }

        if (!loaded) {
            std::cerr << spec->name << " base file not found: " << base_path << "\n";
            return 1;
        }
        if (base_dims != D_VAL || base_count < N_VAL) {
            std::cerr << spec->name << " dimensions/count mismatch (got " << base_dims << "d, " << base_count
                      << " vecs)\n";
            return 1;
        }
        if (base_count > N_VAL)
            base_data.resize(N_VAL * D_VAL);
        store_ptr = std::make_unique<vector_store_t<float>>(std::move(base_data), D_VAL);

        // Random-attribute predicate: assign one uniform uint32 per key, seed-fixed
        // and in key order, so every run (and the GT build) sees the same column.
        if (predicate_random()) {
            stopwatch_t attr_timer;
            std::mt19937_64 rng(PREDICATE_SEED);
            RANDOM_ATTR.resize(N_VAL);
            for (std::size_t i = 0; i < N_VAL; ++i)
                RANDOM_ATTR[i] = static_cast<std::uint32_t>(rng() >> 32);
            std::cout << "Assigned random attributes to " << N_VAL << " keys (seed " << PREDICATE_SEED << ", "
                      << std::fixed << std::setprecision(1) << attr_timer.elapsed_ms() << " ms)\n";
        }
        std::cout << "Using " << spec->name << " dataset (" << N_VAL << " vectors, " << D_VAL << "d)\n";

        std::string query_path = DATA_DIR + "/" + spec->query_file;
        std::size_t q_count = 0, q_dims = 0;
        stopwatch_t load_timer;
        bool q_ok = spec->u8bin ? load_u8bin(query_path.c_str(), query_vecs_data, q_count, q_dims)
                    : spec->fbin ? load_fbin(query_path.c_str(), query_vecs_data, q_count, q_dims)
                                 : load_fvecs(query_path.c_str(), query_vecs_data, q_count, q_dims);
        if (q_ok && q_dims == D_VAL) {
            num_queries = q_count;
            std::cout << "Loaded " << spec->name << " query vectors: " << q_count << " x " << q_dims << "d ("
                      << std::fixed << std::setprecision(1) << load_timer.elapsed_ms() << " ms)\n";
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
    else if (METRIC_NAME == "ip")
        ret = run_benchmark<ip_metric_t<float, vec_key_t, slot_t>>(
            *store_ptr, num_queries > 0 ? query_vecs_data.data() : nullptr, num_queries, "IP");
    else
        ret = run_benchmark<l2_metric_t<float, vec_key_t, slot_t>>(
            *store_ptr, num_queries > 0 ? query_vecs_data.data() : nullptr, num_queries, "L2sq");

    return ret;
}
