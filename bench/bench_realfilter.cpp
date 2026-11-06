// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 * @file bench_realfilter.cpp
 * @brief Filtered ANN benchmark with REAL, per-query predicates and
 *        natural-selectivity bucketing.
 *
 * bench_acorn.cpp sweeps one controlled predicate over a selectivity grid. Here,
 * instead, every query carries its own predicate (e.g. a tag-AND filter from
 * YFCC-10M metadata). The natural selectivity of each query (the passing fraction
 * of the base) is measured. Queries are grouped into selectivity buckets, and a
 * latency-recall ef-sweep is run per bucket for each mode. Ground truth is exact
 * filtered search at --k, cached. --use-official-gt switches the recall reference
 * to the official ground-truth file given by --official-gt.
 *
 * Modes: prefilter / prefilter-scan / prefilter-index / hnsw / acorn-1 / acorn-1-3hop /
 * navix-local (and the other navix-* heuristics) / racorn-1 (br sweep). Four things
 * place the natural filters on the same axes as the controlled predicates of
 * Sections VI-A and VI-B: the three-hop variant, the stride switch, the per-bucket
 * correlation measurement c_e and the `groups[]` summary.
 *
 * The JSON output has the same shape as bench_acorn's (config + results rows). The
 * per-bucket representative selectivity is written to the `selectivity` field and the
 * bucket label to `group`. `label` has the form "<bucket> | <mode> ef=<ef>".
 *
 * cmpkey is intentionally NOT used (cmpkey OFF, as in the other drivers by default).
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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;

// ── u8bin loader ────────────────────────────────────────────────────────────
// Big-ANN format: [uint32 n][uint32 d][uint8 x n*d]. Converted to float.
static bool load_u8bin(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims,
                       std::size_t max_count = 0) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    uint32_t n = 0, d = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), 4) || !ifs.read(reinterpret_cast<char*>(&d), 4))
        return false;
    if (n == 0 || d == 0 || d > 100000)
        return false;
    count = n;
    dims = d;
    if (max_count > 0 && count > max_count)
        count = max_count;
    std::vector<uint8_t> raw(count * dims);
    if (!ifs.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size())))
        return false;
    data.resize(count * dims);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<float>(raw[i]);
    return true;
}

// ── sparse metadata (.spmat) loader ─────────────────────────────────────────
// Big-ANN format: [int64 nrow][int64 ncol][int64 nnz][int64 indptr[nrow+1]]
// [int32 indices[nnz]][float32 data[nnz]]. We keep indptr+indices (tag ids per
// row); `data` (all 1.0) is skipped. Each row's tag slice is sorted ascending
// so membership tests can binary-search.
struct csr_tags_t {
    int64_t nrow = 0, ncol = 0, nnz = 0;
    std::vector<int64_t> indptr; // size nrow+1
    std::vector<int32_t> indices;

    int32_t const* row_begin(std::size_t i) const { return indices.data() + indptr[i]; }
    int32_t const* row_end(std::size_t i) const { return indices.data() + indptr[i + 1]; }
    std::size_t row_len(std::size_t i) const { return static_cast<std::size_t>(indptr[i + 1] - indptr[i]); }

    bool has_tag(std::size_t row, int32_t tag) const {
        return std::binary_search(row_begin(row), row_end(row), tag);
    }
};

static bool load_spmat(char const* path, csr_tags_t& m) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    int64_t sizes[3];
    if (!ifs.read(reinterpret_cast<char*>(sizes), sizeof(sizes)))
        return false;
    m.nrow = sizes[0];
    m.ncol = sizes[1];
    m.nnz = sizes[2];
    if (m.nrow <= 0 || m.nnz < 0)
        return false;
    m.indptr.resize(m.nrow + 1);
    if (!ifs.read(reinterpret_cast<char*>(m.indptr.data()), (m.nrow + 1) * (int64_t)sizeof(int64_t)))
        return false;
    m.indices.resize(m.nnz);
    if (m.nnz > 0 && !ifs.read(reinterpret_cast<char*>(m.indices.data()), m.nnz * (int64_t)sizeof(int32_t)))
        return false;
    // data[] (float32 x nnz) intentionally not read.
    for (int64_t i = 0; i < m.nrow; ++i)
        std::sort(m.indices.begin() + m.indptr[i], m.indices.begin() + m.indptr[i + 1]);
    return true;
}

// ── inverted tag index (prefilter-index) ────────────────────────────────────
// Transpose of the base CSR: for every tag, the ascending list of rows that
// carry it. This is what a system with a per-attribute index consults. A query
// then pays for reading and intersecting the posting lists of its tags, instead
// of testing every row. Built once, only when a prefilter-index arm asks.
struct tag_postings_t {
    std::vector<int64_t> start;      // size ntags+1
    std::vector<std::uint32_t> rows; // size nnz, ascending within each tag
    std::size_t ntags = 0;

    void build(csr_tags_t const& m) {
        int64_t max_tag = m.ncol - 1;
        for (int64_t i = 0; i < m.nnz; ++i)
            if (m.indices[i] > max_tag)
                max_tag = m.indices[i];
        ntags = max_tag < 0 ? 0 : (std::size_t)max_tag + 1;
        start.assign(ntags + 1, 0);
        for (int64_t i = 0; i < m.nnz; ++i)
            ++start[(std::size_t)m.indices[i] + 1];
        for (std::size_t t = 0; t < ntags; ++t)
            start[t + 1] += start[t];
        rows.resize((std::size_t)m.nnz);
        std::vector<int64_t> cur(start.begin(), start.end() - 1);
        for (int64_t r = 0; r < m.nrow; ++r)
            for (int64_t j = m.indptr[r]; j < m.indptr[r + 1]; ++j)
                rows[(std::size_t)cur[(std::size_t)m.indices[j]]++] = (std::uint32_t)r;
    }
    std::uint32_t const* list_begin(int32_t t) const { return rows.data() + start[(std::size_t)t]; }
    std::size_t list_len(int32_t t) const {
        return (std::size_t)(start[(std::size_t)t + 1] - start[(std::size_t)t]);
    }
    bool holds(int32_t t) const { return t >= 0 && (std::size_t)t < ntags; }
};

// ── ibin (official GT) loader ───────────────────────────────────────────────
// [uint32 n][uint32 d][int32 I[n*d]][float32 D[n*d]]. Returns I only.
static bool load_ibin_ids(char const* path, std::vector<int32_t>& ids, std::size_t& n, std::size_t& d) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    uint32_t nn = 0, dd = 0;
    if (!ifs.read(reinterpret_cast<char*>(&nn), 4) || !ifs.read(reinterpret_cast<char*>(&dd), 4))
        return false;
    n = nn;
    d = dd;
    ids.resize((std::size_t)nn * dd);
    if (!ifs.read(reinterpret_cast<char*>(ids.data()), (std::streamsize)ids.size() * 4))
        return false;
    return true;
}

// ── vector store / metric / index infra (mirrors bench_acorn) ───────────────
template <typename T = float> class vector_store_t {
  public:
    vector_store_t(std::vector<T>&& data, std::size_t dims) : dims_(dims), data_(std::move(data)) {}
    std::size_t size() const noexcept { return data_.size() / dims_; }
    std::size_t dims() const noexcept { return dims_; }
    T const* row(std::size_t i) const noexcept { return data_.data() + i * dims_; }

  private:
    std::size_t dims_;
    std::vector<T> data_;
};

using vec_key_t = std::size_t;
using slot_t = std::size_t;
using index_t = index_gt<float, vec_key_t, slot_t>;

template <typename kernel_at> struct dataset_metric_t {
    using member_cref_t = typename index_t::member_cref_t;
    using member_citerator_t = typename index_t::member_citerator_t;
    using kernel_t = kernel_at;
    vector_store_t<float> const* store = nullptr;
    std::size_t dimensions = 0;
    float const* row(std::size_t s) const noexcept { return store->row(s); }
    float operator()(member_cref_t const& a, member_cref_t const& b) const {
        return kernel_at{}(row(get_key(b)), row(get_key(a)), dimensions);
    }
    float operator()(float const* q, member_cref_t const& m) const { return kernel_at{}(q, row(get_key(m)), dimensions); }
    float operator()(member_citerator_t const& a, member_citerator_t const& b) const {
        return kernel_at{}(row(get_key(*b)), row(get_key(*a)), dimensions);
    }
    float operator()(float const* q, member_citerator_t const& m) const {
        return kernel_at{}(q, row(get_key(*m)), dimensions);
    }
};
using l2_metric_t = dataset_metric_t<metric_l2sq_gt<float, float>>;

template <typename index_at> struct aligned_index_t {
    using alloc_t = aligned_allocator_gt<index_at, 64>;
    index_at* ptr = nullptr;
    template <typename... A> explicit aligned_index_t(A&&... a) {
        ptr = alloc_t{}.allocate(1);
        new (ptr) index_at(std::forward<A>(a)...);
    }
    ~aligned_index_t() {
        if (ptr) {
            ptr->~index_at();
            alloc_t{}.deallocate(ptr, 1);
        }
    }
    aligned_index_t(aligned_index_t const&) = delete;
    aligned_index_t& operator=(aligned_index_t const&) = delete;
};

struct stopwatch_t {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start{clock::now()};
    double ms() const { return std::chrono::duration<double, std::milli>(clock::now() - start).count(); }
};

static double median(std::vector<double> v) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return std::isnan(x); }), v.end());
    if (v.empty())
        return std::nan("");
    std::size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2)
        return v[n / 2];
    double hi = v[n / 2];
    return (*std::max_element(v.begin(), v.begin() + n / 2) + hi) / 2.0;
}
static double pct_sorted(std::vector<double> const& s, double p) {
    if (s.empty())
        return 0.0;
    if (p <= 0)
        return s.front();
    if (p >= 1)
        return s.back();
    double idx = p * (s.size() - 1);
    std::size_t lo = (std::size_t)idx, hi = lo + 1;
    if (hi >= s.size())
        return s.back();
    double f = idx - lo;
    return s[lo] * (1 - f) + s[hi] * f;
}

// ── persistent worker pool (mirrors bench_acorn) ────────────────────────────
class query_pool_t {
  public:
    query_pool_t(std::size_t st, std::size_t nq) : st_(st), nq_(nq), chunk_((nq + st - 1) / st) {
        if (st_ <= 1)
            return;
        for (std::size_t t = 0; t < st_; ++t)
            workers_.emplace_back([this, t] { loop(t); });
    }
    ~query_pool_t() {
        if (st_ <= 1)
            return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& w : workers_)
            w.join();
    }
    query_pool_t(query_pool_t const&) = delete;
    query_pool_t& operator=(query_pool_t const&) = delete;
    template <typename Fn> void run(Fn&& fn) {
        if (st_ <= 1) {
            for (std::size_t q = 0; q < nq_; ++q)
                fn(q, (std::size_t)0);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            task_ = std::forward<Fn>(fn);
            pending_ = st_;
            ++gen_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(mu_);
        done_.wait(lk, [this] { return pending_ == 0; });
    }

  private:
    void loop(std::size_t t) {
        std::size_t b = t * chunk_, e = std::min(b + chunk_, nq_), mine = 0;
        std::function<void(std::size_t, std::size_t)> task;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this, mine] { return stop_ || gen_ > mine; });
                if (stop_)
                    return;
                mine = gen_;
                task = task_;
            }
            for (std::size_t q = b; q < e; ++q)
                task(q, t);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (--pending_ == 0)
                    done_.notify_one();
            }
        }
    }
    std::size_t const st_, nq_, chunk_;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_;
    std::function<void(std::size_t, std::size_t)> task_;
    std::size_t gen_ = 0, pending_ = 0;
    bool stop_ = false;
};

// ── per-query tag-AND predicate ─────────────────────────────────────────────
// Passes a base vector iff it carries every tag the query requires.
struct tag_and_predicate_t {
    csr_tags_t const* base = nullptr;
    int32_t const* qtags = nullptr;
    std::size_t qn = 0;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        std::size_t key = get_key(m);
        for (std::size_t i = 0; i < qn; ++i)
            if (!base->has_tag(key, qtags[i]))
                return false;
        return true;
    }
};

// ── config ──────────────────────────────────────────────────────────────────
static std::size_t N_VAL = 10'000'000, D_VAL = 192, M_VAL = 16, K_VAL = 100, EF_CONSTRUCTION_VAL = 100;
static std::size_t NUM_QUERIES = 1000, SEARCH_THREADS = 32, BUILD_THREADS = 0, BENCH_RUNS = 5, BENCH_WARMUP = 2;
static double ACORN_THRESHOLD = 0.3, RECALL_TARGET = 0.97;
static std::vector<std::size_t> EF_VALUES = {100, 200, 300, 400, 500, 600, 700, 800,
                                             900, 1000, 1100, 1200, 1300, 1400, 1500, 1600};
static std::vector<double> BBR_VALUES = {1.0, 1.5, 2.0, 2.5};
static std::vector<std::string> MODES = {"prefilter", "hnsw", "acorn-1", "racorn-1"};
// Bucket upper edges (natural selectivity). A query with selectivity s lands in
// the first bucket whose upper edge is >= s. Focus band <= 5%.
static std::vector<double> BUCKET_EDGES = {0.001, 0.01, 0.05, 1.0}; // <0.1% / 0.1-1% / 1-5% / >5%
static std::string DATA_DIR = "data/raw-yfcc10m", VOLUME_DIR = "data/volumes", GT_DIR = "data/gt";
// Official big-ann GT (GT.public.ibin, k=10) for validation: report top-10 agreement with our
// self-generated exact GT, and optionally use it as the recall reference (needs --k <= 10).
static std::string OFFICIAL_GT = "";
static bool USE_OFFICIAL_GT = false;
static std::string JSON_PATH = "";
static bool BUILD_INDEX_ONLY = false;
static double BUILD_TIME_MS = -1.0;
static double THREE_HOP_TAU = 1.0;  // --three-hop-threshold: acorn-1-3hop trigger, in units of M0
static bool STRIDE_SAMPLING = true; // --stride on|off
static bool MEASURE_CE = true;      // --ce on|off: normalized mean rank of each query's passing set
static std::size_t NQ_USED = 0;
// Base-level average out-degree (index.stats(0): edges/nodes). Emitted as config.mavg, like
// the other two drivers, so the cost model can read it from any result file.
static double MAVG_BASE = 0.0;

// Per-bucket summary written to the JSON: how many queries landed in the bucket, the
// spread of their natural selectivity, the passing-set size, and the measured
// correlation c_e. c_e = 1 - 2 * mean normalized rank of the passing set: 0 = uncorrelated,
// +1 = the passing set is the query's nearest neighbours, -1 = its farthest. Same
// definition as bench_synthfilter, so a natural filter can be placed on the alpha axis
// of Section VI-B.
struct group_stat_t {
    std::string label;
    std::size_t nq, m_median;
    double sel_median, sel_p10, sel_p90, sel_mean;
    bool has_ce;
    double ce_mean, ce_p10, ce_p90;
};
static std::vector<group_stat_t> GROUP_STATS;

static bool iequals(std::string const& a, std::string const& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return true;
}
static int navix_of(std::string const& mode) {
    if (iequals(mode, "navix-onehop-s"))
        return 1;
    if (iequals(mode, "navix-blind"))
        return 2;
    if (iequals(mode, "navix-directed"))
        return 3;
    // adaptive-local: navix-local is the Kuzu artifact's form (8), as in bench_acorn;
    // navix-local-paper is the paper text's form (5).
    if (iequals(mode, "navix-local"))
        return 8;
    if (iequals(mode, "navix-local-paper"))
        return 5;
    return 0;
}

static std::string volume_path() {
    std::ostringstream o;
    o << VOLUME_DIR << "/yfcc_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef" << EF_CONSTRUCTION_VAL << ".usearch";
    return o.str();
}
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

static std::string gt_path() {
    std::ostringstream o;
    o << GT_DIR << "/yfcc_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef" << EF_CONSTRUCTION_VAL << "_realfilter"
      << "_d" << file_fingerprint(DATA_DIR + "/base.10M.u8bin") << "_t"
      << file_fingerprint(DATA_DIR + "/base.metadata.10M.spmat") << "_gt_k" << K_VAL << "_nq" << NUM_QUERIES
      << ".bin";
    return o.str();
}

// ── result row ──────────────────────────────────────────────────────────────
struct row_t {
    std::string label, mode, group;
    double selectivity; // bucket representative (median natural selectivity)
    std::size_t ef_search;
    double bridge_budget_ratio;
    double latency_ms, latency_p50, latency_p90, latency_p99;
    double recall, recall_p50, recall_p90, recall_p99;
    double dist_computed, nodes_visited, predicate_evals;
    double qps; // batch throughput = bucket queries / batch wall time (median over runs)
    std::size_t num_queries, threads;
};

static double recall_of(std::vector<vec_key_t> const& got, std::vector<vec_key_t> const& gt, std::size_t k) {
    std::size_t lim = std::min(gt.size(), k);
    if (!lim)
        return 0.0;
    std::unordered_set<vec_key_t> g(gt.begin(), gt.begin() + lim);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < got.size() && i < k; ++i)
        if (g.count(got[i]))
            ++hits;
    return (double)hits / lim;
}

// ── JSON (schema-compatible with bench_acorn) ───────────────────────────────
static void write_json(std::string const& path, std::vector<row_t> const& rows) {
    std::ofstream o(path);
    if (!o) {
        std::cerr << "cannot open json: " << path << "\n";
        return;
    }
    auto num = [](double v) {
        if (std::isnan(v))
            return std::string("null");
        std::ostringstream s;
        s << std::setprecision(6) << v;
        return s.str();
    };
    o << "{\n  \"config\": {\n    \"dataset\": \"yfcc\", \"n\": " << N_VAL << ", \"d\": " << D_VAL << ", \"m\": " << M_VAL
      << ", \"ef_construction\": " << EF_CONSTRUCTION_VAL << ",\n    \"k\": " << K_VAL << ", \"metric\": \"l2\""
      << ", \"filter\": \"tag-AND (real)\", \"gt_reference\": \"" << (USE_OFFICIAL_GT ? "official" : "self")
      << "\", \"cmpkey\": false, \"search_threads\": " << SEARCH_THREADS
      << ", \"build_time_ms\": " << num(BUILD_TIME_MS) << ",\n    \"filter_first_threshold\": " << num(ACORN_THRESHOLD)
      << ", \"three_hop_threshold\": " << num(THREE_HOP_TAU)
      << ", \"stride_sampling\": " << (STRIDE_SAMPLING ? "true" : "false") << ", \"nq\": " << NQ_USED
      << ", \"runs\": " << BENCH_RUNS << ", \"warmup\": " << BENCH_WARMUP << ", \"mavg\": " << num(MAVG_BASE)
      << ", \"bucket_edges\": [";
    for (std::size_t i = 0; i < BUCKET_EDGES.size(); ++i)
        o << (i ? ", " : "") << num(BUCKET_EDGES[i]);
    o << "],\n    \"groups\": [";
    for (std::size_t i = 0; i < GROUP_STATS.size(); ++i) {
        auto const& g = GROUP_STATS[i];
        o << (i ? ",\n      " : "\n      ") << "{\"label\": \"" << g.label << "\", \"nq\": " << g.nq
          << ", \"selectivity\": " << num(g.sel_median) << ", \"sel_median\": " << num(g.sel_median)
          << ", \"sel_p10\": " << num(g.sel_p10) << ", \"sel_p90\": " << num(g.sel_p90)
          << ", \"sel_mean\": " << num(g.sel_mean) << ", \"m\": " << g.m_median
          << ", \"ce_mean\": " << (g.has_ce ? num(g.ce_mean) : std::string("null"))
          << ", \"ce_p10\": " << (g.has_ce ? num(g.ce_p10) : std::string("null"))
          << ", \"ce_p90\": " << (g.has_ce ? num(g.ce_p90) : std::string("null")) << "}";
    }
    o << (GROUP_STATS.empty() ? "" : "\n    ") << "]\n  },\n  \"results\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto const& r = rows[i];
        o << "    {\"label\": \"" << r.label << "\", \"mode\": \"" << r.mode << "\", \"group\": \"" << r.group
          << "\", \"selectivity\": " << num(r.selectivity)
          << ", \"ef_search\": " << r.ef_search << ", \"bridge_budget_ratio\": " << num(r.bridge_budget_ratio)
          << ", \"exact\": false, \"threads\": " << r.threads << ", \"num_queries\": " << r.num_queries
          << ", \"latency_ms\": " << num(r.latency_ms) << ", \"latency_p50\": " << num(r.latency_p50)
          << ", \"latency_p90\": " << num(r.latency_p90) << ", \"latency_p99\": " << num(r.latency_p99)
          << ", \"recall\": " << num(r.recall) << ", \"recall_p50\": " << num(r.recall_p50)
          << ", \"recall_p90\": " << num(r.recall_p90) << ", \"recall_p99\": " << num(r.recall_p99)
          << ", \"dist_computed\": " << num(r.dist_computed) << ", \"nodes_visited\": " << num(r.nodes_visited)
          << ", \"predicate_evals\": " << num(r.predicate_evals) << ", \"qps\": " << num(r.qps) << "}"
          << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    o << "  ]\n}\n";
    std::cout << "Wrote JSON: " << path << " (" << rows.size() << " rows)\n";
}

static void parse_args(int argc, char** argv);

// ── main ──────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    parse_args(argc, argv);

    // 1) base vectors (u8 -> float)
    std::vector<float> base;
    std::size_t bn = 0, bd = 0;
    std::string base_path = DATA_DIR + "/base.10M.u8bin";
    {
        stopwatch_t t;
        if (!load_u8bin(base_path.c_str(), base, bn, bd, N_VAL)) {
            std::cerr << "failed to load " << base_path << "\n";
            return 1;
        }
        std::cout << "base: " << bn << " x " << bd << "d (" << std::fixed << std::setprecision(1) << t.ms() << " ms)\n";
    }
    if (bd != D_VAL) {
        std::cerr << "dim mismatch: " << bd << " != " << D_VAL << "\n";
        return 1;
    }
    if (bn != N_VAL) {
        std::cerr << "base has " << bn << " rows but --n " << N_VAL << " was requested; refusing (tag CSR and GT are"
                  << " indexed by key < N)\n";
        return 1;
    }
    auto store = std::make_unique<vector_store_t<float>>(std::move(base), D_VAL);

    // 2) base tags
    csr_tags_t base_tags;
    {
        stopwatch_t t;
        if (!load_spmat((DATA_DIR + "/base.metadata.10M.spmat").c_str(), base_tags)) {
            std::cerr << "failed base metadata\n";
            return 1;
        }
        std::cout << "base tags: nrow=" << base_tags.nrow << " nnz=" << base_tags.nnz << " (avg "
                  << std::setprecision(2) << (double)base_tags.nnz / base_tags.nrow << " tags/vec, " << std::setprecision(1)
                  << t.ms() << " ms)\n";
    }

    // 2b) inverted tag index, only when an arm needs it. The transpose costs
    // as much memory as the CSR itself.
    tag_postings_t postings;
    if (std::any_of(MODES.begin(), MODES.end(), [](std::string const& m) { return iequals(m, "prefilter-index"); })) {
        stopwatch_t t;
        postings.build(base_tags);
        std::cout << "tag postings: " << postings.ntags << " tags, " << postings.rows.size() << " entries ("
                  << std::setprecision(1) << t.ms() << " ms)\n";
    }

    // Build/load index
    index_config_t cfg(M_VAL);
    aligned_index_t<index_t> aidx(cfg);
    auto& index = *aidx.ptr;
    l2_metric_t metric{store.get(), D_VAL};
    std::size_t build_threads = BUILD_THREADS ? BUILD_THREADS : std::thread::hardware_concurrency();
    std::string vpath = volume_path();

    bool loaded = false;
    if (!BUILD_INDEX_ONLY) {
        std::ifstream probe(vpath);
        if (probe.good()) {
            std::cout << "loading volume " << vpath << "... " << std::flush;
            stopwatch_t t;
            if (index.load(vpath.c_str())) {
                if (index.size() != N_VAL) {
                    std::cerr << "volume " << vpath << " holds " << index.size() << " vectors but --n " << N_VAL
                              << " was requested; refusing\n";
                    return 1;
                }
                loaded = true;
                index.reserve(index_limits_t(index.size(), std::max<std::size_t>(SEARCH_THREADS, 1)));
                std::cout << "done (" << std::setprecision(1) << t.ms() << " ms, size " << index.size() << ")\n";
            } else
                std::cout << "failed, will rebuild\n";
        }
    }
    if (!loaded) {
        index.reserve(index_limits_t(N_VAL, std::max(build_threads, SEARCH_THREADS)));
        std::cout << "building index (" << build_threads << " threads)... " << std::flush;
        stopwatch_t t;
        std::vector<std::thread> pool;
        std::size_t chunk = (N_VAL + build_threads - 1) / build_threads;
        for (std::size_t tid = 0; tid < build_threads; ++tid) {
            std::size_t b = tid * chunk, e = std::min(b + chunk, N_VAL);
            pool.emplace_back([&, b, e, tid] {
                index_update_config_t u;
                u.expansion = EF_CONSTRUCTION_VAL;
                u.thread = tid;
                for (std::size_t i = b; i < e; ++i)
                    index.add((vec_key_t)i, store->row(i), metric, u);
            });
        }
        for (auto& th : pool)
            th.join();
        BUILD_TIME_MS = t.ms();
        std::cout << "done (" << std::setprecision(1) << BUILD_TIME_MS << " ms)\n";
        if (BUILD_INDEX_ONLY) {
            std::error_code ec;
            std::filesystem::create_directories(VOLUME_DIR, ec);
            if (!index.save(vpath.c_str())) {
                std::cerr << "save failed\n";
                return 1;
            }
            std::cout << "saved " << vpath << "\n";
            return 0;
        }
    }
    {
        auto st0 = index.stats(0);
        MAVG_BASE = st0.nodes ? (double)st0.edges / (double)st0.nodes : 0.0;
    }

    // 3) queries + query tags
    std::vector<float> qdata;
    std::size_t qn = 0, qd = 0;
    if (!load_u8bin((DATA_DIR + "/query.public.100K.u8bin").c_str(), qdata, qn, qd)) {
        std::cerr << "failed queries\n";
        return 1;
    }
    csr_tags_t qtags;
    if (!load_spmat((DATA_DIR + "/query.metadata.public.100K.spmat").c_str(), qtags)) {
        std::cerr << "failed query metadata\n";
        return 1;
    }
    std::size_t nq = std::min(NUM_QUERIES, qn);
    NQ_USED = nq;
    std::cout << "queries: using " << nq << " of " << qn << " (avg " << std::setprecision(2)
              << (double)qtags.nnz / qtags.nrow << " tags/query)\n";
    std::vector<float const*> queries(nq);
    for (std::size_t i = 0; i < nq; ++i)
        queries[i] = qdata.data() + i * D_VAL;

    auto make_pred = [&](std::size_t q) {
        return tag_and_predicate_t{&base_tags, qtags.row_begin(q), qtags.row_len(q)};
    };

    // 4) natural selectivity per query (parallel exact predicate scan). The passing keys are
    // kept: the "prefilter" arm (set handed over) scans exactly this set, and c_e ranks it.
    std::vector<double> qsel(nq);
    std::vector<std::vector<vec_key_t>> qpass(nq);
    {
        std::cout << "measuring per-query selectivity... " << std::flush;
        stopwatch_t t;
        query_pool_t pool(std::min(SEARCH_THREADS, nq), nq);
        pool.run([&](std::size_t q, std::size_t) {
            auto pred = make_pred(q);
            for (std::size_t i = 0; i < N_VAL; ++i) {
                bool ok = true;
                for (std::size_t ti = 0; ti < pred.qn; ++ti)
                    if (!base_tags.has_tag(i, pred.qtags[ti])) {
                        ok = false;
                        break;
                    }
                if (ok)
                    qpass[q].push_back((vec_key_t)i);
            }
            qsel[q] = (double)qpass[q].size() / N_VAL;
        });
        std::cout << "done (" << std::setprecision(1) << t.ms() << " ms)\n";
    }

    // 4b) measured correlation per query: c_e = 1 - 2 * mean normalized rank of the passing
    // set. rank(x) = the number of base vectors strictly closer to the query than x. One full
    // distance pass per query (O(N d)), then O(N log m) to count ranks against the sorted
    // passing distances. Same definition as bench_synthfilter.
    std::vector<double> qce(nq, std::nan(""));
    if (MEASURE_CE) {
        std::cout << "measuring c_e (normalized mean rank of the passing set)... " << std::flush;
        stopwatch_t t;
        query_pool_t pool(std::min(SEARCH_THREADS, nq), nq);
        pool.run([&](std::size_t q, std::size_t) {
            auto const& set = qpass[q];
            if (set.empty())
                return;
            thread_local std::vector<float> dd;
            dd.resize(N_VAL);
            float const* qv = queries[q];
            for (std::size_t i = 0; i < N_VAL; ++i)
                dd[i] = metric_l2sq_gt<float, float>{}(qv, metric.row(i), D_VAL);
            std::vector<float> ds(set.size());
            for (std::size_t j = 0; j < set.size(); ++j)
                ds[j] = dd[set[j]];
            std::sort(ds.begin(), ds.end());
            std::uint64_t rank_sum = 0;
            for (std::size_t i = 0; i < N_VAL; ++i)
                rank_sum += (std::uint64_t)(ds.end() - std::upper_bound(ds.begin(), ds.end(), dd[i]));
            double mean_norm_rank = (double)rank_sum / (double)set.size() / (double)N_VAL;
            qce[q] = 1.0 - 2.0 * mean_norm_rank;
        });
        std::cout << "done (" << std::setprecision(1) << t.ms() << " ms)\n";
    }

    // bucketize
    auto bucket_of = [&](double s) {
        for (std::size_t b = 0; b < BUCKET_EDGES.size(); ++b)
            if (s <= BUCKET_EDGES[b])
                return b;
        return BUCKET_EDGES.size() - 1;
    };
    std::vector<std::vector<std::size_t>> buckets(BUCKET_EDGES.size());
    for (std::size_t q = 0; q < nq; ++q)
        buckets[bucket_of(qsel[q])].push_back(q);
    auto bucket_label = [&](std::size_t b) {
        double lo = b ? BUCKET_EDGES[b - 1] : 0.0, hi = BUCKET_EDGES[b];
        std::ostringstream o;
        o << std::setprecision(3) << (lo * 100) << "-" << (hi * 100) << "%";
        return o.str();
    };
    std::cout << "bucket distribution:\n";
    for (std::size_t b = 0; b < buckets.size(); ++b) {
        auto const& qs = buckets[b];
        group_stat_t gs{bucket_label(b), qs.size(), 0, std::nan(""), std::nan(""), std::nan(""), std::nan(""),
                        false, 0, 0, 0};
        if (!qs.empty()) {
            std::vector<double> sels, ms, ces;
            for (auto q : qs) {
                sels.push_back(qsel[q]);
                ms.push_back((double)qpass[q].size());
                if (!std::isnan(qce[q]))
                    ces.push_back(qce[q]);
            }
            std::sort(sels.begin(), sels.end());
            gs.sel_median = pct_sorted(sels, 0.5);
            gs.sel_p10 = pct_sorted(sels, 0.1);
            gs.sel_p90 = pct_sorted(sels, 0.9);
            double sum = 0;
            for (double s : sels)
                sum += s;
            gs.sel_mean = sum / (double)sels.size();
            gs.m_median = (std::size_t)std::llround(median(ms));
            if (!ces.empty()) {
                std::sort(ces.begin(), ces.end());
                double cs = 0;
                for (double c : ces)
                    cs += c;
                gs.has_ce = true;
                gs.ce_mean = cs / (double)ces.size();
                gs.ce_p10 = pct_sorted(ces, 0.1);
                gs.ce_p90 = pct_sorted(ces, 0.9);
            }
        }
        GROUP_STATS.push_back(gs);
        std::cout << "  [" << gs.label << "] " << qs.size() << " queries";
        if (!qs.empty())
            std::cout << ", selectivity median " << std::setprecision(4) << gs.sel_median * 100 << "% (p10 "
                      << gs.sel_p10 * 100 << "%, p90 " << gs.sel_p90 * 100 << "%), passing median " << gs.m_median;
        if (gs.has_ce)
            std::cout << ", c_e mean " << std::setprecision(4) << gs.ce_mean << " p10 " << gs.ce_p10 << " p90 "
                      << gs.ce_p90;
        std::cout << "\n";
    }

    // 5) ground truth (exact filtered search at --k), cached; --use-official-gt may replace it below
    std::vector<std::vector<vec_key_t>> gt(nq);
    std::error_code ec;
    std::filesystem::create_directories(GT_DIR, ec);
    bool gt_loaded = false;
    // A cache file is usable when its header covers this run: at least nq queries and at
    // least k neighbours. The exact name is tried first. Failing that, any sibling with the
    // same key prefix and a larger _nq is accepted (same rule as bench_acorn/bench_synthfilter).
    // So a 992-query run reuses the 1000-query cache instead of recomputing the ground truth.
    std::string const gpath = gt_path();
    auto try_load = [&](std::string const& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs.good())
            return false;
        char magic[4] = {};
        uint32_t fnq = 0, fk = 0;
        ifs.read(magic, 4);
        ifs.read(reinterpret_cast<char*>(&fnq), 4);
        ifs.read(reinterpret_cast<char*>(&fk), 4);
        if (std::string(magic, 4) != "RFGT" || fnq < nq || fk < K_VAL)
            return false;
        for (auto& g : gt)
            g.clear();
        std::vector<uint64_t> buf(fk);
        for (std::size_t q = 0; q < fnq; ++q) {
            uint32_t cnt = 0;
            ifs.read(reinterpret_cast<char*>(&cnt), 4);
            ifs.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)fk * 8);
            if (q < nq) {
                std::size_t use = std::min<std::size_t>(cnt, K_VAL);
                for (std::size_t i = 0; i < use; ++i)
                    if (buf[i] != UINT64_MAX)
                        gt[q].push_back((vec_key_t)buf[i]);
            }
        }
        if (!ifs.good())
            return false;
        std::cout << "GT loaded: " << path;
        if (path != gpath)
            std::cout << " (nq " << fnq << " >= " << nq << ", prefix match)";
        std::cout << "\n";
        return true;
    };
    gt_loaded = try_load(gpath);
    if (!gt_loaded) {
        std::string prefix = gpath.substr(0, gpath.rfind("_nq"));
        std::error_code ec2;
        for (auto const& e : std::filesystem::directory_iterator(GT_DIR, ec2)) {
            std::string cand = e.path().string();
            if (cand.size() > prefix.size() && cand.compare(0, prefix.size(), prefix) == 0 &&
                cand.compare(prefix.size(), 3, "_nq") == 0 && try_load(cand)) {
                gt_loaded = true;
                break;
            }
        }
    }
    if (!gt_loaded) {
        std::cout << "computing exact GT (k=" << K_VAL << ", " << nq << " queries)... " << std::flush;
        stopwatch_t t;
        query_pool_t pool(std::min(SEARCH_THREADS, nq), nq);
        pool.run([&](std::size_t q, std::size_t tid) {
            index_search_config_t sc;
            sc.exact = true;
            sc.thread = tid;
            auto pred = make_pred(q);
            auto r = index.search(queries[q], K_VAL, metric, sc, pred, dummy_prefetch_t{}, dummy_preevaluate_t{});
            gt[q].reserve(r.size());
            for (std::size_t i = 0; i < r.size(); ++i)
                gt[q].push_back(r[i].member.key);
        });
        std::cout << "done (" << std::setprecision(1) << t.ms() << " ms)\n";
        std::ofstream ofs(gpath, std::ios::binary);
        ofs.write("RFGT", 4);
        uint32_t u = (uint32_t)nq;
        ofs.write(reinterpret_cast<char*>(&u), 4);
        u = (uint32_t)K_VAL;
        ofs.write(reinterpret_cast<char*>(&u), 4);
        std::vector<uint64_t> buf(K_VAL);
        for (std::size_t q = 0; q < nq; ++q) {
            uint32_t cnt = (uint32_t)gt[q].size();
            ofs.write(reinterpret_cast<char*>(&cnt), 4);
            std::fill(buf.begin(), buf.end(), UINT64_MAX);
            for (std::size_t i = 0; i < gt[q].size() && i < K_VAL; ++i)
                buf[i] = gt[q][i];
            ofs.write(reinterpret_cast<char*>(buf.data()), (std::streamsize)K_VAL * 8);
        }
        std::cout << "GT saved: " << gpath << "\n";
    }

    // 5b) validation against the official GT: same filter semantics (tag-AND) and metric (L2),
    // so the top-10 sets should agree up to distance ties.
    if (!OFFICIAL_GT.empty()) {
        std::vector<int32_t> oid;
        std::size_t on = 0, od = 0;
        if (!load_ibin_ids(OFFICIAL_GT.c_str(), oid, on, od) || on < nq || od == 0) {
            std::cerr << "failed official GT " << OFFICIAL_GT << " (n=" << on << ", k=" << od << ")\n";
            return 1;
        }
        std::size_t kk = std::min<std::size_t>({od, K_VAL, (std::size_t)10});
        if (USE_OFFICIAL_GT && K_VAL > kk) {
            // recall@K against a K'-deep reference with K' < K would count hits out of K
            // results against K' truths, and inflate recall. Refuse instead of reporting it.
            std::cerr << "--use-official-gt needs --k <= " << kk << " (official GT depth); got --k " << K_VAL << "\n";
            return 1;
        }
        std::vector<double> agree(nq);
        std::size_t exact = 0;
        for (std::size_t q = 0; q < nq; ++q) {
            std::unordered_set<vec_key_t> ours(gt[q].begin(), gt[q].begin() + std::min(kk, gt[q].size()));
            std::size_t hits = 0;
            for (std::size_t i = 0; i < kk; ++i) {
                int32_t id = oid[q * od + i];
                if (id >= 0 && ours.count((vec_key_t)id))
                    ++hits;
            }
            agree[q] = (double)hits / (double)kk;
            if (hits == kk)
                ++exact;
        }
        std::vector<double> sorted(agree);
        std::sort(sorted.begin(), sorted.end());
        double mean = 0;
        for (double a : agree)
            mean += a;
        mean /= nq;
        std::cout << "official GT (" << OFFICIAL_GT << ", k=" << od << ") vs self GT: agreement@" << kk
                  << " mean=" << std::setprecision(4) << mean << " p10=" << sorted[nq / 10] << " min=" << sorted[0]
                  << " exact=" << exact << "/" << nq << "\n";
        if (USE_OFFICIAL_GT) {
            for (std::size_t q = 0; q < nq; ++q) {
                gt[q].clear();
                for (std::size_t i = 0; i < kk; ++i)
                    if (oid[q * od + i] >= 0)
                        gt[q].push_back((vec_key_t)oid[q * od + i]);
            }
            std::cout << "recall reference switched to official GT@" << kk << "\n";
        }
    }

    // 6) sweep per (bucket, mode, [br], ef)
    std::vector<row_t> rows;
    for (std::size_t b = 0; b < buckets.size(); ++b) {
        auto const& qs = buckets[b];
        if (qs.empty())
            continue;
        // bucket representative selectivity = median natural selectivity
        std::vector<double> sels;
        for (auto q : qs)
            sels.push_back(qsel[q]);
        double rep = median(sels);
        std::string blabel = bucket_label(b);

        std::size_t st = std::min(SEARCH_THREADS, qs.size());
        query_pool_t pool(st, qs.size());

        for (auto const& mode : MODES) {
            // prefilter kinds: 1 = passing set handed over (optimistic, gathered outside the
            // timer), 2 = test the predicate on every row, 3 = read and intersect the query's
            // tag posting lists. Kinds 2 and 3 gather inside the timed region.
            int pf_kind = iequals(mode, "prefilter")         ? 1
                          : iequals(mode, "prefilter-scan")  ? 2
                          : iequals(mode, "prefilter-index") ? 3
                                                             : 0;
            bool is_prefilter = pf_kind > 0;
            bool is_racorn = iequals(mode, "racorn-1");
            bool is_3hop = iequals(mode, "acorn-1-3hop");
            int navix = navix_of(mode);
            double ff = (iequals(mode, "hnsw")) ? -1.0 : ACORN_THRESHOLD;
            double three_hop = is_3hop ? THREE_HOP_TAU : 0.0;
            std::vector<double> brs = is_racorn ? BBR_VALUES : std::vector<double>{0.0};

            for (double br : brs) {
                for (std::size_t ef : EF_VALUES) {
                    // metrics accumulators over bucket queries (per-query median across RUNS)
                    std::vector<double> qms(qs.size()), qrec(qs.size()), qdist(qs.size()), qvis(qs.size()),
                        qpe(qs.size());

                    // warmup
                    for (std::size_t w = 0; w < BENCH_WARMUP; ++w)
                        pool.run([&](std::size_t li, std::size_t tid) {
                            std::size_t q = qs[li];
                            auto pred = make_pred(q);
                            if (is_prefilter) {
                                (void)tid;
                            } else {
                                index_search_config_t sc;
                                sc.expansion = ef;
                                sc.filter_first_threshold = ff;
                                sc.bridge_budget_ratio = br;
                                sc.navix_heuristic = navix;
                                sc.three_hop_threshold = three_hop;
                                sc.stride_sampling = STRIDE_SAMPLING;
                                sc.thread = tid;
                                auto r =
                                    index.search(queries[q], K_VAL, metric, sc, pred, dummy_prefetch_t{}, dummy_preevaluate_t{});
                                (void)r;
                            }
                        });

                    // prefilter (set handed over): the passing keys were gathered once, outside any
                    // timer, when the natural selectivity was measured (qpass). Only the exact scan is timed.
                    std::vector<std::vector<double>> perq_ms(qs.size());
                    std::vector<double> batch_ms; // whole-bucket batch wall time per run → QPS
                    batch_ms.reserve(BENCH_RUNS);
                    for (int run = 0; run < (int)BENCH_RUNS; ++run) {
                        stopwatch_t batch_t;
                        pool.run([&](std::size_t li, std::size_t tid) {
                            std::size_t q = qs[li];
                            auto pred = make_pred(q);
                            stopwatch_t t;
                            std::vector<vec_key_t> got;
                            double dist = 0, vis = 0, pe = 0;
                            if (is_prefilter) {
                                // Kinds 2 and 3 pay for the passing set here; kind 1 got it free.
                                std::vector<vec_key_t> gathered;
                                std::size_t touched = 0;
                                if (pf_kind == 2) {
                                    for (std::size_t i = 0; i < N_VAL; ++i) {
                                        bool ok = true;
                                        for (std::size_t ti = 0; ti < pred.qn; ++ti)
                                            if (!base_tags.has_tag(i, pred.qtags[ti])) {
                                                ok = false;
                                                break;
                                            }
                                        if (ok)
                                            gathered.push_back((vec_key_t)i);
                                    }
                                    touched = N_VAL;
                                } else if (pf_kind == 3) {
                                    std::vector<int32_t> tags(pred.qtags, pred.qtags + pred.qn);
                                    // Shortest list first so the intersection shrinks fastest.
                                    std::sort(tags.begin(), tags.end(), [&](int32_t a, int32_t b) {
                                        return (postings.holds(a) ? postings.list_len(a) : 0) <
                                               (postings.holds(b) ? postings.list_len(b) : 0);
                                    });
                                    if (tags.empty()) {
                                        gathered.resize(N_VAL);
                                        for (std::size_t i = 0; i < N_VAL; ++i)
                                            gathered[i] = (vec_key_t)i;
                                    } else if (!postings.holds(tags[0])) {
                                        // unknown tag: empty posting list, so nothing passes
                                    } else {
                                        std::vector<std::uint32_t> cur(postings.list_begin(tags[0]),
                                                                       postings.list_begin(tags[0]) +
                                                                           postings.list_len(tags[0]));
                                        touched = cur.size();
                                        for (std::size_t j = 1; j < tags.size() && !cur.empty(); ++j) {
                                            if (!postings.holds(tags[j])) {
                                                cur.clear();
                                                break;
                                            }
                                            std::size_t L = postings.list_len(tags[j]);
                                            touched += L;
                                            std::vector<std::uint32_t> nxt;
                                            nxt.reserve(std::min(cur.size(), L));
                                            std::set_intersection(cur.begin(), cur.end(), postings.list_begin(tags[j]),
                                                                  postings.list_begin(tags[j]) + L,
                                                                  std::back_inserter(nxt));
                                            cur.swap(nxt);
                                        }
                                        gathered.assign(cur.begin(), cur.end());
                                    }
                                }
                                std::vector<vec_key_t> const& cand = pf_kind == 1 ? qpass[q] : gathered;
                                using pair_t = std::pair<float, vec_key_t>;
                                std::vector<pair_t> heap;
                                heap.reserve(K_VAL + 1);
                                float const* qv = queries[q];
                                for (vec_key_t key : cand) {
                                    float d = metric_l2sq_gt<float, float>{}(qv, metric.row(key), D_VAL);
                                    if (heap.size() < K_VAL) {
                                        heap.emplace_back(d, key);
                                        std::push_heap(heap.begin(), heap.end());
                                    } else if (d < heap.front().first) {
                                        std::pop_heap(heap.begin(), heap.end());
                                        heap.back() = {d, key};
                                        std::push_heap(heap.begin(), heap.end());
                                    }
                                }
                                std::sort_heap(heap.begin(), heap.end());
                                for (auto& p : heap)
                                    got.push_back(p.second);
                                dist = (double)cand.size();
                                vis = pf_kind == 1 ? dist : (double)touched; // keys touched to gather
                                pe = pf_kind == 2 ? (double)N_VAL : 0.0;
                            } else {
                                index_search_config_t sc;
                                sc.expansion = ef;
                                sc.filter_first_threshold = ff;
                                sc.bridge_budget_ratio = br;
                                sc.navix_heuristic = navix;
                                sc.three_hop_threshold = three_hop;
                                sc.stride_sampling = STRIDE_SAMPLING;
                                sc.thread = tid;
                                auto r =
                                    index.search(queries[q], K_VAL, metric, sc, pred, dummy_prefetch_t{}, dummy_preevaluate_t{});
                                for (std::size_t i = 0; i < r.size(); ++i)
                                    got.push_back(r[i].member.key);
                                dist = (double)r.computed_distances;
                                vis = (double)r.visited_members;
                                pe = (double)r.predicate_evaluations;
                            }
                            double el = t.ms();
                            perq_ms[li].push_back(el);
                            if (run == 0) {
                                qrec[li] = recall_of(got, gt[q], K_VAL);
                                qdist[li] = dist;
                                qvis[li] = vis;
                                qpe[li] = pe;
                            }
                        });
                        batch_ms.push_back(batch_t.ms());
                    }
                    for (std::size_t li = 0; li < qs.size(); ++li)
                        qms[li] = median(perq_ms[li]);
                    double med_batch_ms = median(batch_ms);
                    double qps = med_batch_ms > 0 ? (double)qs.size() / (med_batch_ms / 1000.0) : 0.0;

                    double med_ms = median(qms);
                    std::vector<double> ms_sorted(qms);
                    std::sort(ms_sorted.begin(), ms_sorted.end());
                    double avg_rec = 0;
                    for (double r : qrec)
                        avg_rec += r;
                    avg_rec /= qrec.size();
                    std::vector<double> rec_sorted(qrec);
                    std::sort(rec_sorted.begin(), rec_sorted.end());
                    row_t row;
                    row.label = blabel + " | " + mode + " ef=" + std::to_string(ef);
                    if (is_racorn)
                        row.label += " br=" + [&] { std::ostringstream o; o << std::setprecision(2) << br; return o.str(); }();
                    // The series names are those of bench_acorn.
                    row.mode = pf_kind == 1                 ? "Prefilter"
                               : pf_kind == 2               ? "Prefilter-scan"
                               : pf_kind == 3               ? "Prefilter-index"
                               : iequals(mode, "hnsw")      ? "HNSW"
                               : iequals(mode, "acorn-1")   ? "ACORN-1"
                               : is_3hop                    ? "ACORN-1-3hop"
                               : is_racorn                  ? "RACORN-1"
                               : navix == 8                 ? "NaviX-local"
                               : navix == 5                 ? "NaviX-local-paper"
                               : navix == 1                 ? "NaviX-onehop-s"
                               : navix == 2                 ? "NaviX-blind"
                               : navix == 3                 ? "NaviX-directed"
                                                            : mode;
                    row.group = blabel;
                    row.selectivity = rep;
                    row.ef_search = is_prefilter ? 0 : ef;
                    row.bridge_budget_ratio = is_racorn ? br : 0.0;
                    row.latency_ms = med_ms;
                    row.latency_p50 = pct_sorted(ms_sorted, 0.5);
                    row.latency_p90 = pct_sorted(ms_sorted, 0.9);
                    row.latency_p99 = pct_sorted(ms_sorted, 0.99);
                    row.recall = avg_rec;
                    row.recall_p50 = pct_sorted(rec_sorted, 0.5);
                    row.recall_p90 = pct_sorted(rec_sorted, 0.1);
                    row.recall_p99 = pct_sorted(rec_sorted, 0.01);
                    row.dist_computed = median(qdist);
                    row.nodes_visited = median(qvis);
                    row.predicate_evals = median(qpe);
                    row.qps = qps;
                    row.num_queries = qs.size();
                    row.threads = st;
                    rows.push_back(row);
                    std::cout << "  [" << blabel << "] " << row.mode << " ef=" << (is_prefilter ? 0 : ef)
                              << (is_racorn ? (" br=" + std::to_string(br)) : "") << " -> recall=" << std::setprecision(4)
                              << avg_rec << " lat=" << std::setprecision(2) << med_ms << "ms qps=" << std::setprecision(0)
                              << qps << "\n";
                    if (is_prefilter)
                        break; // ef-independent
                    if (RECALL_TARGET > 0 && avg_rec >= RECALL_TARGET)
                        break; // series reached target
                }
            }
        }
    }

    if (!JSON_PATH.empty())
        write_json(JSON_PATH, rows);
    std::cout << "done.\n";
    return 0;
}

// ── args ────────────────────────────────────────────────────────────────────
static void print_help() {
    std::cout
        << "Usage: bench_realfilter [options]\n"
        << "\n"
        << "The YFCC tag queries of the Big ANN filtered track, grouped by their natural selectivity.\n"
        << "The ground truth of each query is computed exactly in this process and, with --official-gt,\n"
        << "checked against the track's. One row per (method, bucket, ef, br).\n"
        << "\n"
        << "-- Data and index -------------------------------------------------------\n"
        << "  --n <N>                   Number of base vectors (default: 10000000)\n"
        << "  --m <N>                   Graph connectivity M of the volume (default: 16)\n"
        << "  --ef-construction <N>     ef_construction of the volume (default: 100)\n"
        << "  --build-index             Build the volume and exit\n"
        << "  --build-threads <N>       Threads of the build (default: all)\n"
        << "  --data-dir <dir>          The YFCC files (default: data/raw-yfcc10m)\n"
        << "  --volume-dir <dir>        Index volumes (default: data/volumes)\n"
        << "  --gt-dir <dir>            Ground-truth cache (default: data/gt)\n"
        << "  --official-gt <file>      The track's GT.public.ibin, compared with the exact one\n"
        << "  --use-official-gt         Score recall against the track's ground truth (needs --k <= 10)\n"
        << "\n"
        << "-- Groups ---------------------------------------------------------------\n"
        << "  --bucket-edges <list>     Upper edges of the selectivity buckets (default: 0.001,0.01,0.05,1.0)\n"
        << "  --nq <N>                  Queries (default: 1000)\n"
        << "\n"
        << "-- Search ---------------------------------------------------------------\n"
        << "  --modes <list>            hnsw, acorn-1, acorn-1-3hop, racorn-1, navix-local, prefilter,\n"
        << "                            prefilter-scan, prefilter-index (default: prefilter,hnsw,acorn-1,racorn-1)\n"
        << "  --ef-values <list>        The ef values, in order (default: 100,200,...,1600)\n"
        << "  --recall-target <F>       Stop a series once recall reaches this value (default: 0.97)\n"
        << "  --bridge-budget-ratios <list>  RACORN-1's br, one series per value (default: 1.0,1.5,2.0,2.5)\n"
        << "  --filter-first-threshold <F>   The entry gate of ACORN-1 and RACORN-1 (default: 0.3)\n"
        << "  --three-hop-threshold <F>      The trigger of acorn-1-3hop (default: 1.0)\n"
        << "  --stride on|off           Stride sampling or prefix truncation (default: on)\n"
        << "  --ce on|off               Measure the correlation c_e of each bucket (default: on)\n"
        << "  --k <N>                   Neighbors per query (default: 100)\n"
        << "  --runs <N> --warmup <N>   Timed runs (median) after warm-up passes (default: 5, 2)\n"
        << "  --search-threads <N>      Threads (default: 32)\n"
        << "  --json <path>             Write the result as JSON\n"
        << "  --help, -h                This text\n";
}

static void parse_args(int argc, char** argv) {
    auto need = [&](int& i) {
        if (i + 1 >= argc) {
            std::cerr << "missing value for " << argv[i] << "\n";
            std::exit(1);
        }
        return argv[++i];
    };
    auto split_sz = [](std::string s) {
        std::vector<std::size_t> v;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ','))
            if (!t.empty())
                v.push_back((std::size_t)std::stoull(t));
        return v;
    };
    auto split_d = [](std::string s) {
        std::vector<double> v;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ','))
            if (!t.empty())
                v.push_back(std::stod(t));
        return v;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--n")
            N_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--m")
            M_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--ef-construction")
            EF_CONSTRUCTION_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--k")
            K_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--nq")
            NUM_QUERIES = (std::size_t)std::stoull(need(i));
        else if (a == "--official-gt")
            OFFICIAL_GT = need(i);
        else if (a == "--use-official-gt")
            USE_OFFICIAL_GT = true;
        else if (a == "--search-threads")
            SEARCH_THREADS = (std::size_t)std::stoull(need(i));
        else if (a == "--build-threads")
            BUILD_THREADS = (std::size_t)std::stoull(need(i));
        else if (a == "--runs")
            BENCH_RUNS = (std::size_t)std::stoull(need(i));
        else if (a == "--warmup")
            BENCH_WARMUP = (std::size_t)std::stoull(need(i));
        else if (a == "--filter-first-threshold")
            ACORN_THRESHOLD = std::stod(need(i));
        else if (a == "--three-hop-threshold")
            THREE_HOP_TAU = std::stod(need(i));
        else if (a == "--stride" || a == "--ce") {
            std::string v = need(i);
            bool on = iequals(v, "on");
            if (!on && !iequals(v, "off")) {
                std::cerr << a << " takes on|off\n";
                std::exit(1);
            }
            (a == "--stride" ? STRIDE_SAMPLING : MEASURE_CE) = on;
        } else if (a == "--recall-target")
            RECALL_TARGET = std::stod(need(i));
        else if (a == "--ef-values")
            EF_VALUES = split_sz(need(i));
        else if (a == "--bridge-budget-ratios")
            BBR_VALUES = split_d(need(i));
        else if (a == "--bucket-edges")
            BUCKET_EDGES = split_d(need(i));
        else if (a == "--modes") {
            MODES.clear();
            std::stringstream ss(need(i));
            std::string t;
            while (std::getline(ss, t, ','))
                if (!t.empty())
                    MODES.push_back(t);
        } else if (a == "--data-dir")
            DATA_DIR = need(i);
        else if (a == "--volume-dir")
            VOLUME_DIR = need(i);
        else if (a == "--gt-dir")
            GT_DIR = need(i);
        else if (a == "--json")
            JSON_PATH = need(i);
        else if (a == "--build-index")
            BUILD_INDEX_ONLY = true;
        else if (a == "--help" || a == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << a << " (see --help)\n";
            std::exit(1);
        }
    }
}
