// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 * @file bench_synthfilter.cpp
 * @brief Synthetic per-query filters at controlled selectivity, over fvecs (sift, gist,
 *        laion), fbin (text2image) and u8bin (yfcc) datasets.
 *
 * Predicate kinds:
 *   - --filter allowlist (default): a per-query random passing set of size round(N*s) for
 *     each selectivity s. --alpha draws a fraction of the set from the query's nearest
 *     (alpha > 0) or farthest (alpha < 0) base vectors; the rest is uniform
 *   - --filter conj: a conjunction of uniform categorical attributes
 *   - --filter column: a per-query range predicate over a value column
 *   - --allowlist-file / --allowlist-spec: externally supplied per-query passing sets
 * Ground truth is exact filtered search at --k. cmpkey OFF.
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
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

using namespace unum::usearch;

// ── fvecs loader ────────────────────────────────────────────────────────────
static bool load_fvecs(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims,
                       std::size_t max_count = 0) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    int32_t d = 0;
    if (!ifs.read(reinterpret_cast<char*>(&d), 4) || d <= 0 || d > 100000)
        return false;
    dims = d;
    ifs.seekg(0, std::ios::end);
    auto fsz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::size_t rec = 4 + dims * 4;
    count = (std::size_t)fsz / rec;
    if (max_count > 0 && count > max_count)
        count = max_count;
    data.resize(count * dims);
    for (std::size_t i = 0; i < count; ++i) {
        int32_t dc = 0;
        if (!ifs.read(reinterpret_cast<char*>(&dc), 4) || (std::size_t)dc != dims)
            return false;
        if (!ifs.read(reinterpret_cast<char*>(data.data() + i * dims), dims * 4))
            return false;
    }
    return true;
}

// ── fbin loader (float): [int32 n][int32 d][float n*d] ──────────────────────
static bool load_fbin(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims,
                      std::size_t max_count = 0) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    int32_t n = 0, d = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), 4) || !ifs.read(reinterpret_cast<char*>(&d), 4))
        return false;
    if (n <= 0 || d <= 0 || d > 100000)
        return false;
    count = (std::size_t)n;
    dims = (std::size_t)d;
    if (max_count > 0 && count > max_count)
        count = max_count;
    data.resize(count * dims);
    return (bool)ifs.read(reinterpret_cast<char*>(data.data()), (std::streamsize)(count * dims) * 4);
}

// ── u8bin loader (uint8 -> float): [int32 n][int32 d][uint8 n*d] ────────────
static bool load_u8bin(char const* path, std::vector<float>& data, std::size_t& count, std::size_t& dims,
                       std::size_t max_count = 0) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;
    int32_t n = 0, d = 0;
    if (!ifs.read(reinterpret_cast<char*>(&n), 4) || !ifs.read(reinterpret_cast<char*>(&d), 4))
        return false;
    if (n <= 0 || d <= 0 || d > 100000)
        return false;
    count = (std::size_t)n;
    dims = (std::size_t)d;
    if (max_count > 0 && count > max_count)
        count = max_count;
    std::vector<uint8_t> raw(count * dims);
    if (!ifs.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)raw.size()))
        return false;
    data.resize(count * dims);
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = (float)raw[i];
    return true;
}

// L2-normalize each row in place. For cosine datasets, the L2^2 ranking on unit
// vectors is order-equivalent to cosine, so the l2 kernel and a cosine-built
// volume stay consistent.
static void normalize_rows(std::vector<float>& data, std::size_t n, std::size_t d) {
    for (std::size_t i = 0; i < n; ++i) {
        float* v = data.data() + i * d;
        double s = 0;
        for (std::size_t j = 0; j < d; ++j)
            s += (double)v[j] * v[j];
        if (s > 0) {
            float inv = (float)(1.0 / std::sqrt(s));
            for (std::size_t j = 0; j < d; ++j)
                v[j] *= inv;
        }
    }
}

// ── infra (mirrors bench_realfilter) ────────────────────────────────────────
template <typename T = float> class vector_store_t {
  public:
    vector_store_t(std::vector<T>&& d, std::size_t dims) : dims_(dims), data_(std::move(d)) {}
    std::size_t size() const noexcept { return data_.size() / dims_; }
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
    return (*std::max_element(v.begin(), v.begin() + n / 2) + v[n / 2]) / 2.0;
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

// ── dense-key bitset for O(1) allowlist membership ───────────────────────────
// Base keys are dense integers in [0, N), so a directly indexed bitset is the
// optimal predicate store. It is true O(1) (no hashing or probing), has the best
// cache behavior, and costs 1 bit per key. The filter-first methods (ACORN-1/RACORN-1)
// evaluate the predicate many times per distance computation, so this constant
// factor matters. A hash map like absl::flat_hash_map is only preferable when keys
// are sparse or non-integer.
struct dense_bitset_t {
    std::vector<std::uint64_t> bits;
    void build(std::size_t n, std::vector<vec_key_t> const& keys) {
        bits.assign((n + 63) / 64, 0ULL);
        for (vec_key_t k : keys)
            bits[k >> 6] |= (1ULL << (k & 63));
    }
    inline bool contains(vec_key_t k) const noexcept { return (bits[k >> 6] >> (k & 63)) & 1ULL; }
};

// Multi-attribute conjunction filter. attr[j][key] is a uint8 category (dense key index, O(1)).
// inv[j][v] is the sorted key list for attribute j == v. It is a dedicated per-attribute index
// that only covers the marginal, so a conjunction must intersect k lists.
struct conj_t {
    std::size_t n_attrs = 0, card = 0, k = 0;
    std::vector<std::vector<std::uint8_t>> attr;               // [j][key]
    std::vector<std::vector<std::vector<std::uint32_t>>> inv;  // [j][v] sorted keys
    std::vector<std::vector<std::uint8_t>> qv;                 // [q][j] query values
    static inline std::uint64_t mix(std::uint64_t x) noexcept {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    void build(std::size_t n, std::size_t attrs, std::size_t c, std::uint64_t seed) {
        n_attrs = attrs, card = c;
        attr.assign(attrs, std::vector<std::uint8_t>(n));
        inv.assign(attrs, std::vector<std::vector<std::uint32_t>>(c));
        for (std::size_t j = 0; j < attrs; ++j) {
            for (std::size_t key = 0; key < n; ++key) {
                std::uint8_t v = (std::uint8_t)(mix(seed ^ (0x51ed270b3ULL * (j + 1)) ^ (key * 0x2545f4914f6cdd1dULL)) % c);
                attr[j][key] = v;
                inv[j][v].push_back((std::uint32_t)key);
            }
        }
    }
    void set_queries(std::size_t nq, std::uint64_t seed) {
        qv.assign(nq, std::vector<std::uint8_t>(n_attrs));
        for (std::size_t q = 0; q < nq; ++q)
            for (std::size_t j = 0; j < n_attrs; ++j)
                qv[q][j] = (std::uint8_t)(mix(seed ^ 0xabcdef ^ (0x9e3779b1ULL * (q + 1)) ^ (j + 1)) % card);
    }
    inline bool pass(std::size_t q, vec_key_t key) const noexcept {
        for (std::size_t j = 0; j < k; ++j)
            if (attr[j][key] != qv[q][j])
                return false;
        return true;
    }
    // prefilter-scan: one pass over the attribute arrays (no index on any attribute)
    std::vector<vec_key_t> passing_scan(std::size_t q) const {
        std::vector<vec_key_t> out;
        std::size_t n = attr.empty() ? 0 : attr[0].size();
        for (std::size_t key = 0; key < n; ++key)
            if (pass(q, key))
                out.push_back((vec_key_t)key);
        return out;
    }
    // prefilter-index: intersect the k sorted per-attribute lists (each ~N/card long)
    std::vector<vec_key_t> passing_index(std::size_t q, std::size_t& scanned) const {
        std::vector<std::uint32_t> cur = inv[0][qv[q][0]];
        scanned = cur.size();
        for (std::size_t j = 1; j < k; ++j) {
            auto const& other = inv[j][qv[q][j]];
            scanned += other.size();
            std::vector<std::uint32_t> next;
            next.reserve(std::min(cur.size(), other.size()));
            std::set_intersection(cur.begin(), cur.end(), other.begin(), other.end(), std::back_inserter(next));
            cur.swap(next);
        }
        return std::vector<vec_key_t>(cur.begin(), cur.end());
    }
};

// ── natural attribute column (--filter column, Section VI-C) ────────────────
// One u32 attribute per key (dense keys), plus per-query [lo, hi] ranges grouped into named
// groups. A categorical predicate is the range lo == hi over a code column. A numeric range
// is a window over the value column. This is the shape of the attribute filters the datasets
// ship with (LAION NSFW flag, licence, original resolution): one array read and two compares.
// That is the same cost class as the controlled uint32 attribute of Section VI-A. The
// generator (laion_column_workload.py) writes both files.
//
// SFCL: "SFCL", u32 version=1, u64 N, u32 reserved, then N x u32 values.
// SFCQ: "SFCQ", u32 version=1, u32 ngroups, u32 nq, then per group: u32 label_len, label bytes,
//       f64 target_selectivity (NaN = natural, data decides), u8 kind (0 = categorical equality,
//       1 = numeric range), then nq x {u32 lo, u32 hi}. Query order is the dataset's query order.
struct column_t {
    struct group_t {
        std::string label;
        double target = std::nan("");
        std::uint8_t kind = 0;
        std::vector<std::uint32_t> lo, hi; // [q]
    };
    std::string col_path, q_path;
    std::vector<std::uint32_t> val; // [key]
    std::vector<group_t> groups;
    // Index variant, built on demand: the keys ordered by value, with the parallel sorted
    // values for binary search. For a code column this is the per-value posting list. For a
    // numeric column it is a sorted attribute index, and a range query reads one contiguous slice.
    std::vector<std::uint32_t> idx_vals, idx_keys;

    // The value array alone. It is also the range half of a mixed predicate (--predicate-column,
    // Section VI-D). That predicate takes its per-query bounds from an SFRQ file instead of an
    // SFCQ group.
    bool load_values(std::size_t n_expected) {
        std::ifstream f(col_path, std::ios::binary);
        if (!f) {
            std::cerr << "cannot open column file " << col_path << "\n";
            return false;
        }
        char magic[4] = {};
        uint32_t ver = 0, reserved = 0;
        uint64_t n = 0;
        f.read(magic, 4);
        f.read(reinterpret_cast<char*>(&ver), 4);
        f.read(reinterpret_cast<char*>(&n), 8);
        f.read(reinterpret_cast<char*>(&reserved), 4);
        if (std::string(magic, 4) != "SFCL" || ver != 1 || n != n_expected) {
            std::cerr << "column file header mismatch (magic/version/N=" << n << " vs " << n_expected << ")\n";
            return false;
        }
        val.resize((std::size_t)n);
        if (!f.read(reinterpret_cast<char*>(val.data()), (std::streamsize)n * 4)) {
            std::cerr << "column file truncated: " << col_path << "\n";
            return false;
        }
        return true;
    }
    // Keys whose value lies in [lo, hi], read from the sorted index (build_index first).
    std::pair<std::size_t, std::size_t> index_slice(std::uint32_t lo, std::uint32_t hi) const {
        auto a = std::lower_bound(idx_vals.begin(), idx_vals.end(), lo) - idx_vals.begin();
        auto b = std::upper_bound(idx_vals.begin(), idx_vals.end(), hi) - idx_vals.begin();
        return {(std::size_t)a, (std::size_t)b};
    }
    bool load(std::size_t n_expected, std::size_t nq_expected) {
        if (!load_values(n_expected))
            return false;
        std::ifstream f(q_path, std::ios::binary);
        if (!f) {
            std::cerr << "cannot open column query file " << q_path << "\n";
            return false;
        }
        char magic[4] = {};
        uint32_t ver = 0, ng = 0, nq = 0;
        f.read(magic, 4);
        f.read(reinterpret_cast<char*>(&ver), 4);
        f.read(reinterpret_cast<char*>(&ng), 4);
        f.read(reinterpret_cast<char*>(&nq), 4);
        if (std::string(magic, 4) != "SFCQ" || ver != 1 || nq < nq_expected) {
            std::cerr << "column query file header mismatch (magic/version/nq=" << nq << " < " << nq_expected << ")\n";
            return false;
        }
        groups.assign(ng, {});
        std::vector<std::uint32_t> buf(2 * (std::size_t)nq);
        for (uint32_t g = 0; g < ng; ++g) {
            uint32_t len = 0;
            f.read(reinterpret_cast<char*>(&len), 4);
            std::string label(len, '\0');
            f.read(&label[0], len);
            double target = 0;
            std::uint8_t kind = 0;
            f.read(reinterpret_cast<char*>(&target), 8);
            f.read(reinterpret_cast<char*>(&kind), 1);
            f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)buf.size() * 4);
            groups[g].label = label;
            groups[g].target = target;
            groups[g].kind = kind;
            groups[g].lo.resize(nq_expected);
            groups[g].hi.resize(nq_expected);
            for (std::size_t q = 0; q < nq_expected; ++q) {
                groups[g].lo[q] = buf[2 * q];
                groups[g].hi[q] = buf[2 * q + 1];
            }
        }
        return f.good();
    }
    void build_index() {
        if (!idx_keys.empty())
            return;
        idx_keys.resize(val.size());
        for (std::size_t i = 0; i < val.size(); ++i)
            idx_keys[i] = (std::uint32_t)i;
        std::sort(idx_keys.begin(), idx_keys.end(), [&](std::uint32_t a, std::uint32_t b) {
            return val[a] < val[b] || (val[a] == val[b] && a < b);
        });
        idx_vals.resize(val.size());
        for (std::size_t i = 0; i < val.size(); ++i)
            idx_vals[i] = val[idx_keys[i]];
    }
    inline bool pass(std::size_t g, std::size_t q, vec_key_t key) const noexcept {
        std::uint32_t v = val[key];
        return v >= groups[g].lo[q] && v <= groups[g].hi[q];
    }
    // prefilter-scan: test the predicate on every key (no index on the attribute)
    std::vector<vec_key_t> passing_scan(std::size_t g, std::size_t q) const {
        std::vector<vec_key_t> out;
        std::uint32_t lo = groups[g].lo[q], hi = groups[g].hi[q];
        for (std::size_t key = 0; key < val.size(); ++key) {
            std::uint32_t v = val[key];
            if (v >= lo && v <= hi)
                out.push_back((vec_key_t)key);
        }
        return out;
    }
    // prefilter-index: binary-search the sorted attribute and read the slice of keys in range
    std::vector<vec_key_t> passing_index(std::size_t g, std::size_t q, std::size_t& scanned) const {
        auto a = std::lower_bound(idx_vals.begin(), idx_vals.end(), groups[g].lo[q]) - idx_vals.begin();
        auto b = std::upper_bound(idx_vals.begin(), idx_vals.end(), groups[g].hi[q]) - idx_vals.begin();
        scanned = (std::size_t)(b - a);
        return std::vector<vec_key_t>(idx_keys.begin() + a, idx_keys.begin() + b);
    }
};

// Per-query predicate: O(1) bit lookup for allowlists, the k-attribute conjunction, or one
// array read and two compares on a natural attribute column.
struct allowlist_predicate_t {
    dense_bitset_t const* hset = nullptr;
    conj_t const* conj = nullptr;
    std::uint32_t const* col = nullptr; // column mode
    std::uint32_t lo = 0, hi = 0;
    std::size_t q = 0;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        vec_key_t key = (vec_key_t)get_key(m);
        if (col) {
            std::uint32_t v = col[key];
            return v >= lo && v <= hi;
        }
        return conj ? conj->pass(q, key) : hset->contains(key);
    }
};

// One predicate type per mode, for the searches. The search loop is a template on the predicate.
// Handing it `allowlist_predicate_t` would compile the mode branches above into every
// evaluation, and the filter-first methods evaluate the predicate many times per distance.
// Each type below passes exactly the keys `allowlist_predicate_t` passes, so recall and every
// counter are unchanged.
struct bitset_predicate_t {
    dense_bitset_t const* hset;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        return hset->contains((vec_key_t)get_key(m));
    }
};
struct conj_predicate_t {
    conj_t const* conj;
    std::size_t q;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        return conj->pass(q, (vec_key_t)get_key(m));
    }
};
struct column_predicate_t {
    std::uint32_t const* col;
    std::uint32_t lo, hi;
    template <typename member_like_at> bool operator()(member_like_at&& m) const noexcept {
        std::uint32_t v = col[(vec_key_t)get_key(m)];
        return v >= lo && v <= hi;
    }
};
// Calls `search(pred)` with the predicate type of `p`'s mode; the branch runs once per search.
template <typename search_at> static auto search_by_mode(allowlist_predicate_t const& p, search_at&& search) {
    if (p.col)
        return search(column_predicate_t{p.col, p.lo, p.hi});
    if (p.conj)
        return search(conj_predicate_t{p.conj, p.q});
    return search(bitset_predicate_t{p.hset});
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

// SFAL: "SFAL", u32 version, u32 nq, u64 N, then per entry u32 qidx, u32 cnt, cnt x u32 keys.
static bool load_allowlist_file(std::string const& path, std::size_t n_expected, std::vector<std::size_t>& qidx,
                                std::vector<std::vector<vec_key_t>>& sets) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    char magic[4] = {};
    uint32_t ver = 0, nq = 0;
    uint64_t n = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&nq), 4);
    f.read(reinterpret_cast<char*>(&n), 8);
    if (std::string(magic, 4) != "SFAL" || ver != 1 || n != n_expected) {
        std::cerr << "allowlist file header mismatch (magic/version/N=" << n << " vs " << n_expected << ")\n";
        return false;
    }
    qidx.resize(nq);
    sets.assign(nq, {});
    std::vector<uint32_t> buf;
    for (uint32_t i = 0; i < nq; ++i) {
        uint32_t q = 0, cnt = 0;
        f.read(reinterpret_cast<char*>(&q), 4);
        f.read(reinterpret_cast<char*>(&cnt), 4);
        buf.resize(cnt);
        f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)cnt * 4);
        for (uint32_t k : buf)
            if (k >= n_expected) {
                std::cerr << "allowlist file: query " << q << " has key " << k << " >= N=" << n_expected
                          << " (keys must be dense 0..N-1)\n";
                return false;
            }
        qidx[i] = q;
        sets[i].assign(buf.begin(), buf.end());
        std::sort(sets[i].begin(), sets[i].end());
    }
    return f.good();
}

// ── caption predicate structures (prefilter-scan / prefilter-index) ─────────
// An external allowlist says *which* keys pass, not *why*. To charge a scan for
// evaluating the real predicate, we need the per-row token sets the predicate was
// derived from. To charge an index, we need the per-token posting lists. The
// generator writes both next to the SFAL.

// Row -> ascending token ids, big-ann .spmat layout without the all-ones data[].
struct token_csr_t {
    int64_t nrow = 0, ncol = 0, nnz = 0;
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    int32_t const* row_begin(std::size_t i) const { return indices.data() + indptr[i]; }
    int32_t const* row_end(std::size_t i) const { return indices.data() + indptr[i + 1]; }
    bool has(std::size_t row, int32_t t) const { return std::binary_search(row_begin(row), row_end(row), t); }
};

static bool load_token_csr(std::string const& path, std::size_t n_expected, token_csr_t& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    int64_t sizes[3];
    if (!f.read(reinterpret_cast<char*>(sizes), sizeof(sizes)))
        return false;
    m.nrow = sizes[0], m.ncol = sizes[1], m.nnz = sizes[2];
    if (m.nrow != (int64_t)n_expected) {
        std::cerr << "token csr has " << m.nrow << " rows but N=" << n_expected << "\n";
        return false;
    }
    if (m.ncol < 0 || m.nnz < 0)
        return false;
    m.indptr.resize(m.nrow + 1);
    if (!f.read(reinterpret_cast<char*>(m.indptr.data()), (m.nrow + 1) * (int64_t)sizeof(int64_t)))
        return false;
    m.indices.resize(m.nnz);
    if (m.nnz > 0 && !f.read(reinterpret_cast<char*>(m.indices.data()), m.nnz * (int64_t)sizeof(int32_t)))
        return false;
    for (int64_t i = 0; i < m.nnz; ++i)
        if (m.indices[i] < 0 || m.indices[i] >= m.ncol) {
            std::cerr << "token csr: id " << m.indices[i] << " out of range 0.." << m.ncol - 1 << "\n";
            return false;
        }
    return true;
}

// Transpose: token -> ascending rows. Costs as much memory as the CSR, so built on demand.
struct token_postings_t {
    std::vector<int64_t> start;
    std::vector<std::uint32_t> rows;
    std::size_t ntokens = 0;
    void build(token_csr_t const& m) {
        ntokens = (std::size_t)(m.ncol < 0 ? 0 : m.ncol);
        start.assign(ntokens + 1, 0);
        for (int64_t i = 0; i < m.nnz; ++i)
            ++start[(std::size_t)m.indices[i] + 1];
        for (std::size_t t = 0; t < ntokens; ++t)
            start[t + 1] += start[t];
        rows.resize((std::size_t)m.nnz);
        std::vector<int64_t> cur(start.begin(), start.end() - 1);
        for (int64_t r = 0; r < m.nrow; ++r)
            for (int64_t j = m.indptr[r]; j < m.indptr[r + 1]; ++j)
                rows[(std::size_t)cur[(std::size_t)m.indices[j]]++] = (std::uint32_t)r;
    }
    std::uint32_t const* list_begin(int32_t t) const { return rows.data() + start[(std::size_t)t]; }
    std::size_t list_len(int32_t t) const { return (std::size_t)(start[(std::size_t)t + 1] - start[(std::size_t)t]); }
    bool holds(int32_t t) const { return t >= 0 && (std::size_t)t < ntokens; }
};

// SFPR: "SFPR", u32 version, u32 nq, u32 ntokens, then per entry u32 qidx, u8 op(0=AND,1=OR),
// u32 cnt, cnt x u32 token_id. Entry order matches the SFAL written beside it.
struct pred_terms_t {
    std::vector<std::size_t> qidx;
    std::vector<std::uint8_t> op;
    std::vector<std::vector<int32_t>> terms;
};

static bool load_pred_terms(std::string const& path, std::size_t ntokens, pred_terms_t& t) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    char magic[4] = {};
    uint32_t ver = 0, nq = 0, nt = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&nq), 4);
    f.read(reinterpret_cast<char*>(&nt), 4);
    if (std::string(magic, 4) != "SFPR" || ver != 1 || nt != ntokens) {
        std::cerr << "predicate terms header mismatch (magic/version/ntokens=" << nt << " vs " << ntokens << ")\n";
        return false;
    }
    t.qidx.resize(nq);
    t.op.resize(nq);
    t.terms.assign(nq, {});
    for (uint32_t i = 0; i < nq; ++i) {
        uint32_t q = 0, cnt = 0;
        std::uint8_t op = 0;
        f.read(reinterpret_cast<char*>(&q), 4);
        f.read(reinterpret_cast<char*>(&op), 1);
        f.read(reinterpret_cast<char*>(&cnt), 4);
        t.terms[i].resize(cnt);
        if (cnt)
            f.read(reinterpret_cast<char*>(t.terms[i].data()), (std::streamsize)cnt * 4);
        for (int32_t id : t.terms[i])
            if (id < 0 || (std::size_t)id >= ntokens) {
                std::cerr << "predicate terms: token id " << id << " out of range\n";
                return false;
            }
        t.qidx[i] = q;
        t.op[i] = op;
    }
    return f.good();
}

// ── several external allowlists in one process (--allowlist-spec, Section VI-D)
// Each spec is one group: <sfal>|<label>[|<sfpr>[|<sfrq>]]. The SFAL holds the passing sets used
// by the graph arms, the given prefilter and the ground truth (one bit per key, as in single-file
// mode). The optional SFPR holds the query's token terms over the shared --predicate-csr, so the
// scan and index prefilter arms evaluate the real predicate. The optional SFRQ adds a range over
// the shared --predicate-column. The predicate then becomes "tokens AND lo <= column <= hi":
// a text condition and a numeric range at once. No single-type dedicated index serves such a
// predicate.
//
// SFRQ: "SFRQ", u32 version=1, u32 nq, then per entry u32 qidx, u32 lo, u32 hi (SFAL entry order).
struct ext_group_t {
    std::string sfal, label, sfpr, sfrq;
    std::vector<std::size_t> qidx;
    std::vector<std::vector<vec_key_t>> sets;
    pred_terms_t terms;                // empty without an SFPR
    std::vector<std::uint32_t> lo, hi; // empty without an SFRQ
};

static bool load_ranges(std::string const& path, std::vector<std::size_t> const& qidx, std::vector<std::uint32_t>& lo,
                        std::vector<std::uint32_t>& hi) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    char magic[4] = {};
    uint32_t ver = 0, nq = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&ver), 4);
    f.read(reinterpret_cast<char*>(&nq), 4);
    if (std::string(magic, 4) != "SFRQ" || ver != 1 || nq != qidx.size()) {
        std::cerr << "range file header mismatch (magic/version/nq=" << nq << " vs " << qidx.size() << ")\n";
        return false;
    }
    lo.resize(nq);
    hi.resize(nq);
    for (uint32_t i = 0; i < nq; ++i) {
        uint32_t rec[3] = {};
        f.read(reinterpret_cast<char*>(rec), sizeof(rec));
        if (rec[0] != qidx[i]) {
            std::cerr << "range file query order differs from the allowlist at entry " << i << "\n";
            return false;
        }
        lo[i] = rec[1];
        hi[i] = rec[2];
    }
    return f.good();
}

// ── config ──────────────────────────────────────────────────────────────────
static std::string DATASET = "sift";       // sift | gist | text2image | yfcc | laion
static std::size_t N_VAL = 100'000'000, D_VAL = 128, M_VAL = 16, K_VAL = 100, EF_CONSTRUCTION_VAL = 100;
static std::size_t NUM_QUERIES = 1000, SEARCH_THREADS = 32, BUILD_THREADS = 0, BENCH_RUNS = 3, BENCH_WARMUP = 2;
static double ACORN_THRESHOLD = 0.3, RECALL_TARGET = 0.97;
// --alpha accepts a list. Every (selectivity, alpha) pair becomes one group of the same
// process, so a whole correlation sweep shares one cache state (the protocol of Section VI-B).
static std::vector<double> ALPHAS = {0.0};
static double THREE_HOP_TAU = 1.0;    // --three-hop-threshold: acorn-1-3hop trigger, in units of M0
static bool STRIDE_SAMPLING = true;   // --stride on|off
static bool MEASURE_CE = true;        // --ce on|off: normalized mean rank of the passing set
static std::size_t NQ_USED = 0;

// Per-group summary written to the JSON: the passing-set size and the measured correlation
// c_e. A reader can then tell what each row's predicate looked like without re-deriving it.
struct group_stat_t {
    double s, alpha;
    std::size_t m;
    bool has_ce;
    double ce_mean, ce_p10, ce_p90;
    std::string label;                // the group tag the rows' `label` field starts with
    double target = std::nan("");     // column mode: the generator's target selectivity (NaN = natural)
    std::size_t nq = 0;
};
static std::vector<group_stat_t> GROUP_STATS;
static std::uint64_t SEED = 12345;
static std::vector<std::size_t> EF_VALUES = {100, 200, 300, 400, 500, 600, 700, 800,
                                             900, 1000, 1100, 1200, 1300, 1400, 1500, 1600};
static std::vector<double> BBR_VALUES = {1.0, 2.0};
static std::vector<std::string> MODES = {"prefilter", "hnsw", "acorn-1", "racorn-1"};
static std::vector<double> SELECTIVITIES = {0.05, 0.03, 0.01, 0.005, 0.001};
static std::string DATA_DIR = "data/raw", VOLUME_DIR = "data/volumes", GT_DIR = "data/gt", JSON_PATH = "";
// External per-query passing sets (e.g. caption-keyword matches from laion_captions.py).
// When set, the selectivity loop is replaced by this single group labeled ALLOWLIST_LABEL.
static std::string ALLOWLIST_FILE = "", ALLOWLIST_LABEL = "ext";
static std::string PRED_CSR_FILE = "", PRED_TERMS_FILE = "";
// --allowlist-spec (repeatable): several external groups in one process, see ext_group_t.
// --predicate-column: the SFCL a group's SFRQ ranges refer to.
static std::vector<std::string> ALLOWLIST_SPECS;
static std::string PRED_COLUMN_FILE = "";
// --filter conj: multi-attribute conjunction over CONJ_ATTRS uniform categorical attributes
// (uint8 flat arrays indexed by key, cardinality CONJ_CARD). Group k = conjunction of the
// first k attributes, selectivity (1/card)^k. Prefilter variants: given passing set (optimistic),
// attribute scan, and intersection of per-attribute inverted lists (dedicated index per marginal).
static std::string FILTER = "allowlist";
static std::size_t CONJ_ATTRS = 6, CONJ_CARD = 3;
static std::vector<std::size_t> CONJ_KS = {1, 2, 3, 4, 5, 6};
// --filter column: natural attribute columns (Section VI-C). Each --column-spec <col>:<queries> pair
// contributes its groups to the same process; see column_t for the file formats.
static std::vector<std::string> COLUMN_SPECS;
static double BUILD_TIME_MS = -1.0;
// Base-level average out-degree (index.stats(0): edges/nodes). The nominal cap
// is 2*M. RNG pruning leaves most nodes below it, so the cost model needs
// the measured value. Emitted as config.mavg.
static double MAVG_BASE = 0.0;

static bool iequals(std::string const& a, std::string const& b) {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return true;
}
static int navix_of(std::string const& m) {
    if (iequals(m, "navix-onehop-s"))
        return 1;
    if (iequals(m, "navix-blind"))
        return 2;
    if (iequals(m, "navix-directed"))
        return 3;
    // adaptive-local: navix-local is the Kuzu artifact's form (8), as in bench_acorn;
    // navix-local-paper is the paper text's form (5).
    if (iequals(m, "navix-local"))
        return 8;
    if (iequals(m, "navix-local-paper"))
        return 5;
    return 0;
}
static std::string vol_base() { // matches bench_acorn volume naming so volumes are shared
    std::ostringstream o;
    o << VOLUME_DIR << "/" << DATASET << "_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef" << EF_CONSTRUCTION_VAL
      << ".usearch";
    return o.str();
}
static std::string sel_tag(double s) {
    std::ostringstream o;
    o << std::defaultfloat << std::setprecision(4) << (s * 100) << "%";
    return o.str();
}

struct row_t {
    std::string label, mode;
    double selectivity;
    double alpha; // NaN for conj / external groups
    std::size_t ef_search;
    double bridge_budget_ratio;
    double latency_ms, latency_p50, latency_p90, latency_p99;
    double recall, recall_p50, recall_p90, recall_p99;
    double dist_computed, nodes_visited, predicate_evals, qps;
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
static void write_json(std::string const& path, std::vector<row_t> const& rows) {
    std::ofstream o(path);
    if (!o) {
        std::cerr << "cannot open json\n";
        return;
    }
    auto num = [](double v) {
        if (std::isnan(v))
            return std::string("null");
        std::ostringstream s;
        s << std::setprecision(6) << v;
        return s.str();
    };
    o << "{\n  \"config\": {\n    \"dataset\": \"" << DATASET << "\", \"n\": " << N_VAL << ", \"d\": " << D_VAL
      << ", \"m\": " << M_VAL << ", \"ef_construction\": " << EF_CONSTRUCTION_VAL << ",\n    \"k\": " << K_VAL
      << ", \"metric\": \"l2\", \"filter\": \""
      << (FILTER == "conj"     ? "conj:attrs=" + std::to_string(CONJ_ATTRS) + ",card=" + std::to_string(CONJ_CARD)
          : FILTER == "column" ? "column:" + std::to_string(COLUMN_SPECS.size()) + " specs"
          : !ALLOWLIST_SPECS.empty() ? "external:" + std::to_string(ALLOWLIST_SPECS.size()) + " specs"
          : ALLOWLIST_FILE.empty() ? std::string("allowlist")
                                   : "external:" + std::filesystem::path(ALLOWLIST_FILE).filename().string())
      << "\", \"alpha\": " << (ALPHAS.size() == 1 ? num(ALPHAS[0]) : std::string("null")) << ", \"alphas\": [";
    for (std::size_t i = 0; i < ALPHAS.size(); ++i)
        o << (i ? ", " : "") << num(ALPHAS[i]);
    o << "], \"column_specs\": [";
    for (std::size_t i = 0; i < COLUMN_SPECS.size(); ++i)
        o << (i ? ", " : "") << "\"" << COLUMN_SPECS[i] << "\"";
    o << "], \"allowlist_specs\": [";
    for (std::size_t i = 0; i < ALLOWLIST_SPECS.size(); ++i)
        o << (i ? ", " : "") << "\"" << ALLOWLIST_SPECS[i] << "\"";
    o << "], \"predicate_csr\": \"" << PRED_CSR_FILE << "\", \"predicate_column\": \"" << PRED_COLUMN_FILE << "\"";
    o << ", \"seed\": " << SEED << ", \"filter_first_threshold\": " << num(ACORN_THRESHOLD)
      << ", \"three_hop_threshold\": " << num(THREE_HOP_TAU)
      << ", \"stride_sampling\": " << (STRIDE_SAMPLING ? "true" : "false") << ", \"nq\": " << NQ_USED
      << ", \"cmpkey\": false, \"search_threads\": " << SEARCH_THREADS << ", \"build_time_ms\": " << num(BUILD_TIME_MS)
      << ", \"mavg\": " << num(MAVG_BASE) << ",\n    \"groups\": [";
    for (std::size_t i = 0; i < GROUP_STATS.size(); ++i) {
        auto const& g = GROUP_STATS[i];
        o << (i ? ",\n      " : "\n      ") << "{\"label\": \"" << g.label << "\", \"nq\": " << g.nq
          << ", \"selectivity\": " << num(g.s) << ", \"target\": " << num(g.target) << ", \"alpha\": " << num(g.alpha)
          << ", \"m\": " << g.m << ", \"ce_mean\": " << (g.has_ce ? num(g.ce_mean) : std::string("null"))
          << ", \"ce_p10\": " << (g.has_ce ? num(g.ce_p10) : std::string("null"))
          << ", \"ce_p90\": " << (g.has_ce ? num(g.ce_p90) : std::string("null")) << "}";
    }
    o << (GROUP_STATS.empty() ? "" : "\n    ") << "]"
      << "\n  },\n  \"results\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        auto const& r = rows[i];
        o << "    {\"label\": \"" << r.label << "\", \"mode\": \"" << r.mode
          << "\", \"selectivity\": " << num(r.selectivity) << ", \"alpha\": " << num(r.alpha) << ", \"ef_search\": " << r.ef_search
          << ", \"bridge_budget_ratio\": " << num(r.bridge_budget_ratio) << ", \"exact\": false, \"threads\": "
          << r.threads << ", \"num_queries\": " << r.num_queries << ", \"latency_ms\": " << num(r.latency_ms)
          << ", \"latency_p50\": " << num(r.latency_p50) << ", \"latency_p90\": " << num(r.latency_p90)
          << ", \"latency_p99\": " << num(r.latency_p99) << ", \"recall\": " << num(r.recall)
          << ", \"recall_p50\": " << num(r.recall_p50) << ", \"recall_p90\": " << num(r.recall_p90)
          << ", \"recall_p99\": " << num(r.recall_p99) << ", \"dist_computed\": " << num(r.dist_computed)
          << ", \"nodes_visited\": " << num(r.nodes_visited) << ", \"predicate_evals\": " << num(r.predicate_evals)
          << ", \"qps\": " << num(r.qps) << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    o << "  ]\n}\n";
    std::cout << "Wrote JSON: " << path << " (" << rows.size() << " rows)\n";
}

static void parse_args(int argc, char** argv);

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    parse_args(argc, argv);
    // ALPHA in [-1,1]: fraction of each query's passing set drawn from its
    // nearest (alpha>0) or farthest (alpha<0) base vectors; the rest uniform.
    // Building a set takes a full O(N) distance pass per query when alpha != 0
    // or when c_e is measured (--ce on, the default).

    // Base and query vectors. text2image is fbin + cosine: we normalize and use the
    // l2 kernel. On unit vectors the l2 kernel is order-equivalent to cosine, and it
    // matches the cosine-built volume. laion is fvecs, normalized and searched the
    // same way. sift/gist are fvecs + l2; yfcc is u8bin + l2.
    bool const is_t2i = (DATASET == "text2image");
    bool const is_yfcc = (DATASET == "yfcc"); // u8bin vectors, l2 (synthetic filter, not tags)
    bool const need_norm = is_t2i || (DATASET == "laion"); // cosine: normalize + l2 kernel
    std::string base_file = DATA_DIR + "/" +
                            (is_t2i ? "text2image-40M.fbin" : is_yfcc ? "base.10M.u8bin" : DATASET + "_base.fvecs");
    std::string query_file =
        DATA_DIR + "/" +
        (is_t2i ? "text2image-query-100K.fbin" : is_yfcc ? "query.public.100K.u8bin" : DATASET + "_query.fvecs");
    std::vector<float> base;
    std::size_t bn = 0, bd = 0;
    {
        stopwatch_t t;
        bool ok = is_t2i    ? load_fbin(base_file.c_str(), base, bn, bd, N_VAL)
                  : is_yfcc ? load_u8bin(base_file.c_str(), base, bn, bd, N_VAL)
                            : load_fvecs(base_file.c_str(), base, bn, bd, N_VAL);
        if (!ok) {
            std::cerr << "failed base " << base_file << "\n";
            return 1;
        }
        if (need_norm)
            normalize_rows(base, bn, bd);
        std::cout << "base: " << bn << " x " << bd << "d (" << std::fixed << std::setprecision(1) << t.ms() << " ms)\n";
    }
    if (bd != D_VAL) {
        std::cerr << "dim mismatch " << bd << "!=" << D_VAL << "\n";
        return 1;
    }
    if (bn != N_VAL) {
        // Every predicate structure (bitset, attribute arrays) and the alpha ranking index
        // vectors by key in [0, N). A short base file means out-of-bounds reads, not an error.
        std::cerr << "base has " << bn << " rows but --n " << N_VAL << " was requested; refusing (" << base_file
                  << " may be a different dataset than the volume/GT were built from)\n";
        return 1;
    }
    auto store = std::make_unique<vector_store_t<float>>(std::move(base), D_VAL);

    index_config_t cfg(M_VAL);
    aligned_index_t<index_t> aidx(cfg);
    auto& index = *aidx.ptr;
    l2_metric_t metric{store.get(), D_VAL};
    std::string vpath = vol_base();
    std::ifstream probe(vpath);
    if (!probe.good()) {
        // No volume file: build it in-process (multi-threaded add) and save it.
        std::size_t bt = BUILD_THREADS ? BUILD_THREADS : std::thread::hardware_concurrency();
        index.reserve(index_limits_t(store->size(), std::max(bt, SEARCH_THREADS)));
        std::cout << "volume missing; building (" << bt << " threads, n=" << store->size() << ")... " << std::flush;
        stopwatch_t t;
        std::vector<std::thread> pool;
        std::size_t chunk = (store->size() + bt - 1) / bt;
        for (std::size_t tid = 0; tid < bt; ++tid) {
            std::size_t b = tid * chunk, e = std::min(b + chunk, store->size());
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
        std::error_code ec;
        std::filesystem::create_directories(VOLUME_DIR, ec);
        index.save(vpath.c_str());
        std::cout << "done (" << std::setprecision(1) << BUILD_TIME_MS << " ms), saved " << vpath << "\n";
    } else {
    std::cout << "loading volume " << vpath << "... " << std::flush;
    {
        stopwatch_t t;
        if (!index.load(vpath.c_str())) {
            std::cerr << "load failed\n";
            return 1;
        }
        if (index.size() != N_VAL) {
            std::cerr << "volume " << vpath << " holds " << index.size() << " vectors but --n " << N_VAL
                      << " was requested; keys >= N would index predicate structures out of bounds\n";
            return 1;
        }
        index.reserve(index_limits_t(index.size(), std::max<std::size_t>(SEARCH_THREADS, 1)));
        std::cout << "done (" << std::setprecision(1) << t.ms() << " ms, size " << index.size() << ")\n";
    }
    }
    {
        // Measured base-level average degree. Taken after both the build and the load
        // path; otherwise a fresh build reports 0.
        auto st0 = index.stats(0);
        MAVG_BASE = st0.nodes ? (double)st0.edges / (double)st0.nodes : 0.0;
    }

    std::vector<float> qdata;
    std::size_t qn = 0, qd = 0;
    bool q_ok = is_t2i    ? load_fbin(query_file.c_str(), qdata, qn, qd)
                : is_yfcc ? load_u8bin(query_file.c_str(), qdata, qn, qd)
                          : load_fvecs(query_file.c_str(), qdata, qn, qd);
    if (!q_ok) {
        std::cerr << "failed query " << query_file << "\n";
        return 1;
    }
    if (need_norm)
        normalize_rows(qdata, qn, qd);
    std::size_t nq = std::min(NUM_QUERIES, qn);
    std::vector<float const*> queries;
    std::vector<std::vector<vec_key_t>> ext_sets; // external mode: passing set per selected query
    token_csr_t pred_csr;                         // external mode: row -> tokens (prefilter-scan)
    pred_terms_t pred_terms;                      // external mode: query -> predicate tokens
    token_postings_t pred_postings;               // external mode: token -> rows (prefilter-index)
    double ext_sel = 0.0;
    std::vector<ext_group_t> ext_groups;          // --allowlist-spec: one per group
    column_t pred_column;                         // --predicate-column: range half of a mixed predicate
    if (!ALLOWLIST_SPECS.empty() && !ALLOWLIST_FILE.empty()) {
        std::cerr << "--allowlist-spec and --allowlist-file are exclusive\n";
        return 1;
    }
    if (!ALLOWLIST_SPECS.empty()) {
        if (!PRED_TERMS_FILE.empty()) {
            std::cerr << "with --allowlist-spec the predicate terms go in each spec (<sfal>|<label>|<sfpr>)\n";
            return 1;
        }
        stopwatch_t t;
        if (!PRED_CSR_FILE.empty() && !load_token_csr(PRED_CSR_FILE, N_VAL, pred_csr))
            return 1;
        if (!PRED_COLUMN_FILE.empty()) {
            pred_column.col_path = PRED_COLUMN_FILE;
            if (!pred_column.load_values(N_VAL))
                return 1;
        }
        ext_groups.resize(ALLOWLIST_SPECS.size());
        for (std::size_t i = 0; i < ALLOWLIST_SPECS.size(); ++i) {
            ext_group_t& eg = ext_groups[i];
            std::vector<std::string> f;
            {
                std::string s = ALLOWLIST_SPECS[i];
                std::size_t p = 0, b;
                while ((b = s.find('|', p)) != std::string::npos) {
                    f.push_back(s.substr(p, b - p));
                    p = b + 1;
                }
                f.push_back(s.substr(p));
            }
            if (f.empty() || f[0].empty() || f.size() > 4) {
                std::cerr << "--allowlist-spec takes <sfal>|<label>[|<sfpr>[|<sfrq>]], got " << ALLOWLIST_SPECS[i] << "\n";
                return 1;
            }
            eg.sfal = f[0];
            eg.label = f.size() > 1 && !f[1].empty() ? f[1] : std::filesystem::path(eg.sfal).stem().string();
            eg.sfpr = f.size() > 2 ? f[2] : "";
            eg.sfrq = f.size() > 3 ? f[3] : "";
            if (!load_allowlist_file(eg.sfal, N_VAL, eg.qidx, eg.sets)) {
                std::cerr << "failed allowlist file " << eg.sfal << "\n";
                return 1;
            }
            // One query set for the whole process: every group must list the same queries in the same
            // order, so that a cell's arms and the groups share one `queries` array.
            if (i > 0 && eg.qidx != ext_groups[0].qidx) {
                std::cerr << eg.sfal << " lists other queries than " << ext_groups[0].sfal << "\n";
                return 1;
            }
            if (!eg.sfpr.empty()) {
                if (PRED_CSR_FILE.empty()) {
                    std::cerr << "a spec with predicate terms needs --predicate-csr\n";
                    return 1;
                }
                if (!load_pred_terms(eg.sfpr, (std::size_t)pred_csr.ncol, eg.terms))
                    return 1;
                if (eg.terms.qidx != eg.qidx) {
                    std::cerr << eg.sfpr << ": query order differs from " << eg.sfal << "\n";
                    return 1;
                }
            }
            if (!eg.sfrq.empty()) {
                if (PRED_COLUMN_FILE.empty() || eg.sfpr.empty()) {
                    std::cerr << "a spec with a range file needs --predicate-column and predicate terms\n";
                    return 1;
                }
                if (!load_ranges(eg.sfrq, eg.qidx, eg.lo, eg.hi))
                    return 1;
            }
            double tot = 0;
            for (auto const& st : eg.sets)
                tot += (double)st.size();
            std::cout << "external group " << i << " '" << eg.label << "': " << eg.qidx.size() << " queries, mean selectivity "
                      << std::setprecision(4) << tot / (double)eg.qidx.size() / (double)N_VAL * 100 << "%"
                      << (eg.sfpr.empty() ? "" : ", terms") << (eg.sfrq.empty() ? "" : ", ranges") << "\n";
        }
        nq = ext_groups[0].qidx.size();
        queries.resize(nq);
        for (std::size_t i = 0; i < nq; ++i) {
            if (ext_groups[0].qidx[i] >= qn) {
                std::cerr << "allowlist query index " << ext_groups[0].qidx[i] << " >= " << qn << " loaded queries\n";
                return 1;
            }
            queries[i] = qdata.data() + ext_groups[0].qidx[i] * D_VAL;
        }
        std::cout << "external groups loaded (" << std::setprecision(1) << t.ms() << " ms)"
                  << (PRED_CSR_FILE.empty() ? "" : ", token csr " + std::to_string(pred_csr.nnz) + " pairs")
                  << (PRED_COLUMN_FILE.empty() ? "" : ", column " + std::filesystem::path(PRED_COLUMN_FILE).filename().string())
                  << "\n";
    } else if (!ALLOWLIST_FILE.empty()) {
        std::vector<std::size_t> qidx;
        if (!load_allowlist_file(ALLOWLIST_FILE, N_VAL, qidx, ext_sets)) {
            std::cerr << "failed allowlist file " << ALLOWLIST_FILE << "\n";
            return 1;
        }
        nq = qidx.size();
        queries.resize(nq);
        double tot = 0;
        for (std::size_t i = 0; i < nq; ++i) {
            if (qidx[i] >= qn) {
                std::cerr << "allowlist query index " << qidx[i] << " >= " << qn << " loaded queries\n";
                return 1;
            }
            queries[i] = qdata.data() + qidx[i] * D_VAL;
            tot += (double)ext_sets[i].size();
        }
        ext_sel = tot / (double)nq / (double)N_VAL;
        std::cout << "external allowlists: " << nq << " queries, label " << ALLOWLIST_LABEL << ", mean selectivity "
                  << std::setprecision(4) << ext_sel * 100 << "%\n";
        if (!PRED_CSR_FILE.empty() != !PRED_TERMS_FILE.empty()) {
            std::cerr << "--predicate-csr and --predicate-terms go together\n";
            return 1;
        }
        if (!PRED_CSR_FILE.empty()) {
            stopwatch_t t;
            if (!load_token_csr(PRED_CSR_FILE, N_VAL, pred_csr))
                return 1;
            if (!load_pred_terms(PRED_TERMS_FILE, (std::size_t)pred_csr.ncol, pred_terms))
                return 1;
            if (pred_terms.qidx.size() != nq) {
                std::cerr << "predicate terms cover " << pred_terms.qidx.size() << " queries but the allowlist has "
                          << nq << "\n";
                return 1;
            }
            for (std::size_t i = 0; i < nq; ++i)
                if (pred_terms.qidx[i] != qidx[i]) {
                    std::cerr << "predicate terms query order differs from the allowlist at entry " << i << "\n";
                    return 1;
                }
            std::cout << "predicate structures: " << pred_csr.nnz << " (row, token) pairs over " << pred_csr.ncol
                      << " tokens (" << std::setprecision(1) << t.ms() << " ms)\n";
        }
    } else {
        queries.resize(nq);
        for (std::size_t i = 0; i < nq; ++i)
            queries[i] = qdata.data() + i * D_VAL;
        std::cout << "queries: " << nq << " of " << qn << "\n";
    }

    std::error_code ec;
    std::filesystem::create_directories(GT_DIR, ec);

    NQ_USED = nq;
    std::vector<row_t> rows;
    std::size_t st = std::min(SEARCH_THREADS, nq);
    query_pool_t pool(st, nq);

    struct group_t {
        double s;
        std::string tag;
        std::size_t k; // conj mode: conjunction size
        double alpha;  // allowlist mode: correlation knob; NaN otherwise
        int col = -1;  // column mode: index into `columns`
        std::size_t cg = 0; // column mode: group index within that column
        int ext = -1;  // --allowlist-spec: index into `ext_groups`
    };
    auto alpha_tag = [](double a) {
        std::ostringstream o;
        o << std::showpos << std::defaultfloat << std::setprecision(3) << a;
        return o.str();
    };
    std::vector<group_t> groups;
    conj_t conj;
    conj_t const* conj_ptr = nullptr;
    std::vector<column_t> columns; // column mode; sized once so pointers into it stay valid
    if (FILTER == "column") {
        if (COLUMN_SPECS.empty()) {
            std::cerr << "--filter column needs at least one --column-spec <column file>:<query file>\n";
            return 1;
        }
        columns.resize(COLUMN_SPECS.size());
        for (std::size_t i = 0; i < COLUMN_SPECS.size(); ++i) {
            std::size_t sep = COLUMN_SPECS[i].find(':');
            if (sep == std::string::npos) {
                std::cerr << "--column-spec takes <column file>:<query file>, got " << COLUMN_SPECS[i] << "\n";
                return 1;
            }
            columns[i].col_path = COLUMN_SPECS[i].substr(0, sep);
            columns[i].q_path = COLUMN_SPECS[i].substr(sep + 1);
            stopwatch_t t;
            if (!columns[i].load(N_VAL, nq))
                return 1;
            std::cout << "column " << std::filesystem::path(columns[i].col_path).filename().string() << ": " << N_VAL
                      << " values, " << columns[i].groups.size() << " groups from "
                      << std::filesystem::path(columns[i].q_path).filename().string() << " (" << std::setprecision(1)
                      << t.ms() << " ms)\n";
            for (std::size_t g = 0; g < columns[i].groups.size(); ++g)
                groups.push_back({std::nan(""), columns[i].groups[g].label, 0, std::nan(""), (int)i, g});
        }
    } else if (FILTER == "conj") {
        stopwatch_t t;
        conj.build(N_VAL, CONJ_ATTRS, CONJ_CARD, SEED);
        conj.set_queries(nq, SEED);
        conj_ptr = &conj;
        std::cout << "conj attributes: " << CONJ_ATTRS << " x uint8 (card " << CONJ_CARD << ") built in "
                  << std::setprecision(1) << t.ms() << " ms\n";
        for (std::size_t k : CONJ_KS) {
            if (k < 1 || k > CONJ_ATTRS)
                continue;
            double s = std::pow(1.0 / (double)CONJ_CARD, (double)k);
            std::ostringstream o;
            o << std::defaultfloat << std::setprecision(3) << s * 100 << "% k=" << k;
            groups.push_back({s, o.str(), k, std::nan("")});
        }
    } else if (!ext_groups.empty()) {
        for (std::size_t i = 0; i < ext_groups.size(); ++i)
            groups.push_back({std::nan(""), ext_groups[i].label, 0, std::nan(""), -1, 0, (int)i});
    } else if (ALLOWLIST_FILE.empty()) {
        for (double s : SELECTIVITIES)
            for (double a : ALPHAS)
                // The tag is "<sel>%" when one alpha is given; a sweep appends " a=+0.4" and so on.
                groups.push_back({s, ALPHAS.size() == 1 ? sel_tag(s) : sel_tag(s) + " a=" + alpha_tag(a), 0, a});
    } else
        groups.push_back({ext_sel, ALLOWLIST_LABEL, 0, std::nan("")});
    for (group_t const& grp : groups) {
        double s = grp.s;
        if (conj_ptr)
            conj.k = grp.k;
        column_t const* col_ptr = grp.col >= 0 ? &columns[(std::size_t)grp.col] : nullptr;
        ext_group_t* eg_ptr = grp.ext >= 0 ? &ext_groups[(std::size_t)grp.ext] : nullptr;
        // predicate terms of this group (--allowlist-spec) or of the single external file
        pred_terms_t const& terms = eg_ptr ? eg_ptr->terms : pred_terms;
        bool const have_range = eg_ptr && !eg_ptr->lo.empty(); // mixed predicate: terms AND column range
        // Column and spec groups take their selectivity from the data: the median realized
        // passing fraction, filled in below once the sets exist. Every other mode knows it up front.
        std::size_t m = std::isnan(s) ? 0 : (std::size_t)std::llround((double)N_VAL * s);
        if (!std::isnan(s) && m < K_VAL)
            m = K_VAL;
        std::string stag = grp.tag;
        double alpha = std::isnan(grp.alpha) ? 0.0 : grp.alpha;
        if (col_ptr)
            std::cout << "\n=== group " << stag << " (column "
                      << std::filesystem::path(col_ptr->col_path).filename().string() << ") ===\n";
        else if (eg_ptr)
            std::cout << "\n=== group " << stag << " (external " << std::filesystem::path(eg_ptr->sfal).filename().string()
                      << ") ===\n";
        else
            std::cout << "\n=== selectivity " << stag << " (allowlist size " << m << ") ===\n";
        std::vector<double> qce; // per-query c_e

        // per-query allowlist: |alpha|*m from the query's nearest (alpha>0) or
        // farthest (alpha<0) base vectors, the rest uniform random. Seeded.
        std::vector<std::vector<vec_key_t>> sets(nq);
        if (conj_ptr) {
            // materialized passing set: feeds the c_e measurement and the "prefilter" arm
            // (given set); the ground truth goes through the predicate
            pool.run([&](std::size_t q, std::size_t) { sets[q] = conj.passing_scan(q); });
        } else if (col_ptr) {
            stopwatch_t t;
            pool.run([&](std::size_t q, std::size_t) { sets[q] = col_ptr->passing_scan(grp.cg, q); });
            std::vector<double> sz(nq);
            double tot = 0;
            for (std::size_t q = 0; q < nq; ++q)
                tot += sz[q] = (double)sets[q].size();
            std::sort(sz.begin(), sz.end());
            m = (std::size_t)std::llround(pct_sorted(sz, 0.5));
            s = (double)m / (double)N_VAL;
            double target = col_ptr->groups[grp.cg].target;
            std::cout << "  realized selectivity: median " << std::setprecision(4) << s * 100 << "% (p10 "
                      << pct_sorted(sz, 0.1) / (double)N_VAL * 100 << "%, p90 "
                      << pct_sorted(sz, 0.9) / (double)N_VAL * 100 << "%, mean " << tot / nq / (double)N_VAL * 100
                      << "%), passing median " << m;
            if (!std::isnan(target))
                std::cout << ", target " << target * 100 << "%";
            std::cout << " (" << std::setprecision(1) << t.ms() << " ms)\n";
        } else if (eg_ptr) {
            // Each spec group is used once; hand its sets over instead of copying gigabytes.
            sets = std::move(eg_ptr->sets);
            std::vector<double> sz(nq);
            double tot = 0;
            for (std::size_t q = 0; q < nq; ++q)
                tot += sz[q] = (double)sets[q].size();
            std::sort(sz.begin(), sz.end());
            m = (std::size_t)std::llround(pct_sorted(sz, 0.5));
            s = (double)m / (double)N_VAL;
            std::cout << "  realized selectivity: median " << std::setprecision(4) << s * 100 << "% (p10 "
                      << pct_sorted(sz, 0.1) / (double)N_VAL * 100 << "%, p90 " << pct_sorted(sz, 0.9) / (double)N_VAL * 100
                      << "%, mean " << tot / nq / (double)N_VAL * 100 << "%), passing median " << m << ", min "
                      << (std::size_t)sz.front() << "\n";
        } else if (!ALLOWLIST_FILE.empty()) {
            sets = ext_sets;
        } else {
        std::size_t n_corr = (std::size_t)std::llround(std::fabs(alpha) * (double)m);
        if (n_corr > m)
            n_corr = m;
        if (alpha != 0.0)
            std::cout << "  correlated: " << n_corr << "/" << m << " from "
                      << (alpha > 0 ? "nearest" : "farthest") << " (alpha=" << std::setprecision(3) << alpha << ")\n";
        if (MEASURE_CE)
            qce.assign(nq, std::nan(""));
        pool.run([&](std::size_t q, std::size_t) {
            std::mt19937_64 rng(SEED ^ (0x9e3779b97f4a7c15ULL * (q + 1)) ^ (std::uint64_t)m);
            std::unordered_set<vec_key_t> chosen;
            chosen.reserve(m * 2);
            // Distances to every base vector. They are needed to pick the correlated part
            // and, when c_e is measured, to rank the finished set. At alpha = 0 the pass
            // exists only for c_e, so it is skipped when --ce is off.
            std::vector<std::pair<float, vec_key_t>> dd;
            float const* qv = queries[q];
            if (n_corr > 0 || MEASURE_CE) {
                dd.resize(N_VAL);
                for (std::size_t i = 0; i < N_VAL; ++i)
                    dd[i] = {metric_l2sq_gt<float, float>{}(qv, metric.row(i), D_VAL), (vec_key_t)i};
            }
            if (n_corr > 0) {
                if (alpha > 0)
                    std::nth_element(dd.begin(), dd.begin() + n_corr, dd.end(),
                                     [](auto const& a, auto const& b) { return a.first < b.first; });
                else
                    std::nth_element(dd.begin(), dd.begin() + n_corr, dd.end(),
                                     [](auto const& a, auto const& b) { return a.first > b.first; });
                for (std::size_t i = 0; i < n_corr; ++i)
                    chosen.insert(dd[i].second);
            }
            std::uniform_int_distribution<std::size_t> U(0, N_VAL - 1);
            while (chosen.size() < m)
                chosen.insert((vec_key_t)U(rng));
            sets[q].assign(chosen.begin(), chosen.end());
            std::sort(sets[q].begin(), sets[q].end());
            if (MEASURE_CE) {
                // c_e = 1 - 2 * mean normalized rank of the passing set. rank(x) counts the base
                // vectors strictly closer than x, so sum_x rank(x) = sum_i |{x : d(x) > d_i}|.
                // This is O(N log m) over the distance array. The set's own distances (m of them)
                // are recomputed, because nth_element permuted dd.
                std::vector<float> ds(m);
                for (std::size_t j = 0; j < m; ++j)
                    ds[j] = metric_l2sq_gt<float, float>{}(qv, metric.row(sets[q][j]), D_VAL);
                std::sort(ds.begin(), ds.end());
                std::uint64_t rank_sum = 0;
                for (std::size_t i = 0; i < N_VAL; ++i)
                    rank_sum += (std::uint64_t)(ds.end() - std::upper_bound(ds.begin(), ds.end(), dd[i].first));
                double mean_norm_rank = (double)rank_sum / (double)m / (double)N_VAL;
                qce[q] = 1.0 - 2.0 * mean_norm_rank;
            }
        });
        }
        // c_e for the modes whose passing sets come from data (conjunction attributes, an
        // external allowlist, a natural column). One full distance pass per query, then the
        // normalized mean rank of the set. Same definition as the allowlist branch above.
        if (MEASURE_CE && qce.empty()) {
            stopwatch_t t;
            qce.assign(nq, std::nan(""));
            pool.run([&](std::size_t q, std::size_t) {
                auto const& set = sets[q];
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
            std::cout << "  c_e pass: " << std::setprecision(1) << t.ms() << " ms\n";
        }
        {
            group_stat_t gs{s, grp.alpha, m, false, 0, 0, 0};
            gs.label = stag;
            gs.nq = nq;
            gs.target = col_ptr ? col_ptr->groups[grp.cg].target : s;
            if (!qce.empty()) {
                std::vector<double> v;
                for (double c : qce)
                    if (!std::isnan(c))
                        v.push_back(c);
                if (!v.empty()) {
                    std::sort(v.begin(), v.end());
                    double sum = 0;
                    for (double c : v)
                        sum += c;
                    gs.has_ce = true;
                    gs.ce_mean = sum / (double)v.size();
                    gs.ce_p10 = pct_sorted(v, 0.10);
                    gs.ce_p90 = pct_sorted(v, 0.90);
                    std::cout << "  c_e: mean " << std::setprecision(4) << gs.ce_mean << " p10 " << gs.ce_p10 << " p90 "
                              << gs.ce_p90 << " (alpha " << alpha << ")\n";
                }
            }
            GROUP_STATS.push_back(gs);
        }

        // O(1) membership structure per query, built from the allowlist. Unused in conj and
        // column modes, whose predicates read the attribute arrays directly.
        std::vector<dense_bitset_t> hsets(nq);
        if (!conj_ptr && !col_ptr)
            pool.run([&](std::size_t q, std::size_t) { hsets[q].build(N_VAL, sets[q]); });
        auto make_pred = [&](std::size_t q) {
            allowlist_predicate_t p;
            p.q = q;
            if (col_ptr) {
                p.col = col_ptr->val.data();
                p.lo = col_ptr->groups[grp.cg].lo[q];
                p.hi = col_ptr->groups[grp.cg].hi[q];
            } else {
                p.hset = &hsets[q];
                p.conj = conj_ptr;
            }
            return p;
        };

        // GT: exact filtered search at --k, cached in a file whose name carries the dataset,
        // N, D, M, ef_construction, the filter kind with its parameters (s, alpha, seed, or
        // the source files), the base fingerprint, k and nq
        std::ostringstream gp;
        gp << GT_DIR << "/" << DATASET << "_" << N_VAL << "_" << D_VAL << "_m" << M_VAL << "_ef" << EF_CONSTRUCTION_VAL;
        if (conj_ptr)
            gp << "_conj_a" << CONJ_ATTRS << "_c" << CONJ_CARD << "_k" << grp.k << "_seed" << SEED;
        else if (col_ptr)
            // The ground truth depends on the column values and on the query ranges, so both files
            // are fingerprinted; the group index picks the group inside the query file.
            gp << "_col_" << std::filesystem::path(col_ptr->q_path).stem().string() << "_g" << grp.cg << "_f"
               << file_fingerprint(col_ptr->col_path) << "_f" << file_fingerprint(col_ptr->q_path);
        else if (eg_ptr)
            // The SFAL alone fixes the passing sets, mixed or not, so it alone keys the ground truth.
            gp << "_ext_" << std::filesystem::path(eg_ptr->sfal).stem().string() << "_f" << file_fingerprint(eg_ptr->sfal);
        else if (ALLOWLIST_FILE.empty())
            gp << "_allowlist_s" << std::setprecision(6) << s << "_a" << alpha << "_seed" << SEED;
        else
            gp << "_ext_" << std::filesystem::path(ALLOWLIST_FILE).stem().string() << "_f"
               << file_fingerprint(ALLOWLIST_FILE);
        // The GT depends on the base vectors (and, in external mode, on the allowlist file).
        // Keying on their fingerprint prevents reusing a cache after the file was replaced
        // under the same name.
        gp << "_d" << file_fingerprint(base_file);
        gp << "_gt_k" << K_VAL << "_nq" << nq << ".bin";
        std::string gpath = gp.str();
        std::vector<std::vector<vec_key_t>> gt(nq);
        bool gt_loaded = false;
        // A cache file is usable when its header covers this run: same key prefix, at least
        // nq queries and at least k neighbors. The exact name is tried first. Failing that,
        // any sibling with a larger _nq is accepted (same rule as bench_acorn). So a run with
        // 992 queries reuses the 1000-query cache instead of recomputing the ground truth.
        auto try_load = [&](std::string const& path) {
            std::ifstream ifs(path, std::ios::binary);
            if (!ifs.good())
                return false;
            char magic[4] = {};
            uint32_t fnq = 0, fk = 0;
            ifs.read(magic, 4);
            ifs.read(reinterpret_cast<char*>(&fnq), 4);
            ifs.read(reinterpret_cast<char*>(&fk), 4);
            if (std::string(magic, 4) != "SFGT" || fnq < nq || fk < K_VAL)
                return false;
            for (auto& g : gt)
                g.clear();
            std::vector<uint64_t> buf(fk);
            for (std::size_t q = 0; q < fnq; ++q) {
                uint32_t cnt = 0;
                ifs.read(reinterpret_cast<char*>(&cnt), 4);
                ifs.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)fk * 8);
                if (q < nq)
                    for (std::size_t i = 0; i < std::min<std::size_t>(cnt, K_VAL); ++i)
                        if (buf[i] != UINT64_MAX)
                            gt[q].push_back((vec_key_t)buf[i]);
            }
            if (!ifs.good())
                return false;
            if (path != gpath)
                std::cout << "  GT: using " << std::filesystem::path(path).filename().string() << " (nq " << fnq
                          << " >= " << nq << ")\n";
            return true;
        };
        gt_loaded = try_load(gpath);
        if (!gt_loaded) {
            std::string prefix = gpath.substr(0, gpath.rfind("_nq"));
            std::error_code ec;
            for (auto const& e : std::filesystem::directory_iterator(GT_DIR, ec)) {
                std::string cand = e.path().string();
                if (cand.size() > prefix.size() && cand.compare(0, prefix.size(), prefix) == 0 &&
                    cand.compare(prefix.size(), 3, "_nq") == 0 && try_load(cand)) {
                    gt_loaded = true;
                    break;
                }
            }
        }
        if (!gt_loaded) {
            std::cout << "  computing GT... " << std::flush;
            stopwatch_t t;
            pool.run([&](std::size_t q, std::size_t tid) {
                index_search_config_t sc;
                sc.exact = true;
                sc.thread = tid;
                allowlist_predicate_t pred = make_pred(q);
                auto r = search_by_mode(pred, [&](auto const& mp) {
                    return index.search(queries[q], K_VAL, metric, sc, mp, dummy_prefetch_t{}, dummy_preevaluate_t{});
                });
                gt[q].reserve(r.size());
                for (std::size_t i = 0; i < r.size(); ++i)
                    gt[q].push_back(r[i].member.key);
            });
            std::cout << "done (" << std::setprecision(1) << t.ms() << " ms)\n";
            std::ofstream ofs(gpath, std::ios::binary);
            ofs.write("SFGT", 4);
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
        }

        for (auto const& mode : MODES) {
            // prefilter kinds: 1 = given passing set (optimistic), 2 = attribute scan, 3 = per-attribute index
            int pf_kind = iequals(mode, "prefilter")         ? 1
                          : iequals(mode, "prefilter-scan")  ? 2
                          : iequals(mode, "prefilter-index") ? 3
                                                             : 0;
            bool have_terms = !terms.terms.empty();
            if (pf_kind == 3 && !conj_ptr && !have_terms && !col_ptr) {
                // A per-attribute index needs attributes to index. A synthetic allowlist is an
                // arbitrary key set per query, derived from nothing that can be prepared ahead of
                // time. So the "prefilter" arm (set handed over) already is its index variant.
                std::cout << "  " << mode << " needs --filter conj, --filter column or --predicate-csr; skipped\n";
                continue;
            }
            if (pf_kind == 2 && !conj_ptr && !have_terms && !col_ptr && (!ALLOWLIST_FILE.empty() || eg_ptr))
                std::cerr << "  note: scanning allowlist membership, not the predicate the file came from; "
                             "pass --predicate-csr/--predicate-terms for the real cost\n";
            if (pf_kind == 3 && have_terms && pred_postings.rows.empty()) {
                stopwatch_t t;
                pred_postings.build(pred_csr);
                std::cout << "  token postings: " << pred_postings.rows.size() << " entries ("
                          << std::setprecision(1) << t.ms() << " ms)\n";
            }
            if (pf_kind == 3 && col_ptr && col_ptr->idx_keys.empty()) {
                stopwatch_t t;
                columns[(std::size_t)grp.col].build_index();
                std::cout << "  sorted attribute index: " << col_ptr->idx_keys.size() << " entries ("
                          << std::setprecision(1) << t.ms() << " ms)\n";
            }
            if (pf_kind == 3 && have_range && pred_column.idx_keys.empty()) {
                stopwatch_t t;
                pred_column.build_index();
                std::cout << "  sorted attribute index (" << std::filesystem::path(pred_column.col_path).filename().string()
                          << "): " << pred_column.idx_keys.size() << " entries (" << std::setprecision(1) << t.ms()
                          << " ms)\n";
            }
            bool is_prefilter = pf_kind > 0;
            bool is_racorn = iequals(mode, "racorn-1");
            bool is_3hop = iequals(mode, "acorn-1-3hop");
            int navix = navix_of(mode);
            double ff = iequals(mode, "hnsw") ? -1.0 : ACORN_THRESHOLD;
            double three_hop = is_3hop ? THREE_HOP_TAU : 0.0;
            std::vector<double> brs = is_racorn ? BBR_VALUES : std::vector<double>{0.0};
            for (double br : brs) {
                for (std::size_t ef : EF_VALUES) {
                    std::vector<double> qms(nq, std::nan("")), qrec(nq, 0), qdist(nq, 0), qvis(nq, 0), qpe(nq, 0);
                    std::vector<std::vector<double>> perq(nq);
                    std::vector<double> batch_ms;
                    // warmup
                    for (std::size_t w = 0; w < BENCH_WARMUP; ++w)
                        pool.run([&](std::size_t q, std::size_t tid) {
                            allowlist_predicate_t pred = make_pred(q);
                            if (is_prefilter)
                                return;
                            index_search_config_t sc;
                            sc.expansion = ef;
                            sc.filter_first_threshold = ff;
                            sc.bridge_budget_ratio = br;
                            sc.navix_heuristic = navix;
                            sc.three_hop_threshold = three_hop;
                            sc.stride_sampling = STRIDE_SAMPLING;
                            sc.thread = tid;
                            auto r = search_by_mode(pred, [&](auto const& mp) {
                                return index.search(queries[q], K_VAL, metric, sc, mp, dummy_prefetch_t{},
                                                    dummy_preevaluate_t{});
                            });
                            (void)r;
                        });
                    for (int run = 0; run < (int)BENCH_RUNS; ++run) {
                        stopwatch_t bt;
                        pool.run([&](std::size_t q, std::size_t tid) {
                            allowlist_predicate_t pred = make_pred(q);
                            stopwatch_t t;
                            std::vector<vec_key_t> got;
                            double dist = 0, vis = 0, pe = 0;
                            if (is_prefilter) {
                                // candidate gathering is part of the timed work for the scan/index kinds
                                std::vector<vec_key_t> gathered;
                                std::size_t scanned = 0;
                                if (pf_kind == 2) {
                                    // No index of any kind: test the predicate on every key.
                                    if (conj_ptr) {
                                        gathered = conj.passing_scan(q);
                                    } else if (col_ptr) {
                                        gathered = col_ptr->passing_scan(grp.cg, q);
                                    } else if (have_terms) {
                                        auto const& tk = terms.terms[q];
                                        bool any = terms.op[q] == 1;
                                        // Mixed predicate: the range test is one array read, so it goes
                                        // first and the token lookups run only on keys inside the range.
                                        std::uint32_t const rlo = have_range ? eg_ptr->lo[q] : 0;
                                        std::uint32_t const rhi = have_range ? eg_ptr->hi[q] : 0;
                                        for (std::size_t key = 0; key < N_VAL; ++key) {
                                            if (have_range) {
                                                std::uint32_t v = pred_column.val[key];
                                                if (v < rlo || v > rhi)
                                                    continue;
                                            }
                                            bool ok = !any;
                                            for (int32_t t : tk) {
                                                bool h = pred_csr.has(key, t);
                                                if (any && h) { ok = true; break; }
                                                if (!any && !h) { ok = false; break; }
                                            }
                                            if (ok && !tk.empty())
                                                gathered.push_back((vec_key_t)key);
                                        }
                                    } else {
                                        for (std::size_t key = 0; key < N_VAL; ++key)
                                            if (hsets[q].contains((vec_key_t)key))
                                                gathered.push_back((vec_key_t)key);
                                    }
                                    scanned = N_VAL;
                                } else if (pf_kind == 3 && conj_ptr) {
                                    gathered = conj.passing_index(q, scanned);
                                } else if (pf_kind == 3 && col_ptr) {
                                    gathered = col_ptr->passing_index(grp.cg, q, scanned);
                                } else if (pf_kind == 3 && have_range) {
                                    // Mixed predicate, single-type indexes: a token posting index and a sorted
                                    // index on the column exist, but nothing indexes the conjunction. The
                                    // best plan reads the shorter side and checks the other condition on each
                                    // key it returns. Token side: the AND's shortest list, or the OR's union.
                                    std::vector<int32_t> const& tk = terms.terms[q];
                                    bool any = terms.op[q] == 1;
                                    std::uint32_t const rlo = eg_ptr->lo[q], rhi = eg_ptr->hi[q];
                                    std::size_t tok_len = any ? 0 : (std::size_t)-1;
                                    for (int32_t t : tk)
                                        tok_len = any ? tok_len + pred_postings.list_len(t)
                                                      : std::min(tok_len, pred_postings.list_len(t));
                                    auto sl = pred_column.index_slice(rlo, rhi);
                                    std::size_t rng_len = sl.second - sl.first;
                                    auto tokens_pass = [&](std::size_t key) {
                                        bool ok = !any;
                                        for (int32_t t : tk) {
                                            bool h = pred_csr.has(key, t);
                                            if (any && h) { ok = true; break; }
                                            if (!any && !h) { ok = false; break; }
                                        }
                                        return ok && !tk.empty();
                                    };
                                    if (rng_len <= tok_len) {
                                        scanned = rng_len;
                                        for (std::size_t i = sl.first; i < sl.second; ++i)
                                            if (tokens_pass(pred_column.idx_keys[i]))
                                                gathered.push_back((vec_key_t)pred_column.idx_keys[i]);
                                    } else {
                                        // OR: walk every list, a key may sit in several (dedupe by sort+unique)
                                        for (int32_t t : tk) {
                                            if (!any && pred_postings.list_len(t) != tok_len)
                                                continue; // AND: the shortest list only, the rest is post-checked
                                            std::uint32_t const* L = pred_postings.list_begin(t);
                                            std::size_t len = pred_postings.list_len(t);
                                            scanned += len;
                                            for (std::size_t i = 0; i < len; ++i) {
                                                std::uint32_t v = pred_column.val[L[i]];
                                                if (v >= rlo && v <= rhi && (any || tokens_pass(L[i])))
                                                    gathered.push_back((vec_key_t)L[i]);
                                            }
                                            if (!any)
                                                break;
                                        }
                                        if (any && tk.size() > 1) {
                                            std::sort(gathered.begin(), gathered.end());
                                            gathered.erase(std::unique(gathered.begin(), gathered.end()), gathered.end());
                                        }
                                    }
                                } else if (pf_kind == 3) {
                                    // Read the query's posting lists: union for OR, intersect for AND.
                                    std::vector<int32_t> tk = terms.terms[q];
                                    std::sort(tk.begin(), tk.end(), [&](int32_t a, int32_t b) {
                                        return pred_postings.list_len(a) < pred_postings.list_len(b);
                                    });
                                    std::vector<std::uint32_t> cur;
                                    if (!tk.empty()) {
                                        cur.assign(pred_postings.list_begin(tk[0]),
                                                   pred_postings.list_begin(tk[0]) + pred_postings.list_len(tk[0]));
                                        scanned = cur.size();
                                        for (std::size_t j = 1; j < tk.size(); ++j) {
                                            std::size_t L = pred_postings.list_len(tk[j]);
                                            scanned += L;
                                            std::vector<std::uint32_t> nxt;
                                            if (terms.op[q] == 1) {
                                                nxt.reserve(cur.size() + L);
                                                std::set_union(cur.begin(), cur.end(), pred_postings.list_begin(tk[j]),
                                                               pred_postings.list_begin(tk[j]) + L,
                                                               std::back_inserter(nxt));
                                            } else {
                                                if (cur.empty())
                                                    break;
                                                nxt.reserve(std::min(cur.size(), L));
                                                std::set_intersection(cur.begin(), cur.end(),
                                                                      pred_postings.list_begin(tk[j]),
                                                                      pred_postings.list_begin(tk[j]) + L,
                                                                      std::back_inserter(nxt));
                                            }
                                            cur.swap(nxt);
                                        }
                                    }
                                    gathered.assign(cur.begin(), cur.end());
                                }
                                std::vector<vec_key_t> const& cand = pf_kind == 1 ? sets[q] : gathered;
                                using pr = std::pair<float, vec_key_t>;
                                std::vector<pr> heap;
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
                                vis = pf_kind == 1 ? (double)cand.size() : (double)scanned; // keys touched to gather
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
                                auto r = search_by_mode(pred, [&](auto const& mp) {
                                    return index.search(queries[q], K_VAL, metric, sc, mp, dummy_prefetch_t{},
                                                        dummy_preevaluate_t{});
                                });
                                for (std::size_t i = 0; i < r.size(); ++i)
                                    got.push_back(r[i].member.key);
                                dist = (double)r.computed_distances;
                                vis = (double)r.visited_members;
                                pe = (double)r.predicate_evaluations;
                            }
                            perq[q].push_back(t.ms());
                            if (run == 0) {
                                qrec[q] = recall_of(got, gt[q], K_VAL);
                                qdist[q] = dist;
                                qvis[q] = vis;
                                qpe[q] = pe;
                            }
                        });
                        batch_ms.push_back(bt.ms());
                    }
                    for (std::size_t q = 0; q < nq; ++q)
                        qms[q] = median(perq[q]);
                    double med_ms = median(qms), med_batch = median(batch_ms);
                    std::vector<double> mss(qms);
                    std::sort(mss.begin(), mss.end());
                    double avg = 0;
                    for (double r : qrec)
                        avg += r;
                    avg /= nq;
                    std::vector<double> rs(qrec);
                    std::sort(rs.begin(), rs.end());
                    row_t row;
                    row.label = stag + " | " + mode + (is_prefilter ? "" : " ef=" + std::to_string(ef));
                    if (is_racorn) {
                        std::ostringstream o;
                        o << std::setprecision(2) << br;
                        row.label += " br=" + o.str();
                    }
                    row.mode = pf_kind == 1    ? "Prefilter"
                               : pf_kind == 2  ? "Prefilter-scan"
                               : pf_kind == 3  ? "Prefilter-index"
                               : iequals(mode, "hnsw")     ? "HNSW"
                               : iequals(mode, "acorn-1")  ? "ACORN-1"
                               : is_3hop                   ? "ACORN-1-3hop"
                               : iequals(mode, "racorn-1") ? "RACORN-1"
                               : navix == 8                ? "NaviX-local"
                               : navix == 5                ? "NaviX-local-paper"
                               : navix == 1                ? "NaviX-onehop-s"
                               : navix == 2                ? "NaviX-blind"
                               : navix == 3                ? "NaviX-directed"
                                                           : mode;
                    row.selectivity = s;
                    row.alpha = grp.alpha;
                    row.ef_search = is_prefilter ? 0 : ef;
                    row.bridge_budget_ratio = is_racorn ? br : 0.0;
                    row.latency_ms = med_ms;
                    row.latency_p50 = pct_sorted(mss, 0.5);
                    row.latency_p90 = pct_sorted(mss, 0.9);
                    row.latency_p99 = pct_sorted(mss, 0.99);
                    row.recall = avg;
                    row.recall_p50 = pct_sorted(rs, 0.5);
                    row.recall_p90 = pct_sorted(rs, 0.1);
                    row.recall_p99 = pct_sorted(rs, 0.01);
                    row.dist_computed = median(qdist);
                    row.nodes_visited = median(qvis);
                    row.predicate_evals = median(qpe);
                    row.qps = med_batch > 0 ? (double)nq / (med_batch / 1000.0) : 0.0;
                    row.num_queries = nq;
                    row.threads = st;
                    rows.push_back(row);
                    std::cout << "  " << row.mode << " ef=" << (is_prefilter ? 0 : ef)
                              << (is_racorn ? (" br=" + std::to_string(br)) : "") << " -> recall=" << std::setprecision(4)
                              << avg << " lat=" << std::setprecision(2) << med_ms << "ms qps=" << std::setprecision(0)
                              << row.qps << "\n";
                    if (is_prefilter)
                        break;
                    if (RECALL_TARGET > 0 && avg >= RECALL_TARGET)
                        break;
                }
            }
        }
    }
    if (!JSON_PATH.empty())
        write_json(JSON_PATH, rows);
    std::cout << "done.\n";
    return 0;
}

static void print_help() {
    std::cout
        << "Usage: bench_synthfilter [options]\n"
        << "\n"
        << "Predicates on the real data, one group per (selectivity or spec), one row per (method, group,\n"
        << "ef, br). The index volume is read from --volume-dir and built there when missing; the ground\n"
        << "truth of each group is computed exactly in this process and cached in --gt-dir.\n"
        << "\n"
        << "-- Data and index -------------------------------------------------------\n"
        << "  --dataset <name>          sift, gist, text2image, yfcc, laion (default: sift)\n"
        << "  --n <N>                   Number of base vectors (default: 100000000)\n"
        << "  --d <N>                   Dimensions (default: per dataset)\n"
        << "  --m <N>                   Graph connectivity M of the volume (default: 16)\n"
        << "  --ef-construction <N>     ef_construction of the volume (default: 100)\n"
        << "  --build-threads <N>       Threads when the volume is missing and gets built (default: all)\n"
        << "  --data-dir <dir>          Raw vectors (default: data/raw)\n"
        << "  --volume-dir <dir>        Index volumes (default: data/volumes)\n"
        << "  --gt-dir <dir>            Ground-truth cache (default: data/gt)\n"
        << "\n"
        << "-- Predicate ------------------------------------------------------------\n"
        << "  --filter <kind>           allowlist (default), column or conj\n"
        << "  --selectivity <list>      allowlist: the selectivities, as fractions or percents\n"
        << "                            (default: 0.05,0.03,0.01,0.005,0.001)\n"
        << "  --alpha <list>            allowlist: the set correlation to the query, one group per\n"
        << "                            (selectivity, alpha) (default: 0)\n"
        << "  --seed <N>                Seed of the allowlist and conj draws (default: 12345)\n"
        << "  --column-spec <col>:<q>   column: a metadata column (.sfcl) and its queries (.sfcq); repeatable\n"
        << "  --conj-attrs <N>          conj: number of uniform categorical attributes (default: 6)\n"
        << "  --conj-card <N>           conj: cardinality of each attribute (default: 3)\n"
        << "  --conj-k <list>           conj: the conjunction lengths k, one group each (default: 1,2,3,4,5,6)\n"
        << "  --allowlist-spec <spec>   external passing sets: <sfal>|<label>[|<sfpr>[|<sfrq>]]; repeatable\n"
        << "  --predicate-csr <file>    external: the key-token matrix (.spmat) the .sfpr terms refer to\n"
        << "  --predicate-column <file> external: the metadata column (.sfcl) the .sfrq ranges refer to\n"
        << "  --allowlist-file <file>   external: one passing-set file as a single group\n"
        << "  --allowlist-label <text>  external: the label of that group (default: ext)\n"
        << "  --predicate-terms <file>  external: the .sfpr terms of the --allowlist-file group\n"
        << "\n"
        << "-- Search ---------------------------------------------------------------\n"
        << "  --modes <list>            hnsw, acorn-1, acorn-1-3hop, racorn-1, navix-local, prefilter,\n"
        << "                            prefilter-scan, prefilter-index (default: prefilter,hnsw,acorn-1,racorn-1)\n"
        << "  --ef-values <list>        The ef values, in order (default: 100,200,...,1600)\n"
        << "  --recall-target <F>       Stop a series once recall reaches this value (default: 0.97)\n"
        << "  --bridge-budget-ratios <list>  RACORN-1's br, one series per value (default: 1.0,2.0)\n"
        << "  --filter-first-threshold <F>   The entry gate of ACORN-1 and RACORN-1 (default: 0.3)\n"
        << "  --three-hop-threshold <F>      The trigger of acorn-1-3hop (default: 1.0)\n"
        << "  --stride on|off           Stride sampling or prefix truncation (default: on)\n"
        << "  --ce on|off               Measure the correlation c_e of each group (default: on)\n"
        << "  --k <N>                   Neighbors per query (default: 100)\n"
        << "  --nq <N>                  Queries (default: 1000)\n"
        << "  --runs <N> --warmup <N>   Timed runs (median) after warm-up passes (default: 3, 2)\n"
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
    auto sd = [](std::string s) {
        std::vector<double> v;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ',')) {
            if (t.empty())
                continue;
            if (t.back() == '%')
                v.push_back(std::stod(t.substr(0, t.size() - 1)) / 100.0);
            else
                v.push_back(std::stod(t));
        }
        return v;
    };
    auto ssz = [](std::string s) {
        std::vector<std::size_t> v;
        std::stringstream ss(s);
        std::string t;
        while (std::getline(ss, t, ','))
            if (!t.empty())
                v.push_back((std::size_t)std::stoull(t));
        return v;
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dataset")
            DATASET = need(i);
        else if (a == "--n")
            N_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--d")
            D_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--m")
            M_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--ef-construction")
            EF_CONSTRUCTION_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--k")
            K_VAL = (std::size_t)std::stoull(need(i));
        else if (a == "--nq")
            NUM_QUERIES = (std::size_t)std::stoull(need(i));
        else if (a == "--search-threads")
            SEARCH_THREADS = (std::size_t)std::stoull(need(i));
        else if (a == "--runs")
            BENCH_RUNS = (std::size_t)std::stoull(need(i));
        else if (a == "--warmup")
            BENCH_WARMUP = (std::size_t)std::stoull(need(i));
        else if (a == "--filter-first-threshold")
            ACORN_THRESHOLD = std::stod(need(i));
        else if (a == "--recall-target")
            RECALL_TARGET = std::stod(need(i));
        else if (a == "--alpha")
            ALPHAS = sd(need(i));
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
        }
        else if (a == "--seed")
            SEED = std::stoull(need(i));
        else if (a == "--selectivity")
            SELECTIVITIES = sd(need(i));
        else if (a == "--ef-values")
            EF_VALUES = ssz(need(i));
        else if (a == "--bridge-budget-ratios")
            BBR_VALUES = sd(need(i));
        else if (a == "--modes") {
            MODES.clear();
            std::stringstream ss(need(i));
            std::string t;
            while (std::getline(ss, t, ','))
                if (!t.empty())
                    MODES.push_back(t);
        } else if (a == "--build-threads")
            BUILD_THREADS = (std::size_t)std::stoull(need(i));
        else if (a == "--filter")
            FILTER = need(i);
        else if (a == "--conj-attrs")
            CONJ_ATTRS = (std::size_t)std::stoull(need(i));
        else if (a == "--conj-card")
            CONJ_CARD = (std::size_t)std::stoull(need(i));
        else if (a == "--conj-k")
            CONJ_KS = ssz(need(i));
        else if (a == "--column-spec")
            COLUMN_SPECS.push_back(need(i));
        else if (a == "--allowlist-file")
            ALLOWLIST_FILE = need(i);
        else if (a == "--predicate-csr")
            PRED_CSR_FILE = need(i);
        else if (a == "--predicate-terms")
            PRED_TERMS_FILE = need(i);
        else if (a == "--allowlist-label")
            ALLOWLIST_LABEL = need(i);
        else if (a == "--allowlist-spec")
            ALLOWLIST_SPECS.push_back(need(i));
        else if (a == "--predicate-column")
            PRED_COLUMN_FILE = need(i);
        else if (a == "--data-dir")
            DATA_DIR = need(i);
        else if (a == "--volume-dir")
            VOLUME_DIR = need(i);
        else if (a == "--gt-dir")
            GT_DIR = need(i);
        else if (a == "--json")
            JSON_PATH = need(i);
        else if (a == "--help" || a == "-h") {
            print_help();
            std::exit(0);
        } else {
            std::cerr << "unknown arg: " << a << " (see --help)\n";
            std::exit(1);
        }
    }
    // per-dataset dims default
    if (DATASET == "gist")
        D_VAL = (D_VAL == 128 ? 960 : D_VAL);
    else if (DATASET == "text2image")
        D_VAL = (D_VAL == 128 ? 200 : D_VAL);
    else if (DATASET == "yfcc")
        D_VAL = (D_VAL == 128 ? 192 : D_VAL);
    else if (DATASET == "laion")
        D_VAL = (D_VAL == 128 ? 512 : D_VAL);
}
