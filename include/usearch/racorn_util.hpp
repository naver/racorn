// RACORN
// Copyright (c) 2026-present NAVER Corp.
// Apache-2.0

/**
 *  @file   racorn_util.hpp
 *  @brief  Namespace-scope utilities used only by the RACORN-1 search path.
 *
 *  Included at namespace unum::usearch scope in index.hpp, before index_gt.
 *  Standalone header — safe to include from any translation unit.
 *
 *  Contents:
 *    - gen_hash_set_gt           fixed-capacity hash-set with O(1) generation clear
 *    - stride_downsample_inplace in-place stride down-sampling
 */

#ifndef USEARCH_RACORN_UTIL_HPP
#define USEARCH_RACORN_UTIL_HPP

/**
 *  @brief  Fixed-capacity hash-set with O(1) generation-counter clear.
 *
 *  Designed for use cases that call `clear()` very frequently within a single query
 *  (e.g., per-cycle dedup sets). Instead of memset-ing the slot array on each clear,
 *  it bumps a per-instance generation counter; slots whose stored generation does not
 *  match the current one are treated as empty. Wraparound triggers a one-time memset
 *  of the generation array to keep correctness.
 */
template <typename element_at, typename hasher_at = hash_gt<element_at>, typename allocator_at = std::allocator<byte_t>>
class gen_hash_set_gt {

    using element_t = element_at;
    using hasher_t = hasher_at;

    using allocator_t = allocator_at;
    using byte_t = typename allocator_t::value_type;
    using gen_t = std::uint32_t;
    static_assert(sizeof(byte_t) == 1, "Allocator must allocate separate addressable bytes");

    element_t* slots_{};
    gen_t* gens_{};
    gen_t current_gen_{0};
    std::size_t capacity_{};
    std::size_t count_{};
    hasher_t hasher_{};

  public:
    gen_hash_set_gt() noexcept {}
    ~gen_hash_set_gt() noexcept { reset(); }

    explicit operator bool() const noexcept { return slots_; }
    std::size_t size() const noexcept { return count_; }

    /// @brief O(1) clear: only bumps the generation counter (no memset).
    void clear() noexcept {
        ++current_gen_;
        count_ = 0;
        if (current_gen_ == 0 && gens_) {
            // Generation overflow: full reset to keep correctness.
            std::memset(gens_, 0, capacity_ * sizeof(gen_t));
            current_gen_ = 1;
        }
    }

    void reset() noexcept {
        if (slots_)
            allocator_t{}.deallocate((byte_t*)slots_, capacity_ * sizeof(element_t));
        if (gens_)
            allocator_t{}.deallocate((byte_t*)gens_, capacity_ * sizeof(gen_t));
        slots_ = nullptr;
        gens_ = nullptr;
        capacity_ = 0;
        count_ = 0;
        current_gen_ = 0;
    }

    gen_hash_set_gt(std::size_t capacity) noexcept {
        if (capacity == 0)
            return;
        std::size_t cap = ceil2(capacity);
        slots_ = (element_t*)allocator_t{}.allocate(cap * sizeof(element_t));
        if (!slots_)
            return;
        gens_ = (gen_t*)allocator_t{}.allocate(cap * sizeof(gen_t));
        if (!gens_) {
            allocator_t{}.deallocate((byte_t*)slots_, cap * sizeof(element_t));
            slots_ = nullptr;
            return;
        }
        capacity_ = cap;
        std::memset(gens_, 0, cap * sizeof(gen_t));
        current_gen_ = 1; // 0 reserved for "uninitialized slot"
    }

    gen_hash_set_gt(gen_hash_set_gt&& other) noexcept {
        slots_ = exchange(other.slots_, nullptr);
        gens_ = exchange(other.gens_, nullptr);
        capacity_ = exchange(other.capacity_, 0);
        count_ = exchange(other.count_, 0);
        current_gen_ = exchange(other.current_gen_, 0);
    }

    gen_hash_set_gt& operator=(gen_hash_set_gt&& other) noexcept {
        std::swap(slots_, other.slots_);
        std::swap(gens_, other.gens_);
        std::swap(capacity_, other.capacity_);
        std::swap(count_, other.count_);
        std::swap(current_gen_, other.current_gen_);
        return *this;
    }

    gen_hash_set_gt(gen_hash_set_gt const&) = delete;
    gen_hash_set_gt& operator=(gen_hash_set_gt const&) = delete;

    /// @brief  Checks if the element is present in the current generation.
    inline bool test(element_t const& elem) const noexcept {
        std::size_t index = hasher_(elem) & (capacity_ - 1);
        while (gens_[index] == current_gen_) {
            if (slots_[index] == elem)
                return true;
            index = (index + 1) & (capacity_ - 1);
        }
        return false;
    }

    /// @brief  Inserts an element into the hash-set for the current generation.
    /// @return `true` if it was already present in this generation, else `false`.
    inline bool set(element_t const& elem) noexcept {
        std::size_t index = hasher_(elem) & (capacity_ - 1);
        while (gens_[index] == current_gen_) {
            if (slots_[index] == elem)
                return true;
            index = (index + 1) & (capacity_ - 1);
        }
        slots_[index] = elem;
        gens_[index] = current_gen_;
        ++count_;
        return false;
    }
};

/**
 *  @brief  In-place stride down-sampling: write `data[i] := data[i * stride]`
 *          for `i ∈ [0, limit)`, where `stride = total / limit` (integer div).
 *
 *  Used by RACORN-1 to spread a budget-bounded selection across a larger
 *  candidate pool instead of taking only the head. Two call sites:
 *    - 2-hop passed candidates in `acorn_expand_two_hop_`
 *    - 2-hop failed candidates (bridge pool) in `racorn_fallback_`
 *
 *  Safety: `i * stride >= i` always (since `stride >= 1`), so each read
 *  position is at or ahead of the write position — no read-after-write
 *  hazard within the loop.
 *
 *  No-ops when `limit == 0` or `limit >= total` (latter implies `stride == 1`,
 *  i.e. nothing to sub-sample).
 */
template <typename element_at>
inline void stride_downsample_inplace(element_at* data, std::size_t total, std::size_t limit) noexcept {
    if (limit == 0 || limit >= total)
        return;
    std::size_t const stride = total / limit;
    for (std::size_t i = 0; i < limit; ++i)
        data[i] = data[i * stride];
}

#endif // USEARCH_RACORN_UTIL_HPP
