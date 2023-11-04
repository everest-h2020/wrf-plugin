/// Implements the hrect type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/index.h"
#include "fxx/bits/index_tuple.h"

#include <algorithm>
#include <compare>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>

namespace fxx {

namespace detail {

template<index_t Order, bool Reverse>
struct hrect_iterator {
    using value_type = index_tuple<Order>;
    using difference_type = value_type;
    using pointer = value_type;
    using reference = value_type;
    using iterator_category = std::random_access_iterator_tag;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /*implicit*/ constexpr hrect_iterator() noexcept : m_limit(), m_index() {}

    explicit hrect_iterator(value_type limit) : m_limit(limit), m_index()
    {
        if constexpr (Reverse)
            for (index_t dim = 0; dim < Order; ++dim)
                m_index[dim] = m_limit[dim] - 1;
    }

    explicit hrect_iterator(value_type limit, std::nullptr_t)
            : m_limit(limit),
              m_index()
    {
        const auto is_empty =
            std::any_of(limit.begin(), limit.end(), [](index_t size) {
                return size <= 0;
            });

        if constexpr (Reverse)
            if (is_empty)
                for (index_t dim = 0; dim < Order; ++dim)
                    m_index[dim] = m_limit[dim] - 1;
            else
                move_predecessor();
        else if (!is_empty)
            m_index.front() = m_limit.front();
    }

    explicit hrect_iterator(value_type limit, value_type index)
            : m_limit(limit),
              m_index(index)
    {}

    /*implicit*/ constexpr hrect_iterator(const hrect_iterator &) noexcept =
        default;

    constexpr hrect_iterator &
    operator=(const hrect_iterator &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Obtains the iteration domain limits.
    [[nodiscard]] constexpr const value_type &limit() const noexcept
    {
        return m_limit;
    }

    /// Obtains the current immutable iteration index.
    [[nodiscard]] constexpr const value_type &index() const noexcept
    {
        return m_index;
    }
    /// Obtains the current iteration index.
    [[nodiscard]] constexpr value_type &index() noexcept { return m_index; }

    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr bool
    operator==(const hrect_iterator &rhs) const noexcept
    {
        return index() == rhs.index();
    }

    [[nodiscard]] constexpr std::strong_ordering
    operator<=>(const hrect_iterator &rhs) const noexcept
    {
        return index() <=> rhs.index();
    }

    //===------------------------------------------------------------------===//
    // Iterator
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr value_type operator*() const noexcept
    {
        return index();
    }

    constexpr hrect_iterator &operator++() noexcept
    {
        if constexpr (Reverse)
            move_predecessor();
        else
            move_successor();

        return *this;
    }

    constexpr hrect_iterator &operator--() noexcept
    {
        if constexpr (Reverse)
            move_successor();
        else
            move_predecessor();

        return *this;
    }

    constexpr hrect_iterator &operator+=(const value_type &rhs) const noexcept
    {
        index() += rhs;
        return *this;
    }

    [[nodiscard]] constexpr hrect_iterator
    operator+(const difference_type &rhs) const noexcept
    {
        return hrect_iterator(*this) += rhs;
    }

    [[nodiscard]] friend constexpr hrect_iterator
    operator+(const difference_type &lhs, hrect_iterator rhs) noexcept
    {
        return rhs += lhs;
    }

    constexpr hrect_iterator &
    operator-=(const difference_type &rhs) const noexcept
    {
        index() -= rhs;
        return *this;
    }

    [[nodiscard]] constexpr hrect_iterator
    operator-(const difference_type &rhs) const noexcept
    {
        return hrect_iterator(*this) -= rhs;
    }

    [[nodiscard]] constexpr difference_type
    operator-(const hrect_iterator &rhs) const noexcept
    {
        return index() - rhs.index();
    }

    [[nodiscard]] constexpr difference_type
    operator[](const difference_type &diff) const noexcept
    {
        return index() + diff;
    }

private:
    void move_successor()
    {
        for (index_t dim = Order - 1; dim >= 0; --dim) {
            if (++index()[dim] < limit()[dim]) return;
            index()[dim] = 0;
        }

        index().front() = limit().front();
    }

    void move_predecessor()
    {
        for (index_t dim = Order - 1; dim >= 0; --dim) {
            if (--index()[dim] >= 0) return;
            index()[dim] = limit()[dim] - 1;
        }

        index().front() = -1;
    }

    value_type m_limit;
    value_type m_index;
};

template<bool Reverse>
struct hrect_iterator<0, Reverse> {
    using value_type = index_tuple<0>;
    using difference_type = value_type;
    using pointer = value_type;
    using reference = value_type;
    using iterator_category = std::bidirectional_iterator_tag;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /*implicit*/ constexpr hrect_iterator() noexcept : m_sentinel(1) {}

    explicit hrect_iterator(value_type) : m_sentinel(0) {}

    explicit hrect_iterator(value_type, std::nullptr_t)
            : m_sentinel(Reverse ? -1 : 1)
    {}

    /*implicit*/ constexpr hrect_iterator(const hrect_iterator &) noexcept =
        default;

    constexpr hrect_iterator &
    operator=(const hrect_iterator &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Determines whether this iterator is a sentinel.
    [[nodiscard]] constexpr bool is_sentinel() const noexcept
    {
        return m_sentinel != 0;
    }

    /// Obtains the 0 tuple.
    [[nodiscard]] constexpr value_type limit() const noexcept { return {}; }

    /// Obtains the 0 index.
    [[nodiscard]] constexpr value_type index() const noexcept { return {}; }

    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr bool
    operator==(const hrect_iterator &rhs) const noexcept
    {
        return is_sentinel() == rhs.is_sentinel();
    }

    [[nodiscard]] constexpr std::strong_ordering
    operator<=>(const hrect_iterator &rhs) const noexcept
    {
        return is_sentinel() <=> rhs.is_sentinel();
    }

    //===------------------------------------------------------------------===//
    // Iterator
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr value_type operator*() const noexcept { return {}; }

    constexpr hrect_iterator &operator++() noexcept
    {
        if constexpr (Reverse)
            --m_sentinel;
        else
            ++m_sentinel;
        return *this;
    }

    constexpr hrect_iterator &operator--() noexcept
    {
        if constexpr (Reverse)
            ++m_sentinel;
        else
            --m_sentinel;
        return *this;
    }

private:
    int m_sentinel;
};

} // namespace detail

/// Defines a hyperrectangle of statically known order.
///
/// @tparam Order   The order of the hyperrectangle (number of dimensions).
///
/// @pre    `Order >= 0`
template<index_t Order>
struct hrect {
    static_assert(Order >= 0, "Order must be non-negative.");

    /// The index type used with this hyperrectangle.
    using index_type = index_tuple<Order>;

    /// The index iterator type.
    using iterator = detail::hrect_iterator<Order, false>;
    /// The reversed index iterator type.
    using reverse_iterator = detail::hrect_iterator<Order, true>;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /// Initializes the empty hyperrectangle.
    /*implicit*/ constexpr hrect() noexcept : m_sizes() {}

    /// Initializes a hyperrectangle with @p sizes .
    /*implicit*/ constexpr hrect(index_type sizes) noexcept : m_sizes(sizes) {}

    /// @copydoc hrect(index_type)
    explicit constexpr hrect(
        std::convertible_to<index_t> auto... values) noexcept
        requires(sizeof...(values) == Order)
            : m_sizes(values...)
    {}

    /*implicit*/ constexpr hrect(const hrect &) noexcept = default;

    constexpr hrect &operator=(const hrect &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the order of this hyperrectangle.
    [[nodiscard]] constexpr index_t order() const noexcept { return Order; }

    /// Gets the immutable sizes.
    [[nodiscard]] constexpr const index_type &sizes() const noexcept
    {
        return m_sizes;
    }
    /// Gets the sizes.
    [[nodiscard]] constexpr index_type &sizes() noexcept { return m_sizes; }

    /// Determines whether this hyperrectangle is empty.
    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return std::any_of(
            sizes().begin(),
            sizes().end(),
            [](index_t size) { return size <= 0; });
    }

    /// Calculates the volume of this hyperrectangle.
    [[nodiscard]] constexpr std::optional<std::uint64_t> volume() const noexcept
    {
        std::uint64_t result = 1;
        for (const auto &sz : sizes()) {
            if (sz < 0) return std::nullopt;
            if (__builtin_mul_overflow(
                    result,
                    static_cast<std::uint64_t>(sz),
                    &result))
                return std::nullopt;
        }
        return result;
    }

    /// Gets the lexicographically least index.
    [[nodiscard]] constexpr index_type min() const noexcept { return {}; }

    /// Gets the lexicographically greatest index.
    [[nodiscard]] constexpr index_type max() const noexcept
    {
        return (--end()).index();
    }

    /// Determines whether @p index is contained in the hyperrectangle.
    [[nodiscard]] constexpr bool
    contains(const index_type &index) const noexcept
    {
        return index >= min() && index <= max();
    }

    /// Prints @p hrect to @p os .
    friend std::ostream &operator<<(std::ostream &os, const hrect &hrect)
    {
        return os << "hrect" << hrect.sizes();
    }

    //===------------------------------------------------------------------===//
    // Iteration
    //===------------------------------------------------------------------===//
    //
    // Hyperrectangles allow bidirectional iteration over their indices.

    /// Obtains the range start iterator.
    [[nodiscard]] constexpr iterator begin() const noexcept
    {
        return iterator(sizes());
    }
    /// Obtains the range end iterator.
    [[nodiscard]] constexpr iterator end() const noexcept
    {
        return iterator(sizes(), nullptr);
    }

    /// Obtains the reversed range start iterator.
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept
    {
        return reverse_iterator(sizes());
    }

    /// Obtains the reversed range end iterator.
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept
    {
        return reverse_iterator(sizes(), nullptr);
    }

    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//
    //
    // Hyperrectangles are equality-comparable.

    [[nodiscard]] constexpr bool operator==(const hrect &rhs) const noexcept
    {
        return sizes() == rhs.sizes();
    }

private:
    // NOTE: For Order == 0, index_type is an empty type that should not incur
    //       any additional storage (as shouldn't hrect). We are in C++20, so
    //       let's use the attribute instead of EBO.
    [[no_unique_address]] index_type m_sizes;
};

} // namespace fxx
