/// Implements the strided_layout type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/hrect.h"
#include "fxx/bits/index.h"
#include "fxx/bits/index_tuple.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <type_traits>

namespace fxx {

namespace detail {

template<index_t Order, bool Reverse>
struct layout_iterator : hrect_iterator<Order, Reverse> {
    using base = hrect_iterator<Order, Reverse>;

    using index_type = index_tuple<Order>;
    using difference_type = typename base::difference_type;
    using value_type = std::size_t;
    using pointer = value_type;
    using reference = value_type;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /*implicit*/ constexpr layout_iterator() noexcept : base(), m_strides() {}

    explicit layout_iterator(base it, const index_type &strides) noexcept
            : base(it),
              m_strides(strides)
    {}

    /*implicit*/ constexpr layout_iterator(const layout_iterator &) noexcept =
        default;

    constexpr layout_iterator &
    operator=(const layout_iterator &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Obtains the stride vector.
    [[nodiscard]] constexpr const index_type &strides() const noexcept
    {
        return m_strides;
    }

    /// Obtains the current offset.
    [[nodiscard]] constexpr value_type offset() const noexcept
    {
        return std::inner_product(
            strides().begin(),
            strides().end(),
            this->index().begin(),
            std::size_t{0});
    }

    //===------------------------------------------------------------------===//
    // Iterator
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr value_type operator*() const noexcept
    {
        return offset();
    }

    constexpr layout_iterator &
    operator+=(const difference_type &rhs) const noexcept
    {
        this->index() += rhs;
        return *this;
    }

    [[nodiscard]] constexpr layout_iterator
    operator+(const difference_type &rhs) const noexcept
    {
        return layout_iterator(*this) += rhs;
    }

    [[nodiscard]] friend constexpr layout_iterator
    operator+(const difference_type &lhs, layout_iterator rhs) noexcept
    {
        return rhs += lhs;
    }

    constexpr layout_iterator &
    operator-=(const difference_type &rhs) const noexcept
    {
        this->index() -= rhs;
        return *this;
    }

    [[nodiscard]] constexpr layout_iterator
    operator-(const difference_type &rhs) const noexcept
    {
        return layout_iterator(*this) -= rhs;
    }

    [[nodiscard]] constexpr difference_type
    operator-(const layout_iterator &rhs) const noexcept
    {
        return this->index() - rhs.index();
    }

    [[nodiscard]] constexpr value_type
    operator[](const difference_type &diff) const noexcept
    {
        return *(this + diff);
    }

private:
    index_type m_strides;
};

} // namespace detail

/// Defines a strided hyperrectangular memory layout.
///
/// @tparam Order   The order of the layout (number of dimensions).
///
/// @pre    `Order >= 0`
template<index_t Order>
struct strided_layout {
    static_assert(Order >= 0, "Order must be non-negative.");

    /// The type that defines the hyperrectangle.
    using hrect_type = hrect<Order>;
    /// The type that stores an index into the hyperrectangle.
    using index_type = typename hrect_type::index_type;

    /// The offset iterator type.
    using iterator = detail::layout_iterator<Order, false>;
    /// The reversed offset iterator type.
    using reverse_iterator = detail::layout_iterator<Order, true>;

    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//
    //
    // We support construction of the two most common standard layouts:
    // innermost ("row-major") and outermost ("column-major").

    /// Constructs an innermost layout for @p sizes .
    ///
    /// @retval std::nullopt    Hyperrectangle is too big.
    /// @retval strided_layout  The strided_layout.
    [[nodiscard]] static constexpr std::optional<strided_layout>
    innermost(const index_type &sizes)
    {
        index_type strides;
        if (!make_strides(strides.rbegin(), sizes.rbegin()))
            return std::nullopt;
        return strided_layout(sizes, strides);
    }

    /// @copydoc innermost(const index_type &)
    [[nodiscard]] static constexpr std::optional<strided_layout>
    innermost(std::convertible_to<index_t> auto... sizes) requires(
        sizeof...(sizes) == Order)
    {
        return innermost(index_type(sizes...));
    }

    /// Constructs an innermost layout for @p hrect .
    ///
    /// @retval std::nullopt    @p hrect is too big.
    /// @retval strided_layout  The strided_layout.
    [[nodiscard]] static constexpr std::optional<strided_layout>
    innermost(const hrect_type &hrect)
    {
        return innermost(hrect.sizes());
    }

    /// Constructs an outermost layout for @p sizes .
    ///
    /// @retval std::nullopt    Hyperrectangle is too big.
    /// @retval strided_layout  The strided_layout.
    [[nodiscard]] static constexpr std::optional<strided_layout>
    outermost(const index_type &sizes)
    {
        index_type strides;
        if (!make_strides(strides.begin(), sizes.begin())) return std::nullopt;
        return strided_layout(sizes, strides);
    }

    /// @copydoc outermost(const index_type &)
    [[nodiscard]] static constexpr std::optional<strided_layout>
    outermost(std::convertible_to<index_t> auto... sizes) requires(
        sizeof...(sizes) == Order)
    {
        return outermost(index_type(sizes...));
    }

    /// Constructs an outermost layout for @p hrect .
    ///
    /// @retval std::nullopt    @p hrect is too big.
    /// @retval strided_layout  The strided_layout.
    [[nodiscard]] static constexpr std::optional<strided_layout>
    outermost(const hrect_type &hrect)
    {
        return outermost(hrect.sizes());
    }

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /// Initializes an empty strided layout.
    /*implicit*/ constexpr strided_layout() noexcept : m_hrect(), m_strides() {}

    /// Initializes a strided layout from @p hrect and @p strides .
    explicit constexpr strided_layout(
        const hrect_type &hrect,
        const index_type &strides) noexcept
            : m_hrect(hrect),
              m_strides(strides)
    {}

    /*implicit*/ constexpr strided_layout(const strided_layout &) noexcept =
        default;

    constexpr strided_layout &
    operator=(const strided_layout &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the immutable hyperrectangle.
    [[nodiscard]] constexpr const hrect_type &hrect() const noexcept
    {
        return m_hrect;
    }
    /// Gets the hyperrectangle.
    [[nodiscard]] constexpr hrect_type &hrect() noexcept { return m_hrect; }

    /// Gets the immutable stride vector.
    [[nodiscard]] constexpr const index_type &strides() const noexcept
    {
        return m_strides;
    }
    /// Gets the stride vector.
    [[nodiscard]] constexpr index_type &strides() noexcept { return m_strides; }

    //===------------------------------------------------------------------===//
    // Iteration
    //===------------------------------------------------------------------===//
    //
    // Strided layout allows bidirectional iteration over their offsets.

    /// Obtains the range start iterator.
    [[nodiscard]] constexpr iterator begin() const noexcept
    {
        return iterator(hrect().begin(), strides());
    }
    /// Obtains the range end iterator.
    [[nodiscard]] constexpr iterator end() const noexcept
    {
        return iterator(hrect().end(), strides());
    }

    /// Obtains the reversed range start iterator.
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept
    {
        return reverse_iterator(hrect().rbegin(), strides());
    }

    /// Obtains the reversed range end iterator.
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept
    {
        return reverse_iterator(hrect().rend(), strides());
    }

    //===------------------------------------------------------------------===//
    // Offset calculation
    //===------------------------------------------------------------------===//
    //
    // The strided layout is mainly responsible for mapping index tuples to
    // offsets.

    /// Calculates the linear offset of @p index .
    ///
    /// @retval std::nullopt    @p index is out of bounds.
    /// @retval std::size_t     The linear offset of @p index .
    [[nodiscard]] constexpr std::optional<std::size_t>
    offset(const index_type &index) const noexcept
    {
        std::size_t result = 0;
        for (auto [sz, st, i] = std::tuple(
                 hrect().sizes().begin(),
                 strides().begin(),
                 index.begin());
             i != index.end();
             ++sz, ++st, ++i) {
            if (*i < 0 || *i >= *sz) return std::nullopt;
            result += static_cast<std::size_t>(*i * *st);
        }
        return result;
    }

    /// Calculates the linear offset of @p index without bounds checking.
    ///
    /// @pre    `hrect().contains(index)`
    [[nodiscard]] constexpr std::size_t
    operator[](const index_type &index) const noexcept
    {
        assert(hrect().contains(index));
        return std::inner_product(
            strides().begin(),
            strides().end(),
            index.begin(),
            std::size_t{0});
    }

    /// @copydoc operator[](const index_type &)
    [[nodiscard]] constexpr std::size_t
    operator()(std::convertible_to<index_t> auto... index) const noexcept
        requires(sizeof...(index) == Order)
    {
        return operator[](index_type(index...));
    }

private:
    [[nodiscard]] static constexpr bool make_strides(
        std::output_iterator<index_t> auto stride_it,
        std::input_iterator auto size_it)
    {
        const auto stride_end = stride_it + Order;
        index_t accu{1};
        for (; stride_it != stride_end; ++size_it, ++stride_it) {
            if (*size_it <= 0) return false;
            const auto part = checked_mul(accu, *size_it);
            if (!part) return false;
            *stride_it = accu;
            accu = *part;
        }

        return true;
    }

    // NOTE: For Order == 0, we don't need any storage. We are in C++20, so
    //       let's use the attribute instead of EBO.
    [[no_unique_address]] hrect_type m_hrect;
    [[no_unique_address]] index_type m_strides;
};

} // namespace fxx
