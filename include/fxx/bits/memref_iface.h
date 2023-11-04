/// Implements the memref_iface mixin.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/hrect.h"
#include "fxx/bits/index.h"
#include "fxx/bits/index_tuple.h"
#include "fxx/bits/strided_layout.h"

#include <type_traits>

namespace fxx {

namespace detail {

template<class T, index_t Order, bool Reverse>
struct memref_iterator : layout_iterator<Order, Reverse> {
    using base = layout_iterator<Order, Reverse>;

    using value_type = std::remove_cv_t<T>;
    using difference_type = typename base::difference_type;
    using element_type = T;
    using pointer = element_type*;
    using reference = element_type &;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /*implicit*/ constexpr memref_iterator() noexcept : base(), m_data() {}

    explicit constexpr memref_iterator(pointer data, base it) noexcept
            : base(it),
              m_data(data)
    {}

    /*implicit*/ constexpr memref_iterator(const memref_iterator &) noexcept =
        default;

    constexpr memref_iterator &
    operator=(const memref_iterator &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the pointer to the data segment.
    [[nodiscard]] constexpr pointer data() const noexcept { return m_data; }

    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr bool
    operator==(const memref_iterator &rhs) const noexcept
    {
        return data() == rhs.data() && this->index() == rhs.index();
    }

    [[nodiscard]] constexpr std::partial_ordering
    operator<=>(const memref_iterator &rhs) const noexcept
    {
        if (data() != rhs.data()) return std::partial_ordering::unordered;
        return this->index() <=> rhs.index();
    }

    //===------------------------------------------------------------------===//
    // Iterator
    //===------------------------------------------------------------------===//

    [[nodiscard]] constexpr reference operator*() const noexcept
    {
        return data()[this->offset()];
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept
    {
        return data() + this->offset();
    }

    constexpr memref_iterator &
    operator+=(const difference_type &rhs) const noexcept
    {
        this->index() += rhs;
        return *this;
    }

    [[nodiscard]] constexpr memref_iterator
    operator+(const difference_type &rhs) const noexcept
    {
        return memref_iterator(*this) += rhs;
    }

    [[nodiscard]] friend constexpr memref_iterator
    operator+(const difference_type &lhs, memref_iterator rhs) noexcept
    {
        return rhs += lhs;
    }

    constexpr memref_iterator &
    operator-=(const difference_type &rhs) const noexcept
    {
        this->index() -= rhs;
        return *this;
    }

    [[nodiscard]] constexpr memref_iterator
    operator-(const difference_type &rhs) const noexcept
    {
        return memref_iterator(*this) -= rhs;
    }

    [[nodiscard]] constexpr difference_type
    operator-(const memref_iterator &rhs) const noexcept
    {
        return this->index() - rhs.index();
    }

    [[nodiscard]] constexpr value_type
    operator[](const difference_type &diff) const noexcept
    {
        return *(this + diff);
    }

private:
    pointer m_data;
};

} // namespace detail

/// Implements the memref interface as a mixin for @p Derived.
///
/// @tparam Derived     The derived type.
/// @tparam T           The element type.
/// @tparam Order       The order of the memref (number of dimensions).
///
/// @pre    `Order >= 0`
template<class Derived, class T, index_t Order>
struct memref_iface {
    static_assert(Order >= 0, "Order must be non-negative.");

    /// The stored value type.
    using value_type = std::remove_cv_t<T>;
    /// The accessible element type.
    using element_type = T;
    /// The pointer-to-element type.
    using pointer = element_type*;
    /// The reference-to-element type.
    using reference = element_type &;

    /// The applicable strided_layout type.
    using layout_type = strided_layout<Order>;
    /// The applicable index_tuple type.
    using index_type = typename layout_type::index_type;

    /// The element iterator.
    using iterator = detail::memref_iterator<T, Order, false>;
    /// The reversed element iterator.
    using reverse_iterator = detail::memref_iterator<T, Order, true>;

    //===------------------------------------------------------------------===//
    // Iteration
    //===------------------------------------------------------------------===//
    //
    // Memref allows bidirectional iteration over their stored elements.

    /// Obtains the range start iterator.
    [[nodiscard]] constexpr iterator begin() const noexcept
    {
        return iterator(_data(), _layout().begin());
    }

    /// Obtains the range end iterator.
    [[nodiscard]] constexpr iterator end() const noexcept
    {
        return iterator(_data(), _layout().end());
    }

    /// Obtains the reversed range begin iterator.
    [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept
    {
        return reverse_iterator(_data(), _layout().rbegin());
    }

    /// Obtains the reversed range end iterator.
    [[nodiscard]] constexpr reverse_iterator rend() const noexcept
    {
        return reverse_iterator(_data(), _layout().rend());
    }

    //===------------------------------------------------------------------===//
    // Indexing
    //===------------------------------------------------------------------===//
    //
    // Memref allows accessing elements by index.

    /// Calculates the pointer to @p index .
    ///
    /// @retval std::nullopt    @p index is out of range.
    /// @retval pointer         The pointer to @p index .
    [[nodiscard]] constexpr std::optional<pointer>
    offset(const index_type &index) const noexcept
    {
        if (const auto off = _layout().offset(index)) return *off;
        return std::nullopt;
    }

    /// Obtains a reference to @p index without bounds checking.
    ///
    /// @pre    `layout().hrect().contains(index)`
    [[nodiscard]] constexpr reference
    operator[](const index_type &index) const noexcept
    {
        return _data()[_layout()[index]];
    }

    /// @copydoc operator[](const index_type &)
    [[nodiscard]] constexpr reference
    operator()(std::convertible_to<index_t> auto... index) const noexcept
        requires(sizeof...(index) == Order)
    {
        return operator[](index_type(index...));
    }

private:
    [[nodiscard]] constexpr const Derived &_self() const noexcept
    {
        return static_cast<const Derived &>(*this);
    }

    [[nodiscard]] constexpr const pointer _data() const noexcept
    {
        return _self().data();
    }

    [[nodiscard]] constexpr const layout_type &_layout() const noexcept
    {
        return _self().layout();
    }
};

} // namespace fxx
