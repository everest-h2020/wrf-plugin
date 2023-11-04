/// Implements the index_tuple type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/index.h"

#include <array>
#include <compare>
#include <cstddef>
#include <optional>
#include <ostream>
#include <tuple>

namespace fxx {

/// Holds a statically sized tuple of index_t values.
///
/// The index_tuple type is a literal type that also implements the std::tuple
/// like interface (i.e., supports structured binding). It is implemented using
/// an underlying std::array, and extends it with comparison and arithmetic.
///
/// @tparam Size    Size of the tuple.
///
/// @pre    `Size >= 0`
template<index_t Size>
struct index_tuple : std::array<index_t, Size> {
    static_assert(Size >= 0, "Size must be non-negative.");

    /// The underlying storage type.
    using base = std::array<index_t, Size>;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /// Initializes the 0-valued index tuple.
    /*implicit*/ constexpr index_tuple() noexcept : base() {}

    /// Initializes an index tuple with @p values .
    /*implicit*/ constexpr index_tuple(const base &values) noexcept
            : base(values)
    {}

    /// @copydoc index_tuple(const base &)
    explicit constexpr index_tuple(
        std::convertible_to<index_t> auto... values) noexcept
        requires(sizeof...(values) == Size)
            : base({static_cast<index_t>(values)...})
    {}

    /*implicit*/ constexpr index_tuple(const index_tuple &) noexcept = default;

    constexpr index_tuple &operator=(const index_tuple &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//
    //
    // Index tuples are std::tuple-like types.

    /// Obtains an immutable reference to the @p I th component of @p tuple .
    ///
    /// @pre    `I < Size`
    template<std::size_t I>
    [[nodiscard]] friend constexpr const index_t &
    get(const index_tuple &tuple) noexcept
    {
        static_assert(I < Size, "I is out of range.");
        return tuple[I];
    }

    /// Obtains a reference to the @p I th component of @p tuple .
    ///
    /// @pre    `I < Size`
    template<std::size_t I>
    [[nodiscard]] friend constexpr index_t &get(index_tuple &tuple) noexcept
    {
        static_assert(I < Size, "I is out of range.");
        return tuple[I];
    }

    /// Prints @p indices to @p os .
    friend std::ostream &
    operator<<(std::ostream &os, const index_tuple &indices)
    {
        os << "(";
        auto it = indices.begin();
        if (it != indices.end()) {
            os << *it;
            while (++it != indices.end()) os << ", " << *it;
        }
        os << ")";
        return os;
    }

    //===------------------------------------------------------------------===//
    // Comparison
    //===------------------------------------------------------------------===//
    //
    // Index tuples are strongly ordered.

    [[nodiscard]] constexpr bool
    operator==(const index_tuple &) const noexcept = default;

    [[nodiscard]] constexpr std::strong_ordering
    operator<=>(const index_tuple &rhs) const noexcept
    {
        for (auto [l, r] = std::tuple(this->begin(), rhs.begin());
             l != this->end();
             ++l, ++r) {
            const auto comp = *l <=> *r;
            if (!std::is_eq(comp)) return comp;
        }
        return std::strong_ordering::equal;
    }

    //===------------------------------------------------------------------===//
    // Arithmetic
    //===------------------------------------------------------------------===//
    //
    // Index tuples can be added and subtracted, which defaults to wrapping
    // (unchecked) arithmetic.

    constexpr index_tuple &operator+=(const index_tuple &rhs) noexcept
    {
        for (auto [l, r] = std::tuple(this->begin(), rhs.begin());
             l != this->end();
             ++l, ++r)
            *l += *r;
        return *this;
    }

    [[nodiscard]] constexpr index_tuple
    operator+(const index_tuple &rhs) noexcept
    {
        return index_tuple(*this) += rhs;
    }

    constexpr index_tuple &operator-=(const index_tuple &rhs) noexcept
    {
        for (auto [l, r] = std::tuple(this->begin(), rhs.begin());
             l != this->end();
             ++l, ++r)
            *l -= *r;
        return *this;
    }

    [[nodiscard]] constexpr index_tuple
    operator-(const index_tuple &rhs) noexcept
    {
        return index_tuple(*this) -= rhs;
    }

    /// Performs checked addition of @p lhs and @p rhs .
    ///
    /// @retval std::nullopt    Result is out of range.
    /// @retval index_t         `lhs + rhs`
    [[nodiscard]] friend constexpr std::optional<index_tuple>
    checked_add(const index_tuple &lhs, const index_tuple &rhs)
    {
        index_tuple<Size> result;
        for (auto [l, r, o] =
                 std::tuple(lhs.begin(), rhs.begin(), result.begin());
             o != result.end();
             ++l, ++r, ++o) {
            const auto part = checked_add(*l, *r);
            if (!part) return std::nullopt;
            *o = *part;
        }
        return result;
    }

    /// Performs checked subtraction of @p lhs and @p rhs .
    ///
    /// @retval std::nullopt    Result is out of range.
    /// @retval index_t         `lhs - rhs`
    [[nodiscard]] friend constexpr std::optional<index_tuple>
    checked_sub(const index_tuple &lhs, const index_tuple &rhs)
    {
        index_tuple<Size> result;
        for (auto [l, r, o] =
                 std::tuple(lhs.begin(), rhs.begin(), result.begin());
             o != result.end();
             ++l, ++r, ++o) {
            const auto part = checked_sub(*l, *r);
            if (!part) return std::nullopt;
            *o = *part;
        }
        return result;
    }
};

} // namespace fxx

namespace std {

template<fxx::index_t Size>
struct tuple_size<fxx::index_tuple<Size>>
        : std::integral_constant<std::size_t, Size> {};

template<std::size_t I, fxx::index_t Size>
struct tuple_element<I, fxx::index_tuple<Size>>
        : std::type_identity<fxx::index_t> {};

} // namespace std
