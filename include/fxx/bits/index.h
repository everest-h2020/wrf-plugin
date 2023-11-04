/// Declares the index_t type and related functions.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include <cstdint>
#include <optional>

namespace fxx {

/// Type that stores an index.
using index_t = std::int64_t;

/// Performs checked addition of @p lhs and @p rhs .
///
/// @retval std::nullopt    Result is out of range.
/// @retval index_t         `lhs + rhs`
[[nodiscard]] constexpr std::optional<index_t>
checked_add(index_t lhs, index_t rhs) noexcept
{
    index_t result;
    if (__builtin_add_overflow(lhs, rhs, &result)) return std::nullopt;
    return result;
}

/// Performs checked subtraction of @p lhs and @p rhs .
///
/// @retval std::nullopt    Result is out of range.
/// @retval index_t         `lhs - rhs`
[[nodiscard]] constexpr std::optional<index_t>
checked_sub(index_t lhs, index_t rhs) noexcept
{
    index_t result;
    if (__builtin_sub_overflow(lhs, rhs, &result)) return std::nullopt;
    return result;
}

/// Performs checked multiplication of @p lhs and @p rhs .
///
/// @retval std::nullopt    Result is out of range.
/// @retval index_t         `lhs * rhs`
[[nodiscard]] constexpr std::optional<index_t>
checked_mul(index_t lhs, index_t rhs) noexcept
{
    index_t result;
    if (__builtin_mul_overflow(lhs, rhs, &result)) return std::nullopt;
    return result;
}

} // namespace fxx
