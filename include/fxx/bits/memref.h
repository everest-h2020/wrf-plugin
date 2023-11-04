/// Implements the memref type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/index.h"
#include "fxx/bits/index_tuple.h"
#include "fxx/bits/memref_iface.h"
#include "fxx/bits/strided_layout.h"

namespace fxx {

/// Implements a self-describing reference to an indexed family in memory.
///
/// This type is layout-compatible to the default LLVM implementation of MLIR
/// memrefs.
///
/// @tparam T       The element type.
/// @tparam Order   The order of the memref (number of dimensions).
///
/// @pre    `Order >= 0`
template<class T, index_t Order>
struct memref : memref_iface<memref<T, Order>, T, Order> {
    static_assert(Order >= 0, "Order must be non-negative.");
    using iface = memref_iface<memref<T, Order>, T, Order>;

    /// The pointer-to-element type.
    using pointer = typename iface::pointer;
    /// The reference-to-element type.
    using reference = typename iface::reference;

    /// The applicable strided_layout type.
    using layout_type = typename iface::layout_type;
    /// The applicable index_tuple type.
    using index_type = typename iface::index_type;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /// Initializes an empty memref.
    /*implicit*/ constexpr memref() noexcept
            : m_allocated(),
              m_aligned(),
              m_offset(),
              m_layout()
    {}

    /// Initializes a memref from @p allocated , @p aligned , @p offset and
    /// @p layout .
    explicit constexpr memref(
        pointer allocated,
        pointer aligned,
        index_t offset,
        const layout_type &layout) noexcept
            : m_allocated(allocated),
              m_aligned(aligned),
              m_offset(offset),
              m_layout(layout)
    {}

    /// Initializes a memref from @p data and @p sizes with innermost layout.
    ///
    /// @pre    @p sizes must describe a valid hyperrectangle.
    explicit constexpr memref(pointer data, const index_type &sizes) noexcept
            : memref(data, data, 0, *layout_type::innermost(sizes))
    {}

    /// @copydoc memref(pointer, const index_type &)
    explicit constexpr memref(
        std::in_place_t,
        pointer data,
        std::convertible_to<index_t> auto... sizes) noexcept
        requires(sizeof...(sizes) == Order)
            : memref(data, index_type(sizes...))
    {}

    /*implicit*/ constexpr memref(const memref &) noexcept = default;

    constexpr memref &operator=(const memref &) noexcept = default;

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the immutable allocated pointer.
    [[nodiscard]] constexpr pointer allocated() const noexcept
    {
        return m_allocated;
    }
    /// Gets the allocated pointer.
    [[nodiscard]] constexpr pointer &allocated() noexcept
    {
        return m_allocated;
    }

    /// Gets the immutable aligned pointer.
    [[nodiscard]] constexpr pointer aligned() const noexcept
    {
        return m_aligned;
    }
    /// Gets the aligned pointer.
    [[nodiscard]] constexpr pointer &aligned() noexcept { return m_aligned; }

    /// Gets the immutable start offset.
    [[nodiscard]] constexpr index_t offset() const noexcept { return m_offset; }
    /// Gets the start offset.
    [[nodiscard]] constexpr index_t &offset() noexcept { return m_offset; }

    /// Gets the data pointer.
    [[nodiscard]] constexpr pointer data() const noexcept
    {
        return aligned() + offset();
    }

    /// Gets the immutable layout.
    [[nodiscard]] constexpr const layout_type &layout() const noexcept
    {
        return m_layout;
    }
    /// Gets the layout.
    [[nodiscard]] constexpr layout_type &layout() noexcept { return m_layout; }

private:
    pointer m_allocated;
    pointer m_aligned;
    index_t m_offset;

    // NOTE: The layout_type can be empty in case of a scalar. We are in C++20,
    //       so let's use the attribute instead of EBO.
    [[no_unique_address]] layout_type m_layout;
};

} // namespace fxx
