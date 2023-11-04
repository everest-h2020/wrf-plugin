/// Implements the tensor type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/index.h"
#include "fxx/bits/memref.h"
#include "fxx/bits/memref_iface.h"
#include "fxx/bits/strided_layout.h"
#include "fxx/bits/tensor_storage.h"

#include <stdexcept>

namespace fxx {

/// Implements an owning, allocated indexed family.
///
/// @tparam T           The scalar type.
/// @tparam Order       The order of the tensor (number of dimensions).
/// @tparam Allocator   The allocator type.
///
/// @pre    `Order >= 0`
/// @pre    @p Allocator must use bare pointers.
template<class T, index_t Order, class Allocator = std::allocator<T>>
struct tensor : tensor_storage<T, Order, Allocator>,
                memref_iface<tensor<T, Order, Allocator>, T, Order> {
    using storage_type = tensor_storage<T, Order, Allocator>;
    using value_type = typename storage_type::value_type;
    using allocator_type = typename storage_type::allocator_type;
    using memref_type = memref<value_type, Order>;
    using const_memref_type = memref<const value_type, Order>;
    using layout_type = strided_layout<Order>;

    //===------------------------------------------------------------------===//
    // Constructors
    //===------------------------------------------------------------------===//

    /*implicit*/ constexpr tensor() noexcept(noexcept(allocator_type()))
            : storage_type(),
              m_layout()
    {}

    explicit constexpr tensor(const allocator_type &alloc) noexcept
            : storage_type(alloc),
              m_layout()
    {}

    explicit constexpr tensor(
        const allocator_type &alloc,
        std::convertible_to<
            index_t> auto... sizes) requires(sizeof...(sizes) == Order)
            : tensor(*layout_type::innermost(sizes...), alloc)
    {}

    explicit constexpr tensor(
        std::convertible_to<
            index_t> auto... sizes) requires(sizeof...(sizes) == Order)
            : tensor(*layout_type::innermost(sizes...))
    {}

    /*implicit*/ constexpr tensor(const tensor &copy)
            : storage_type(copy),
              m_layout(copy.layout())
    {}

    constexpr tensor &operator=(const tensor &copy)
    {
        *this = static_cast<const storage_type &>(copy);
        m_layout = copy.layout();
        return *this;
    }

    /*implicit*/ constexpr tensor(tensor &&move) noexcept
            : storage_type(std::move(move)),
              m_layout(std::exchange(move.m_layout, layout_type{}))
    {}

    constexpr tensor &operator=(tensor &&move)
    {
        *this = static_cast<storage_type &&>(move);
        m_layout = std::exchange(move.m_layout, layout_type{});
        return *this;
    }

    friend constexpr void swap(tensor &lhs, tensor &rhs) noexcept
    {
        using std::swap;

        swap(
            static_cast<storage_type &>(lhs),
            static_cast<storage_type &>(rhs));
        swap(lhs.m_layout, rhs.m_layout);
    }

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the layout descriptor.
    [[nodiscard]] constexpr const layout_type &layout() const noexcept
    {
        return m_layout;
    }

    //===------------------------------------------------------------------===//
    // Implicit conversion
    //===------------------------------------------------------------------===//
    //
    // We allow implicitly obtaining a non-owning reference (memref).

    /*implicit*/ constexpr operator memref_type() noexcept
    {
        return memref_type(this->data(), this->data(), 0, layout());
    }
    /*implicit*/ constexpr operator const_memref_type() const noexcept
    {
        return const_memref_type(this->data(), this->data(), 0, layout());
    }

private:
    [[no_unique_address]] layout_type m_layout;
};

} // namespace fxx
