/// Implements the tensor_storage type.
///
/// @file
/// @author     Karl F. A. Friebel (karl.friebel@tu-dresden.de)

#pragma once

#include "fxx/bits/index.h"

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

namespace fxx {

/// Implements a simple owning allocator-aware storage for tensors.
///
/// @tparam T           The scalar type.
/// @tparam Order       The order of the tensor (number of dimensions).
/// @tparam Allocator   The allocator type.
///
/// @pre    `Order >= 0`
/// @pre    @p Allocator must use bare pointers.
template<class T, index_t Order, class Allocator = std::allocator<T>>
struct tensor_storage {
    static_assert(Order >= 0, "Order must be non-negative.");

    /// The scalar type.
    using value_type = std::remove_cv_t<T>;
    /// The allocator type.
    using allocator_type = typename std::allocator_traits<
        Allocator>::template rebind_alloc<value_type>;
    /// The std::allocator_traits specialization for the allocator_type.
    using allocator_traits = std::allocator_traits<allocator_type>;
    /// The mutable pointer type.
    using pointer = typename allocator_traits::pointer;
    /// The immutable pointer type.
    using const_pointer = typename allocator_traits::const_pointer;
    /// The mutable reference type.
    using reference = value_type &;
    /// The immutable reference type.
    using const_reference = const value_type &;
    /// The size type.
    using size_type = typename allocator_traits::size_type;
    /// The pointer difference type.
    using difference_type = typename allocator_traits::difference_type;

    static_assert(
        std::is_pointer_v<pointer>,
        "Allocator must use bare pointers.");

    //===------------------------------------------------------------------===//
    // Constructors & Destructor
    //===------------------------------------------------------------------===//

    /*implicit*/ constexpr tensor_storage() noexcept(noexcept(allocator_type()))
            : m_allocator(),
              m_data(),
              m_size()
    {}

    explicit constexpr tensor_storage(const allocator_type &alloc) noexcept
            : m_allocator(alloc),
              m_data(),
              m_size()
    {}

    explicit constexpr tensor_storage(
        size_type size,
        const allocator_type &alloc = {})
            : m_allocator(alloc),
              m_data(),
              m_size()
    {
        resize(size);
    }

    explicit constexpr tensor_storage(
        size_type size,
        const T &value,
        const allocator_type &alloc = {})
            : m_allocator(alloc)
    {
        resize(size, value);
    }

    /*implicit*/ constexpr tensor_storage(const tensor_storage &copy)
            : m_allocator(
                allocator_traits::select_on_container_copy_construction(
                    copy.get_allocator())),
              m_data(),
              m_size()
    {
        acquire(copy.size());

        std::copy_n(copy.data(), copy.size(), data());
    }

    constexpr tensor_storage &operator=(const tensor_storage &copy)
    {
        if constexpr (allocator_traits::propagate_on_container_copy_assignment::
                          value) {
            if (get_allocator() != copy.get_allocator()) discard();
            get_allocator() = copy.get_allocator();
        }

        if (size() == copy.size()) {
            std::copy_n(copy.data(), copy.size(), data());
            return;
        }

        discard();
        acquire(copy.size());

        for (auto [lhs, rhs] = std::tuple(data(), copy.data());
             lhs != data() + size();
             ++lhs)
            std::uninitialized_construct_using_allocator(
                lhs,
                get_allocator(),
                *rhs);
    }

    /*implicit*/ constexpr tensor_storage(tensor_storage &&move) noexcept
            : m_allocator(std::move(move.get_allocator())),
              m_data(std::exchange(move.m_data, pointer())),
              m_size(std::exchange(move.m_size, size_type()))
    {}

    constexpr tensor_storage &operator=(tensor_storage &&move) noexcept(
        allocator_traits::propagate_on_container_move_assignment::value
        || allocator_traits::is_always_equal::value)
    {
        if constexpr (allocator_traits::propagate_on_container_move_assignment::
                          value) {
            if (get_allocator() != move.get_allocator()) discard();
            // For some reason, we are supposed to copy it here.
            get_allocator() = move.get_allocator();
        }

        if (get_allocator() == move.get_allocator()) {
            using std::swap;
            discard();
            swap(m_data, move.m_data);
            swap(m_size, move.m_size);
            return *this;
        }

        if (size() == move.size()) {
            for (auto [lhs, rhs] = std::tuple(data(), move.data());
                 lhs != data() + size();
                 ++lhs)
                *lhs = std::move(*rhs);
            return *this;
        }

        discard();
        acquire(std::exchange(move.m_size, size_type()));

        for (auto [lhs, rhs] = std::tuple(data(), move.data());
             lhs != data() + size();
             ++lhs)
            std::uninitialized_construct_using_allocator(
                lhs,
                get_allocator(),
                std::move(*rhs));
        return *this;
    }

    friend constexpr void
    swap(tensor_storage &lhs, tensor_storage &rhs) noexcept
    {
        using std::swap;

        if constexpr (allocator_traits::propagate_on_container_swap::value)
            swap(lhs.get_allocator(), rhs.get_allocator());
        else {
            // The standard says this can be undefined behavior.
            assert(lhs.get_allocator() == rhs.get_allocator());
        }

        swap(lhs.m_data, rhs.m_data);
        swap(lhs.m_size, rhs.m_size);
    }

    constexpr ~tensor_storage() { discard(); }

    //===------------------------------------------------------------------===//
    // Properties
    //===------------------------------------------------------------------===//

    /// Gets the associated allocator.
    [[nodiscard]] const allocator_type &get_allocator() const noexcept
    {
        return m_allocator;
    }

    /// Gets the data pointer.
    [[nodiscard]] constexpr pointer data() const noexcept { return m_data; }

    /// Gets the number of elements.
    [[nodiscard]] constexpr size_type size() const noexcept { return m_size; }

    /// Determines whether the storage is empty.
    [[nodiscard]] constexpr bool empty() const noexcept
    {
        return size() == size_type();
    }

protected:
    //===------------------------------------------------------------------===//
    // Resizing
    //===------------------------------------------------------------------===//

    /// Resizes this storage to hold @p new_size default-constructed elements.
    constexpr void resize(size_type new_size)
    {
        discard();
        acquire(new_size);

        for (auto lhs = data(); lhs != data() + size(); ++lhs)
            std::uninitialized_construct_using_allocator(lhs, get_allocator());
    }

    /// Resizes this storage to hold @p new_size copies of @p value .
    constexpr void resize(size_type new_size, const value_type &value)
    {
        discard();
        acquire(new_size);

        for (auto lhs = data(); lhs != data() + size(); ++lhs)
            std::uninitialized_construct_using_allocator(
                lhs,
                get_allocator(),
                value);
    }

private:
    [[nodiscard]] allocator_type &get_allocator() noexcept
    {
        return m_allocator;
    }

    constexpr void acquire(size_type size)
    {
        assert(m_size == 0);

        if (size == 0) return;

        m_data = allocator_traits::allocate(get_allocator(), size);
        m_size = size;
    }

    constexpr void discard()
    {
        if (size() == 0) return;

        std::destroy_n(data(), size());
        allocator_traits::deallocate(get_allocator(), data(), size());
        m_size = 0;
    }

    [[no_unique_address]] allocator_type m_allocator;
    pointer m_data;
    size_type m_size;
};

} // namespace fxx
