#pragma once

#include "ABI.h"

#include <array>
#include <concepts>
#include <cstddef>

namespace rrtmg {

using index_t = std::ptrdiff_t;

namespace detail {

template<class T, std::size_t...>
struct ndarray {
    using type = T;
};
template<class T, std::size_t Head, std::size_t... Tail>
struct ndarray<T, Head, Tail...> {
    using type = std::array<typename ndarray<T, Tail...>::type, Head>;
};

} // namespace detail

template<class T, std::convertible_to<std::size_t> auto... Extents>
using ndarray =
    typename detail::ndarray<T, static_cast<std::size_t>(Extents)...>::type;

static constexpr index_t N_CELL = 60;
static constexpr index_t N_GAS = 7;
static constexpr index_t N_GPB = 16;
static constexpr index_t N_BND = 16;

void taumol_sw(
    const ndarray<REAL, N_CELL> &T,
    const ndarray<REAL, N_CELL> &p,
    const ndarray<REAL, N_CELL> &n_d,
    const ndarray<REAL, N_GAS, N_CELL> &r_gas,
    ndarray<REAL, N_BND, N_CELL, N_GPB> &tau_g,
    ndarray<REAL, N_BND, N_CELL, N_GPB> &tau_r);

} // namespace rrtmg
