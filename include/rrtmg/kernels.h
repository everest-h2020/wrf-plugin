#pragma once

#include "ABI.h"

#include <array>
#include <concepts>
#include <cstddef>

namespace rrtmg {

using index_t = std::ptrdiff_t;

static constexpr index_t N_CELL = 60;
static constexpr index_t N_GAS = 7;
static constexpr index_t N_GPB = 16;
static constexpr index_t N_BND = 16;

void taumol_sw(
    const REAL T[N_CELL],
    const REAL p[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    REAL tau_g[N_BND][N_CELL][N_GPB],
    REAL tau_r[N_BND][N_CELL][N_GPB]);

} // namespace rrtmg
