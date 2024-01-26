#pragma once

#include "ABI.h"

#include <array>
#include <cstddef>

namespace rrtmg {

using index_t = std::ptrdiff_t;

static constexpr index_t N_CELL = 60;
static constexpr index_t N_GAS = 7;
static constexpr index_t N_GPB = 16;
static constexpr index_t N_BND = 16;

} // namespace rrtmg

extern "C" {

void plugin_rrtmg_taumol_sw(
    const REAL T[rrtmg::N_CELL],
    const REAL p[rrtmg::N_CELL],
    const REAL n_d[rrtmg::N_CELL],
    const REAL r_gas[rrtmg::N_GAS][rrtmg::N_CELL],
    REAL tau_g[rrtmg::N_BND][rrtmg::N_CELL][rrtmg::N_GPB],
    REAL tau_r[rrtmg::N_BND][rrtmg::N_CELL][rrtmg::N_GPB]);

} // extern "C"
