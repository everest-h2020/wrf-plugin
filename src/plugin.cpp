#include "rrtmg.h"
#include "rrtmg/kernels.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace std;
using namespace rrtmg;

#include "data.inc"

namespace {

[[maybe_unused]] void dump_atmosphere(
    size_t n_layers,
    size_t n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *r_gas)
{
    const auto get_r_gas = [=](size_t i_lay, size_t i_gas) {
        return r_gas[i_lay * n_gas + i_gas];
    };

    // clang-format off
    cout
        << setw(5) << "i_eta"
        << setw(14) << "T_lay (K)"
        << setw(14) << "p_lay (Pa)"
        << setw(14) << "n_d (cm^-2)"
        << setw(14) << "n(h2o)"
        << setw(14) << "n(co2)"
        << setw(14) << "n(o3)"
        << setw(14) << "n(n2o)"
        << setw(14) << "n(ch4)"
        << setw(14) << "n(o2)"
        << endl;
    for (size_t i_lay = 0; i_lay < n_layers; ++i_lay)
    {
        cout
            << setw(5) << i_lay
            << fixed
            << setw(14) << T_lay[i_lay]
            << setw(14) << p_lay[i_lay]
            << scientific
            << setw(14) << n_d[i_lay]
            << setw(14) << get_r_gas(i_lay, 0)
            << setw(14) << get_r_gas(i_lay, 1)
            << setw(14) << get_r_gas(i_lay, 2)
            << setw(14) << get_r_gas(i_lay, 3)
            << setw(14) << get_r_gas(i_lay, 5)
            << setw(14) << get_r_gas(i_lay, 6)
            << endl;
    }
    // clang-format on
}

static bool is_plugin_disabled()
{
    if (const auto env_var = getenv("PLUGIN_RRTMG")) {
        std::string_view value(env_var);
        return value == "0";
    }

    return false;
}

} // namespace

extern "C" {

INTEGER plugin_rrtmg_sw_init(
    INTEGER is_setup,
    [[maybe_unused]] INTEGER n_layers,
    [[maybe_unused]] INTEGER n_gpt,
    [[maybe_unused]] INTEGER n_gas,
    [[maybe_unused]] REAL cp_d)
{
    if (is_setup) {
#ifndef NDEBUG
        // This one should only be called once, but better safe than sorry.
        cout << "plugin_rrtmg_sw: init(" << boolalpha
             << static_cast<bool>(is_setup) << ", " << n_layers << ", " << n_gpt
             << ", " << n_gas << ", " << scientific << cp_d << ")" << endl;
#endif
    }

    return is_plugin_disabled() ? PLUGIN_ERROR : PLUGIN_OK;
}

void plugin_rrtmg_sw_taumol(
    INTEGER n_layers,
    [[maybe_unused]] INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *r_gas,
    REAL *tau_gas,
    REAL *tau_rayl)
{
#ifndef NDEBUG
    cout << "plugin_rrtmg_sw: taumol(...)" << endl;
    dump_atmosphere(n_layers, n_gas, T_lay, p_lay, n_d, r_gas);
#endif

    assert(n_layers <= static_cast<index_t>(N_CELL));
    assert(n_gpt == C_N_GPT);
    assert(n_gas >= static_cast<index_t>(N_GAS));

    index_t i_lay = 0;
    while (i_lay < n_layers) {
        // Copy to minimized gas VMR array.
        REAL r_gas_part[N_GAS][N_CELL];
        for (size_t i_cell = 0; i_cell < N_CELL; ++i_cell)
            for (size_t i_gas = 0; i_gas < N_GAS; ++i_gas)
                r_gas_part[i_gas][i_cell] = r_gas[i_cell * n_gas + i_gas];

        // Output to regular-sized tau arrays.
        REAL tau_g_part[N_BND][N_CELL][N_GPB];
        REAL tau_r_part[N_BND][N_CELL][N_GPB];
        taumol_sw(T_lay, p_lay, n_d, r_gas_part, tau_g_part, tau_r_part);

        // Copy taus to result array.
        const auto n_cell = min(n_layers - i_lay, N_CELL);
        index_t i_gpt = 0;
        for (index_t i_bnd = 0; i_bnd < C_N_BND; ++i_bnd) {
            for (index_t i_gpb = 0; i_gpb < C_BND_WIDTH[i_bnd]; ++i_gpb) {
                for (index_t i_cell = 0; i_cell < n_cell; ++i_cell) {
                    tau_gas[i_gpt * n_layers + i_lay + i_cell] =
                        tau_g_part[i_bnd][i_cell][i_gpb];
                    tau_rayl[i_gpt * n_layers + i_lay + i_cell] =
                        tau_r_part[i_bnd][i_cell][i_gpb];
                }
                ++i_gpt;
            }
        }

        // Advance iterators.
        i_lay += N_CELL;
        T_lay += N_CELL;
        p_lay += N_CELL;
        n_d += N_CELL;
        r_gas += N_CELL * n_gas;
    }
}

void plugin_rrtmg_sw_solar_source(
    INTEGER,
    [[maybe_unused]] INTEGER n_gpt,
    INTEGER,
    const REAL *,
    REAL *E_solar)
{
    assert(n_gpt == C_N_GPT);

    // Copy default-estimated solar source.
    for (index_t i_gpt = 0; i_gpt < C_N_GPT; ++i_gpt)
        E_solar[i_gpt] = C_E_SOLAR[i_gpt];
}

INTEGER plugin_rrtmg_lw_init(
    INTEGER is_setup,
    [[maybe_unused]] INTEGER n_layers,
    [[maybe_unused]] INTEGER n_gpt,
    [[maybe_unused]] INTEGER n_gas,
    [[maybe_unused]] REAL cp_d)
{
    if (is_setup) {
#ifndef NDEBUG
        // This one's called on every timestep, so this guard matters!
        cout << "plugin_rrtmg_lw: init(" << boolalpha
             << static_cast<bool>(is_setup) << ", " << n_layers << ", " << n_gpt
             << ", " << n_gas << ", " << scientific << cp_d << ")" << endl;
#endif
    }

    // Dummy implementation: fallback to built-in implementation.
    return PLUGIN_ERROR;
}

void plugin_rrtmg_lw_planck_source(
    INTEGER n_layers,
    INTEGER n_bands,
    INTEGER,
    const REAL *,
    const REAL *,
    REAL,
    const REAL *,
    REAL *E_lay,
    REAL *E_lev,
    REAL *E_bnd)
{
    // Dummy implementation: fill with tiny non-zero value.
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    fill_n(E_lay, n_layers * n_bands, tiny);
    fill_n(E_lev, n_layers * n_bands, tiny);
    fill_n(E_bnd, n_bands, tiny);
}

void plugin_rrtmg_lw_taumol(
    INTEGER n_layers,
    INTEGER n_gpt,
    [[maybe_unused]] INTEGER n_gas,
    [[maybe_unused]] const REAL *T_lay,
    [[maybe_unused]] const REAL *p_lay,
    [[maybe_unused]] const REAL *n_d,
    [[maybe_unused]] const REAL *r_gas,
    REAL *tau_gas,
    REAL *a_planck)
{
#ifndef NDEBUG
    cout << "plugin_rrtmg_lw: taumol(...)" << endl;
    dump_atmosphere(n_layers, n_gas, T_lay, p_lay, n_d, r_gas);
#endif

    // Dummy implementation: fill with tiny non-zero value (used in division).
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    fill_n(tau_gas, n_gpt * n_layers, tiny);
    fill_n(a_planck, n_gpt * n_layers, tiny);
}
}
