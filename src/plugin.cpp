#include "rrtmg.h"

#include <cstddef>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>

namespace {

void dump_atmosphere(
    std::size_t n_layers,
    std::size_t n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *n_prime)
{
    const auto get_n_prime = [=](std::size_t i_lay, std::size_t i_gas)
    {
        return n_prime[i_lay * n_gas + i_gas];
    };

    std::cout
        << std::setw(5) << "i_eta"
        << std::setw(14) << "T_lay (K)"
        << std::setw(14) << "p_lay (Pa)"
        << std::setw(14) << "n_d (cm^-2)"
        << std::setw(14) << "n(h2o)"
        << std::setw(14) << "n(co2)"
        << std::setw(14) << "n(o3)"
        << std::setw(14) << "n(n2o)"
        << std::setw(14) << "n(ch4)"
        << std::setw(14) << "n(o2)"
        << std::endl;
    for (std::size_t i_lay = 0; i_lay < n_layers; ++i_lay)
    {
        std::cout
            << std::setw(5) << i_lay
            << std::fixed
            << std::setw(14) << T_lay[i_lay]
            << std::setw(14) << p_lay[i_lay]
            << std::scientific
            << std::setw(14) << n_d[i_lay]
            << std::setw(14) << get_n_prime(i_lay, 0)
            << std::setw(14) << get_n_prime(i_lay, 1)
            << std::setw(14) << get_n_prime(i_lay, 2)
            << std::setw(14) << get_n_prime(i_lay, 3)
            << std::setw(14) << get_n_prime(i_lay, 5)
            << std::setw(14) << get_n_prime(i_lay, 6)
            << std::endl;
    }
}

} // namespace

extern "C" {

INTEGER plugin_rrtmg_sw_init(
    INTEGER is_setup,
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    REAL cp_d)
{
#ifndef NDEBUG
    if (is_setup) {
        // This one should only be called once, but better safe than sorry.
        std::cout
            << "plugin_rrtmg_sw: init("
            << std::boolalpha << static_cast<bool>(is_setup)
            << ", " << n_layers
            << ", " << n_gpt
            << ", " << n_gas
            << ", " << std::scientific << cp_d
            << ")" << std::endl;
    }
#endif

    // Dummy implementation: fallback to built-in implementation.
    return PLUGIN_ERROR;
}

void plugin_rrtmg_sw_taumol(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *n_prime,
    REAL *tau_gas,
    REAL *tau_rayl)
{
#ifndef NDEBUG
    std::cout
        << "plugin_rrtmg_sw: taumol(...)" << std::endl;
    dump_atmosphere(n_layers, n_gas, T_lay, p_lay, n_d, n_prime);
#endif

    // Dummy implementation: fill with tiny non-zero value (used in division).
    // constexpr auto tiny = std::numeric_limits<REAL>::denorm_min();
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    std::fill_n(tau_gas, n_gpt * n_layers, tiny);
    std::fill_n(tau_rayl, n_gpt * n_layers, tiny);
}

void plugin_rrtmg_sw_solar_source(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *n_prime,
    REAL *E_solar)
{
    // Dummy implementation: fill with tiny non-zero value.
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    std::fill_n(E_solar, n_gpt, tiny);
}

INTEGER plugin_rrtmg_lw_init(
    INTEGER is_setup,
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    REAL cp_d)
{
#ifndef NDEBUG
    if (is_setup) {
        // This one's called on every timestep, so this guard matters!
        std::cout
            << "plugin_rrtmg_lw: init("
            << std::boolalpha << static_cast<bool>(is_setup)
            << ", " << n_layers
            << ", " << n_gpt
            << ", " << n_gas
            << ", " << std::scientific << cp_d
            << ")" << std::endl;
    }
#endif

    // Dummy implementation: fallback to built-in implementation.
    return PLUGIN_ERROR;
}

void plugin_rrtmg_lw_planck_source(
    INTEGER n_layers,
    INTEGER n_bands,
    INTEGER n_gas,
    const REAL *T_lay,
    const REAL *T_lev,
    REAL T_sfc,
    const REAL *epsilon_sfc,
    REAL *E_lay,
    REAL *E_lev,
    REAL *E_bnd)
{
    // Dummy implementation: fill with tiny non-zero value.
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    std::fill_n(E_lay, n_layers * n_bands, tiny);
    std::fill_n(E_lev, n_layers * n_bands, tiny);
    std::fill_n(E_bnd, n_bands, tiny);
}

void plugin_rrtmg_lw_taumol(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *n_prime,
    REAL *tau_gas,
    REAL *a_planck)
{
#ifndef NDEBUG
    std::cout
        << "plugin_rrtmg_lw: taumol(...)" << std::endl;
    dump_atmosphere(n_layers, n_gas, T_lay, p_lay, n_d, n_prime);
#endif

    // Dummy implementation: fill with tiny non-zero value (used in division).
    // constexpr auto tiny = std::numeric_limits<REAL>::denorm_min();
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    std::fill_n(tau_gas, n_gpt * n_layers, tiny);
    std::fill_n(a_planck, n_gpt * n_layers, tiny);
}

}
