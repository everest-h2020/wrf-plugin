#include "rrtmg.h"
#include "rrtmgp/Constants.h"
#include "rrtmgp/kernel_launchers.h"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

using namespace std;
using namespace rrtmgp;

namespace {

void dump_atmosphere(
    std::size_t n_layers,
    std::size_t n_gas,
    const REAL* T_lay,
    const REAL* p_lay,
    const REAL* n_d,
    const REAL* n_prime)
{
    const auto get_n_prime = [=](std::size_t i_lay, std::size_t i_gas) {
        return n_prime[i_lay * n_gas + i_gas];
    };

    // clang-format off
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
    // clang-format on
}

Constants &get_or_load_constants()
{
    static Constants constants(
        "/home/friebel/everest/rte-rrtmgp-cpp/rte-rrtmgp/rrtmgp/data/"
        "rrtmgp-data-sw-g112-210809.nc");

    return constants;
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
        std::cout << "plugin_rrtmg_sw: init(" << std::boolalpha
                  << static_cast<bool>(is_setup) << ", " << n_layers << ", "
                  << n_gpt << ", " << n_gas << ", " << std::scientific << cp_d
                  << ")" << std::endl;
    }
#endif

    std::ignore = get_or_load_constants();
    return PLUGIN_OK;
}

void plugin_rrtmg_sw_taumol(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL* T_lay,
    const REAL* p_lay,
    const REAL* n_d,
    const REAL* n_prime,
    REAL* tau_gas,
    REAL* tau_rayl)
{
#ifndef NDEBUG
    std::cout << "plugin_rrtmg_sw: taumol(...)" << std::endl;
    dump_atmosphere(n_layers, n_gas, T_lay, p_lay, n_d, n_prime);
#endif

    const auto &constants = get_or_load_constants();

    const auto n_bnd = constants.band2gpt.dim(1);
    const auto n_flav = constants.flavor.dim(1);
    const auto n_eta = constants.kmajor.dim(2);
    const auto n_pres = constants.kmajor.dim(3) - 1;
    const auto n_temp = constants.kmajor.dim(4);

    REAL* col_gas = (REAL*)calloc(n_layers * n_gas, sizeof(REAL));

    for (int i = 0; i < n_layers; ++i)
        for (int j = 0; j < n_gas; ++j)
            col_gas[i * n_gas + j] = n_prime[i * n_gas + j] * n_d[i];

    REAL* fmajor = (REAL*)calloc(n_flav * n_layers * 2 * 2 * 2, sizeof(REAL));
    REAL* fminor = (REAL*)calloc(n_flav * n_layers * 2 * 2, sizeof(REAL));
    REAL* col_mix = (REAL*)calloc(n_flav * n_layers * 2, sizeof(REAL));
    BOOL* tropo = (BOOL*)calloc(n_layers, sizeof(BOOL));
    int* jtemp = (int*)calloc(n_layers, sizeof(int));
    int* jpress = (int*)calloc(n_layers, sizeof(int));
    int* jeta = (int*)calloc(n_flav * n_layers * 2, sizeof(int));

    interpolation_fpga(
        1,
        n_layers,
        n_gas,
        n_flav,
        n_eta,
        n_pres,
        n_temp,
        constants.flavor.data.data(),
        constants.press_ref_log.data.data(),
        constants.temp_ref.data.data(),
        constants.press_ref_log_delta,
        constants.temp_ref_min,
        constants.temp_ref_max,
        constants.press_ref_trop_log,
        constants.vmr_ref.data.data(),
        p_lay,
        T_lay,
        col_gas,
        jtemp,
        fmajor,
        fminor,
        col_mix,
        tropo,
        jeta,
        jpress);

    const int n_minor_lower = constants.minor_scales_with_density_lower.dim(1);
    const int n_minork_lower = constants.kminor_lower.dim(3);
    const int n_minor_upper = constants.minor_scales_with_density_upper.dim(1);
    const int n_minork_upper = constants.kminor_upper.dim(3);

    compute_tau_absorption_fpga(
        1,
        n_layers,
        n_bnd,
        n_gpt,
        n_gas,
        n_flav,
        n_eta,
        n_pres,
        n_temp,
        n_minor_lower,
        n_minork_lower,
        n_minor_upper,
        n_minork_upper,
        1,
        constants.gpoint_flavor.data.data(),
        constants.band2gpt.data.data(),
        constants.kmajor.data.data(),
        constants.kminor_lower.data.data(),
        constants.kminor_upper.data.data(),
        constants.minor_limits_gpt_lower.data.data(),
        constants.minor_limits_gpt_upper.data.data(),
        constants.minor_scales_with_density_lower.data.data(),
        constants.minor_scales_with_density_upper.data.data(),
        constants.scale_by_complement_lower.data.data(),
        constants.scale_by_complement_upper.data.data(),
        constants.idx_minor_lower.data.data(),
        constants.idx_minor_upper.data.data(),
        constants.idx_minor_scaling_lower.data.data(),
        constants.idx_minor_scaling_upper.data.data(),
        constants.kminor_start_lower.data.data(),
        constants.kminor_start_upper.data.data(),
        tropo,
        col_mix,
        fmajor,
        fminor,
        p_lay,
        T_lay,
        col_gas,
        jeta,
        jtemp,
        jpress,
        tau_gas);

    compute_tau_rayleigh_fpga(
        1,
        n_layers,
        n_bnd,
        n_gpt,
        n_gas,
        n_flav,
        n_eta,
        n_pres,
        n_temp,
        constants.gpoint_flavor.data.data(),
        constants.gpt2band.data.data(),
        constants.band2gpt.data.data(),
        constants.krayl.data.data(),
        1,
        n_d,
        col_gas,
        fminor,
        jeta,
        tropo,
        jtemp,
        tau_rayl);
}

void plugin_rrtmg_sw_solar_source(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL* n_prime,
    REAL* E_solar)
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
        std::cout << "plugin_rrtmg_lw: init(" << std::boolalpha
                  << static_cast<bool>(is_setup) << ", " << n_layers << ", "
                  << n_gpt << ", " << n_gas << ", " << std::scientific << cp_d
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
    const REAL* T_lay,
    const REAL* T_lev,
    REAL T_sfc,
    const REAL* epsilon_sfc,
    REAL* E_lay,
    REAL* E_lev,
    REAL* E_bnd)
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
    const REAL* T_lay,
    const REAL* p_lay,
    const REAL* n_d,
    const REAL* n_prime,
    REAL* tau_gas,
    REAL* a_planck)
{
#ifndef NDEBUG
    std::cout << "plugin_rrtmg_lw: taumol(...)" << std::endl;
    dump_atmosphere(n_layers, n_gas, T_lay, p_lay, n_d, n_prime);
#endif

    // Dummy implementation: fill with tiny non-zero value (used in division).
    // constexpr auto tiny = std::numeric_limits<REAL>::denorm_min();
    constexpr auto tiny = static_cast<REAL>(1.0e-25);
    std::fill_n(tau_gas, n_gpt * n_layers, tiny);
    std::fill_n(a_planck, n_gpt * n_layers, tiny);
}
}
