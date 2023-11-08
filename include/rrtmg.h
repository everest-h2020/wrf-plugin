#pragma once

#include "ABI.h"

enum : INTEGER {
    /// Indicates that the plugin was initialized successfully.
    PLUGIN_OK = 0,
    /// Indicates that there was an error initializing the plugin.
    PLUGIN_ERROR = -1
};

// WRF gas species: "h20", "co2", "o3", "n2o", N/A, "ch4", "o2", N/A...

extern "C" {

/// Initializes the RRTMG shortwave plugin.
///
/// @param              is_setup    @c 1 if this is the one-time-setup.
/// @param              n_layers    Number of model layers (1).
/// @param              n_gpt       Number of model g-points (1).
/// @param              n_gas       Number of model gas species (1).
/// @param              cp_d        Isobaric heat capacity of dry air (J/kg/K).
///
/// @retval PLUGIN_OK       Plugin can be used.
/// @retval PLUGIN_ERROR    Fallback to the built-in solver.
INTEGER plugin_rrtmg_sw_init(
    INTEGER is_setup,
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    REAL cp_d);

/// Computes the optical depth in the shortwave domain.
///
/// @param              n_layers    Number of model layers (1).
/// @param              n_gpt       Number of model g-points (1).
/// @param              n_gas       Number of model gas species (1).
/// @param  [in]        T_lay       Layer temperatures [n_layers] (K).
/// @param  [in]        p_lay       Layer pressures [n_layers] (Pa).
/// @param  [in]        n_d         Dry air mass in column [n_layers] (molec./cm^2).
/// @param  [in]        n_prime     Volume mixing ratios of gas contributors [n_layers,n_gas] (1).
/// @param  [out]       tau_gas     Optical depth from gas species [n_gpt,n_layers] (ln 1).
/// @param  [out]       tau_rayl    Optical depth from Rayleigh scattering [n_gpt,n_layers] (ln 1).
void plugin_rrtmg_sw_taumol(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *n_prime,
    REAL *tau_gas,
    REAL *tau_rayl);

/// Computes the solar source flux in the shortwave domain.
///
/// @param              n_layers    Number of model layers (1).
/// @param              n_gpt       Number of model g-points (1).
/// @param              n_gas       Number of model gas species (1).
/// @param  [in]        n_prime     Volume mixing ratios of gas contributors [n_layers,n_gas] (1).
/// @param  [out]       E_solar     Solar source flux [n_gpt] (W/cm^2).
void plugin_rrtmg_sw_solar_source(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *n_prime,
    REAL *E_solar);

/// Initializes the RRTMG longwave plugin.
///
/// @param              is_setup    @c 1 if this is the one-time-setup.
/// @param              n_layers    Number of model layers (1).
/// @param              n_gpt       Number of model g-points (1).
/// @param              n_gas       Number of model gas species (1).
/// @param              cp_d        Isobaric heat capacity of dry air (J/kg/K).
///
/// @retval PLUGIN_OK       Plugin can be used.
/// @retval PLUGIN_ERROR    Fallback to the built-in solver.
INTEGER plugin_rrtmg_lw_init(
    INTEGER is_setup,
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    REAL cp_d);

/// Computes the contribution of Planck radiance.
///
/// @param              n_layers    Number of model layers (1).
/// @param              n_bands     Number of model wavelength bands (1).
/// @param              n_gas       Number of model gas species (1).
/// @param  [in]        T_lay       Layer temperatures [n_layers] (K).
/// @param  [in]        T_lev       Level temperatures [n_layers+1] (K).
/// @param  [in]        T_sfc       Surface temperature (K).
/// @param  [in]        epsilon_sfc Surface emissivity per band [n_bands] (1).
/// @param  [out]       E_lay       Planck flux per layer [n_bands,n_layers] (W/m^2???).
/// @param  [out]       E_lev       Planck flux per level [n_bands,n_layers] (W/m^2???).
/// @param  [out]       E_bnd       Surface Planck flux??? [n_layers,n_bands] (W/m^2???).
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
    REAL *E_bnd);

/// Computes the optical depth in the longwave domain.
///
/// @param              n_layers    Number of model layers (1).
/// @param              n_gpt       Number of model g-points (1).
/// @param              n_gas       Number of model gas species (1).
/// @param  [in]        T_lay       Layer temperatures [n_layers] (K).
/// @param  [in]        p_lay       Layer pressures [n_layers] (Pa).
/// @param  [in]        n_d         Dry air mass in column [n_layers] (molec./cm^2).
/// @param  [in]        n_prime     Volume mixing ratios of gas contributors [n_layers,n_gas] (1).
/// @param  [out]       tau_gas     Optical depth from gas species [n_gpt,n_layers] (ln 1).
/// @param  [out]       a_planck    Planck fractions [n_layers,n_gpt] (1).
void plugin_rrtmg_lw_taumol(
    INTEGER n_layers,
    INTEGER n_gpt,
    INTEGER n_gas,
    const REAL *T_lay,
    const REAL *p_lay,
    const REAL *n_d,
    const REAL *n_prime,
    REAL *tau_gas,
    REAL *a_planck);

}
