// Modified from:
//     https://github.com/microhh/rte-rrtmgp-cpp/blob/main/include/Gas_optics_rrtmgp.h

#pragma once

#include "ABI.h"
#include "rrtmgp/Array.h"

#include <string>
#include <string_view>

namespace rrtmgp {

struct Constants {
    explicit Constants(std::string_view filename);

    void init_abs_coeffs(
        const Array<int,3>& key_species,
        const Array<int,2>& band2gpt,
        const Array<REAL,2>& band_lims_wavenum,
        const Array<REAL,1>& press_ref,
        const Array<REAL,1>& temp_ref,
        const REAL press_ref_trop,
        const REAL temp_ref_p,
        const REAL temp_ref_t,
        const Array<std::string,1>& gas_minor,
        const Array<std::string,1>& identifier_minor,
        const Array<std::string,1>& minor_gases_lower,
        const Array<std::string,1>& minor_gases_upper,
        const Array<std::string,1>& scaling_gas_lower,
        const Array<std::string,1>& scaling_gas_upper,
        const Array<REAL,3>& rayl_lower,
        const Array<REAL,3>& rayl_upper);

    Array<int,2> band2gpt;
    Array<int,1> gpt2band;
    Array<REAL,2> band_lims_wvn;

    Array<REAL,2> totplnk;
    Array<REAL,4> planck_frac;
    REAL totplnk_delta;
    REAL temp_ref_min, temp_ref_max;
    REAL press_ref_min, press_ref_max;
    REAL press_ref_trop_log;

    REAL press_ref_log_delta;
    REAL temp_ref_delta;

    Array<REAL,1> press_ref, press_ref_log, temp_ref;

    Array<std::string,1> gas_names;

    Array<REAL,3> vmr_ref;

    Array<int,2> flavor;
    Array<int,2> gpoint_flavor;

    Array<REAL,4> kmajor;

    Array<REAL,3> kminor_lower;
    Array<REAL,3> kminor_upper;

    Array<int,2> minor_limits_gpt_lower;
    Array<int,2> minor_limits_gpt_upper;

    Array<BOOL,1> minor_scales_with_density_lower;
    Array<BOOL,1> minor_scales_with_density_upper;

    Array<BOOL,1> scale_by_complement_lower;
    Array<BOOL,1> scale_by_complement_upper;

    Array<int,1> kminor_start_lower;
    Array<int,1> kminor_start_upper;

    Array<int,1> idx_minor_lower;
    Array<int,1> idx_minor_upper;

    Array<int,1> idx_minor_scaling_lower;
    Array<int,1> idx_minor_scaling_upper;

    Array<int,1> is_key;

    Array<REAL,1> solar_source_quiet;
    Array<REAL,1> solar_source_facular;
    Array<REAL,1> solar_source_sunspot;
    Array<REAL,1> solar_source;

    Array<REAL,4> krayl;
};

} // namespace rrtmgp
