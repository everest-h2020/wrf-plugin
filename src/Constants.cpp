// Excerpt from:
//     https://github.com/microhh/rte-rrtmgp-cpp/blob/main/src_test/Radiation_solver.cpp

#include "rrtmgp/Constants.h"

#include "rrtmgp/Netcdf_interface.h"

#include <cctype>
#include <cmath>

using namespace rrtmgp;
using namespace std;

namespace {

int find_index(const Array<std::string, 1> &data, const std::string &value)
{
    const auto it = std::find(data.data.begin(), data.data.end(), value);
    if (it == data.data.end()) return -1;
    return std::distance(data.data.begin(), it);
}

void trim(std::string &s)
{
    constexpr auto is_space = [](char x) { return std::isspace(x); };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), is_space));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), is_space).base(), s.end());
}

std::vector<std::string> get_variable_string(
    const std::string &var_name,
    std::vector<int> i_count,
    Netcdf_handle &input_nc,
    const int string_len,
    bool trim = true)
{
    // Multiply all elements in i_count.
    int total_count =
        std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

    // Add the string length as the rightmost dimension.
    i_count.push_back(string_len);

    // Read the entire char array;
    std::vector<char> var_char;
    var_char = input_nc.get_variable<char>(var_name, i_count);

    std::vector<std::string> var;

    for (int n = 0; n < total_count; ++n) {
        std::string s(
            var_char.begin() + n * string_len,
            var_char.begin() + (n + 1) * string_len);
        if (trim) ::trim(s);
        var.push_back(s);
    }

    return var;
}

constexpr auto wrf_gases =
    std::array{"h20", "co2", "o3", "n2o", "", "ch4", "o2", ""};

template<typename REAL>
void reduce_minor_arrays(
    const Array<std::string, 1> &gas_names,
    const Array<std::string, 1> &gas_minor,
    const Array<std::string, 1> &identifier_minor,
    const Array<REAL, 3> &kminor_atm,
    const Array<std::string, 1> &minor_gases_atm,
    const Array<int, 2> &minor_limits_gpt_atm,
    const Array<BOOL, 1> &minor_scales_with_density_atm,
    const Array<std::string, 1> &scaling_gas_atm,
    const Array<BOOL, 1> &scale_by_complement_atm,
    const Array<int, 1> &kminor_start_atm,

    Array<REAL, 3> &kminor_atm_red,
    Array<std::string, 1> &minor_gases_atm_red,
    Array<int, 2> &minor_limits_gpt_atm_red,
    Array<BOOL, 1> &minor_scales_with_density_atm_red,
    Array<std::string, 1> &scaling_gas_atm_red,
    Array<BOOL, 1> &scale_by_complement_atm_red,
    Array<int, 1> &kminor_start_atm_red)
{
    int nm = minor_gases_atm.dim(1);
    int tot_g = 0;

    Array<BOOL, 1> gas_is_present({nm});

    for (int i = 1; i <= nm; ++i) {
        const int idx_mnr = find_index(identifier_minor, minor_gases_atm({i}));

        // Search for
        std::string gas_minor_trimmed = gas_minor({idx_mnr});
        trim(gas_minor_trimmed);

        gas_is_present({i}) =
            std::find(wrf_gases.begin(), wrf_gases.end(), gas_minor_trimmed)
            != wrf_gases.end();
        if (gas_is_present({i}))
            tot_g +=
                minor_limits_gpt_atm({2, i}) - minor_limits_gpt_atm({1, i}) + 1;
    }

    const int red_nm = std::accumulate(
        gas_is_present.v().begin(),
        gas_is_present.v().end(),
        0);

    Array<REAL, 3> kminor_atm_red_t;

    if (red_nm == nm) {
        kminor_atm_red_t = kminor_atm;
        minor_gases_atm_red = minor_gases_atm;
        minor_limits_gpt_atm_red = minor_limits_gpt_atm;
        minor_scales_with_density_atm_red = minor_scales_with_density_atm;
        scaling_gas_atm_red = scaling_gas_atm;
        scale_by_complement_atm_red = scale_by_complement_atm;
        kminor_start_atm_red = kminor_start_atm;
    } else {
        // Use a lambda function as the operation has to be repeated many times.
        auto resize_and_set = [&](auto &a_red, const auto &a) {
            a_red.set_dims({red_nm});
            int counter = 1;
            for (int i = 1; i <= gas_is_present.dim(1); ++i) {
                if (gas_is_present({i})) {
                    a_red({counter}) = a({i});
                    ++counter;
                }
            }
        };

        resize_and_set(minor_gases_atm_red, minor_gases_atm);
        resize_and_set(
            minor_scales_with_density_atm_red,
            minor_scales_with_density_atm);
        resize_and_set(scaling_gas_atm_red, scaling_gas_atm);
        resize_and_set(scale_by_complement_atm_red, scale_by_complement_atm);
        resize_and_set(kminor_start_atm_red, kminor_start_atm);

        minor_limits_gpt_atm_red.set_dims({2, red_nm});
        kminor_atm_red_t.set_dims(
            {tot_g, kminor_atm.dim(2), kminor_atm.dim(3)});

        int icnt = 0;
        int n_elim = 0;
        for (int i = 1; i <= nm; ++i) {
            int ng =
                minor_limits_gpt_atm({2, i}) - minor_limits_gpt_atm({1, i}) + 1;
            if (gas_is_present({i})) {
                ++icnt;
                minor_limits_gpt_atm_red({1, icnt}) =
                    minor_limits_gpt_atm({1, i});
                minor_limits_gpt_atm_red({2, icnt}) =
                    minor_limits_gpt_atm({2, i});
                kminor_start_atm_red({icnt}) = kminor_start_atm({i}) - n_elim;

                for (int j = 1; j <= ng; ++j)
                    for (int i2 = 1; i2 <= kminor_atm.dim(2); ++i2)
                        for (int i3 = 1; i3 <= kminor_atm.dim(3); ++i3)
                            kminor_atm_red_t(
                                {kminor_start_atm_red({icnt}) + j - 1,
                                 i2,
                                 i3}) =
                                kminor_atm(
                                    {kminor_start_atm({i}) + j - 1, i2, i3});
            } else
                n_elim += ng;
        }
    }

    // Reshape following the new ordering in v1.5.
    kminor_atm_red.set_dims(
        {kminor_atm_red_t.dim(3),
         kminor_atm_red_t.dim(2),
         kminor_atm_red_t.dim(1)});
    for (int i3 = 1; i3 <= kminor_atm_red.dim(3); ++i3)
        for (int i2 = 1; i2 <= kminor_atm_red.dim(2); ++i2)
            for (int i1 = 1; i1 <= kminor_atm_red.dim(1); ++i1)
                kminor_atm_red({i1, i2, i3}) = kminor_atm_red_t({i3, i2, i1});
}

void create_idx_minor(
    const Array<std::string, 1> &gas_names,
    const Array<std::string, 1> &gas_minor,
    const Array<std::string, 1> &identifier_minor,
    const Array<std::string, 1> &minor_gases_atm,
    Array<int, 1> &idx_minor_atm)
{
    Array<int, 1> idx_minor_atm_out({minor_gases_atm.dim(1)});

    for (int imnr = 1; imnr <= minor_gases_atm.dim(1); ++imnr) {
        // Find identifying string for minor species in list of possible
        // identifiers (e.g. h2o_slf)
        const int idx_mnr =
            find_index(identifier_minor, minor_gases_atm({imnr}));

        // Find name of gas associated with minor species identifier (e.g. h2o)
        idx_minor_atm_out({imnr}) = find_index(gas_names, gas_minor({idx_mnr}));
    }

    idx_minor_atm = idx_minor_atm_out;
}

void create_idx_minor_scaling(
    const Array<std::string, 1> &gas_names,
    const Array<std::string, 1> &scaling_gas_atm,
    Array<int, 1> &idx_minor_scaling_atm)
{
    Array<int, 1> idx_minor_scaling_atm_out({scaling_gas_atm.dim(1)});

    for (int imnr = 1; imnr <= scaling_gas_atm.dim(1); ++imnr)
        idx_minor_scaling_atm_out({imnr}) =
            find_index(gas_names, scaling_gas_atm({imnr}));

    idx_minor_scaling_atm = idx_minor_scaling_atm_out;
}

void create_key_species_reduce(
    const Array<std::string, 1> &gas_names,
    const Array<std::string, 1> &gas_names_red,
    const Array<int, 3> &key_species,
    Array<int, 3> &key_species_red,
    Array<BOOL, 1> &key_species_present_init)
{
    const int np = key_species.dim(1);
    const int na = key_species.dim(2);
    const int nt = key_species.dim(3);

    key_species_red.set_dims(
        {key_species.dim(1), key_species.dim(2), key_species.dim(3)});
    key_species_present_init.set_dims({gas_names.dim(1)});

    for (int i = 1; i <= key_species_present_init.dim(1); ++i)
        key_species_present_init({i}) = 1;

    for (int ip = 1; ip <= np; ++ip)
        for (int ia = 1; ia <= na; ++ia)
            for (int it = 1; it <= nt; ++it) {
                const int ks = key_species({ip, ia, it});
                if (ks != 0) {
                    const int ksr = find_index(gas_names_red, gas_names({ks}));
                    key_species_red({ip, ia, it}) = ksr;
                    if (ksr == -1) key_species_present_init({ks}) = 0;
                } else
                    key_species_red({ip, ia, it}) = ks;
            }
}

void check_key_species_present_init(
    const Array<std::string, 1> &gas_names,
    const Array<BOOL, 1> &key_species_present_init)
{
    for (int i = 1; i <= key_species_present_init.dim(1); ++i) {
        if (key_species_present_init({i}) == 0) {
            std::string error_message =
                "Gas optics: required gas " + gas_names({i}) + " is missing";
            throw std::runtime_error(error_message);
        }
    }
}

void create_flavor(const Array<int, 3> &key_species, Array<int, 2> &flavor)
{
    Array<int, 2> key_species_list({2, key_species.dim(3) * 2});

    // Prepare list of key species.
    int i = 1;
    for (int ibnd = 1; ibnd <= key_species.dim(3); ++ibnd)
        for (int iatm = 1; iatm <= key_species.dim(2); ++iatm) {
            key_species_list({1, i}) = key_species({1, iatm, ibnd});
            key_species_list({2, i}) = key_species({2, iatm, ibnd});
            ++i;
        }

    // Rewrite single key_species pairs.
    for (int i = 1; i <= key_species_list.dim(2); ++i) {
        if (key_species_list({1, i}) == 0 && key_species_list({2, i}) == 0) {
            key_species_list({1, i}) = 2;
            key_species_list({2, i}) = 2;
        }
    }

    // Count unique key species pairs.
    int iflavor = 0;
    for (int i = 1; i <= key_species_list.dim(2); ++i) {
        bool pair_exists = false;
        for (int ii = 1; ii <= i - 1; ++ii) {
            if ((key_species_list({1, i}) == key_species_list({1, ii}))
                && (key_species_list({2, i}) == key_species_list({2, ii}))) {
                pair_exists = true;
                break;
            }
        }
        if (!pair_exists) ++iflavor;
    }

    // Fill flavors.
    flavor.set_dims({2, iflavor});
    iflavor = 0;
    for (int i = 1; i <= key_species_list.dim(2); ++i) {
        bool pair_exists = false;
        for (int ii = 1; ii <= i - 1; ++ii) {
            if ((key_species_list({1, i}) == key_species_list({1, ii}))
                && (key_species_list({2, i}) == key_species_list({2, ii}))) {
                pair_exists = true;
                break;
            }
        }
        if (!pair_exists) {
            ++iflavor;
            flavor({1, iflavor}) = key_species_list({1, i});
            flavor({2, iflavor}) = key_species_list({2, i});
        }
    }
}

int key_species_pair2flavor(
    const Array<int, 2> &flavor,
    const Array<int, 1> &key_species_pair)
{
    // Search for match.
    for (int iflav = 1; iflav <= flavor.dim(2); ++iflav)
        if (key_species_pair({1}) == flavor({1, iflav})
            && key_species_pair({2}) == flavor({2, iflav}))
            return iflav;

    // No match found.
    return -1;
}

void create_gpoint_flavor(
    const Array<int, 3> &key_species,
    const Array<int, 1> &gpt2band,
    const Array<int, 2> &flavor,
    Array<int, 2> &gpoint_flavor)
{
    const int ngpt = gpt2band.dim(1);
    gpoint_flavor.set_dims({2, ngpt});

    for (int igpt = 1; igpt <= ngpt; ++igpt)
        for (int iatm = 1; iatm <= 2; ++iatm) {
            int pair_1 = key_species({1, iatm, gpt2band({igpt})});
            int pair_2 = key_species({2, iatm, gpt2band({igpt})});

            // Rewrite species pair.
            Array<int, 1> rewritten_pair({2});
            if (pair_1 == 0 && pair_2 == 0) {
                rewritten_pair({1}) = 2;
                rewritten_pair({2}) = 2;
            } else {
                rewritten_pair({1}) = pair_1;
                rewritten_pair({2}) = pair_2;
            }

            // Write the output.
            gpoint_flavor({iatm, igpt}) =
                key_species_pair2flavor(flavor, rewritten_pair);
        }
}

} // namespace

Constants::Constants(string_view filename)
{
    Netcdf_file coef_nc(string(filename), Netcdf_mode::Read);

    // Read k-distribution information.
    const int n_temps = coef_nc.get_dimension_size("temperature");
    const int n_press = coef_nc.get_dimension_size("pressure");
    const int n_absorbers = coef_nc.get_dimension_size("absorber");
    const int n_char = coef_nc.get_dimension_size("string_len");
    const int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
    const int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
    const int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
    const int n_layers = coef_nc.get_dimension_size("atmos_layer");
    const int n_bnds = coef_nc.get_dimension_size("bnd");
    const int n_gpts = coef_nc.get_dimension_size("gpt");
    const int n_pairs = coef_nc.get_dimension_size("pair");
    const int n_minor_absorber_intervals_lower =
        coef_nc.get_dimension_size("minor_absorber_intervals_lower");
    const int n_minor_absorber_intervals_upper =
        coef_nc.get_dimension_size("minor_absorber_intervals_upper");
    const int n_contributors_lower =
        coef_nc.get_dimension_size("contributors_lower");
    const int n_contributors_upper =
        coef_nc.get_dimension_size("contributors_upper");

    // Read gas names.
    gas_names = Array<std::string, 1>(
        get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true),
        {n_absorbers});

    Array<int, 3> key_species(
        coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
        {2, n_layers, n_bnds});
    band_lims_wvn = Array<REAL, 2>(
        coef_nc.get_variable<REAL>("bnd_limits_wavenumber", {n_bnds, 2}),
        {2, n_bnds});
    band2gpt = Array<int, 2>(
        coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}),
        {2, n_bnds});
    press_ref = Array<REAL, 1>(
        coef_nc.get_variable<REAL>("press_ref", {n_press}),
        {n_press});
    temp_ref = Array<REAL, 1>(
        coef_nc.get_variable<REAL>("temp_ref", {n_temps}),
        {n_temps});

    // Make a map between g-points and bands.
    this->gpt2band.set_dims({band2gpt.max()});
    for (int iband = 1; iband <= band2gpt.dim(2); ++iband)
        for (int i = band2gpt({1, iband}); i <= band2gpt({2, iband}); ++i)
            this->gpt2band({i}) = iband;

    REAL temp_ref_p =
        coef_nc.get_variable<REAL>("absorption_coefficient_ref_P");
    REAL temp_ref_t =
        coef_nc.get_variable<REAL>("absorption_coefficient_ref_T");
    REAL press_ref_trop = coef_nc.get_variable<REAL>("press_ref_trop");

    kminor_lower = Array<REAL, 3>(
        coef_nc.get_variable<REAL>(
            "kminor_lower",
            {n_temps, n_mixingfracs, n_contributors_lower}),
        {n_contributors_lower, n_mixingfracs, n_temps});
    kminor_upper = Array<REAL, 3>(
        coef_nc.get_variable<REAL>(
            "kminor_upper",
            {n_temps, n_mixingfracs, n_contributors_upper}),
        {n_contributors_upper, n_mixingfracs, n_temps});

    Array<std::string, 1> gas_minor(
        get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
        {n_minorabsorbers});

    Array<std::string, 1> identifier_minor(
        get_variable_string(
            "identifier_minor",
            {n_minorabsorbers},
            coef_nc,
            n_char),
        {n_minorabsorbers});

    Array<std::string, 1> minor_gases_lower(
        get_variable_string(
            "minor_gases_lower",
            {n_minor_absorber_intervals_lower},
            coef_nc,
            n_char),
        {n_minor_absorber_intervals_lower});
    Array<std::string, 1> minor_gases_upper(
        get_variable_string(
            "minor_gases_upper",
            {n_minor_absorber_intervals_upper},
            coef_nc,
            n_char),
        {n_minor_absorber_intervals_upper});

    minor_limits_gpt_lower = Array<int, 2>(
        coef_nc.get_variable<int>(
            "minor_limits_gpt_lower",
            {n_minor_absorber_intervals_lower, n_pairs}),
        {n_pairs, n_minor_absorber_intervals_lower});
    minor_limits_gpt_upper = Array<int, 2>(
        coef_nc.get_variable<int>(
            "minor_limits_gpt_upper",
            {n_minor_absorber_intervals_upper, n_pairs}),
        {n_pairs, n_minor_absorber_intervals_upper});

    minor_scales_with_density_lower = Array<BOOL, 1>(
        coef_nc.get_variable<BOOL>(
            "minor_scales_with_density_lower",
            {n_minor_absorber_intervals_lower}),
        {n_minor_absorber_intervals_lower});
    minor_scales_with_density_upper = Array<BOOL, 1>(
        coef_nc.get_variable<BOOL>(
            "minor_scales_with_density_upper",
            {n_minor_absorber_intervals_upper}),
        {n_minor_absorber_intervals_upper});

    scale_by_complement_lower = Array<BOOL, 1>(
        coef_nc.get_variable<BOOL>(
            "scale_by_complement_lower",
            {n_minor_absorber_intervals_lower}),
        {n_minor_absorber_intervals_lower});
    scale_by_complement_upper = Array<BOOL, 1>(
        coef_nc.get_variable<BOOL>(
            "scale_by_complement_upper",
            {n_minor_absorber_intervals_upper}),
        {n_minor_absorber_intervals_upper});

    Array<std::string, 1> scaling_gas_lower(
        get_variable_string(
            "scaling_gas_lower",
            {n_minor_absorber_intervals_lower},
            coef_nc,
            n_char),
        {n_minor_absorber_intervals_lower});
    Array<std::string, 1> scaling_gas_upper(
        get_variable_string(
            "scaling_gas_upper",
            {n_minor_absorber_intervals_upper},
            coef_nc,
            n_char),
        {n_minor_absorber_intervals_upper});

    kminor_start_lower = Array<int, 1>(
        coef_nc.get_variable<int>(
            "kminor_start_lower",
            {n_minor_absorber_intervals_lower}),
        {n_minor_absorber_intervals_lower});
    kminor_start_upper = Array<int, 1>(
        coef_nc.get_variable<int>(
            "kminor_start_upper",
            {n_minor_absorber_intervals_upper}),
        {n_minor_absorber_intervals_upper});

    vmr_ref = Array<REAL, 3>(
        coef_nc.get_variable<REAL>(
            "vmr_ref",
            {n_temps, n_extabsorbers, n_layers}),
        {n_layers, n_extabsorbers, n_temps});

    kmajor = Array<REAL, 4>(
        coef_nc.get_variable<REAL>(
            "kmajor",
            {n_temps, n_press + 1, n_mixingfracs, n_gpts}),
        {n_gpts, n_mixingfracs, n_press + 1, n_temps});

    // Is it really LW if so read these variables as well.
    if (coef_nc.variable_exists("totplnk")) {
        int n_internal_sourcetemps =
            coef_nc.get_dimension_size("temperature_Planck");

        Array<REAL, 2> totplnk(
            coef_nc.get_variable<REAL>(
                "totplnk",
                {n_bnds, n_internal_sourcetemps}),
            {n_internal_sourcetemps, n_bnds});
        Array<REAL, 4> planck_frac(
            coef_nc.get_variable<REAL>(
                "plank_fraction",
                {n_temps, n_press + 1, n_mixingfracs, n_gpts}),
            {n_gpts, n_mixingfracs, n_press + 1, n_temps});

        totplnk_delta = (temp_ref_max - temp_ref_min) / (totplnk.dim(1) - 1);
    } else {
        Array<REAL, 3> rayl_lower(
            coef_nc.get_variable<REAL>(
                "rayl_lower",
                {n_temps, n_mixingfracs, n_gpts}),
            {n_gpts, n_mixingfracs, n_temps});
        Array<REAL, 3> rayl_upper(
            coef_nc.get_variable<REAL>(
                "rayl_upper",
                {n_temps, n_mixingfracs, n_gpts}),
            {n_gpts, n_mixingfracs, n_temps});

        solar_source_quiet = Array<REAL, 1>(
            coef_nc.get_variable<REAL>("solar_source_quiet", {n_gpts}),
            {n_gpts});
        solar_source_facular = Array<REAL, 1>(
            coef_nc.get_variable<REAL>("solar_source_facular", {n_gpts}),
            {n_gpts});
        solar_source_sunspot = Array<REAL, 1>(
            coef_nc.get_variable<REAL>("solar_source_sunspot", {n_gpts}),
            {n_gpts});

        REAL tsi = coef_nc.get_variable<REAL>("tsi_default");
        REAL mg_index = coef_nc.get_variable<REAL>("mg_default");
        REAL sb_index = coef_nc.get_variable<REAL>("sb_default");
    }
}

void Constants::init_abs_coeffs(
    const Array<int, 3> &key_species,
    const Array<int, 2> &band2gpt,
    const Array<REAL, 2> &band_lims_wavenum,
    const Array<REAL, 1> &press_ref,
    const Array<REAL, 1> &temp_ref,
    const REAL press_ref_trop,
    const REAL temp_ref_p,
    const REAL temp_ref_t,
    const Array<std::string, 1> &gas_minor,
    const Array<std::string, 1> &identifier_minor,
    const Array<std::string, 1> &minor_gases_lower,
    const Array<std::string, 1> &minor_gases_upper,
    const Array<std::string, 1> &scaling_gas_lower,
    const Array<std::string, 1> &scaling_gas_upper,
    const Array<REAL, 3> &rayl_lower,
    const Array<REAL, 3> &rayl_upper)
{
    // Which gases known to the gas optics are present in the host model
    // (available_gases)?
    std::vector<std::string> gas_names_to_use;

    for (const std::string &s : gas_names.v())
        if (std::find(wrf_gases.begin(), wrf_gases.end(), s) != wrf_gases.end())
            gas_names_to_use.push_back(s);

    // Now the number of gases is the union of those known to the k-distribution
    // and provided by the host model.
    const int n_gas = gas_names_to_use.size();
    Array<std::string, 1> gas_names_this(std::move(gas_names_to_use), {n_gas});
    this->gas_names = gas_names_this;

    // Initialize the gas optics object, keeping only those gases known to the
    // gas optics and also present in the host model.
    // Add an offset to the indexing to interface the negative ranging of
    // fortran.
    Array<REAL, 3> vmr_ref_red({vmr_ref.dim(1), n_gas + 1, vmr_ref.dim(3)});
    vmr_ref_red.set_offsets({0, -1, 0});

    // Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    for (int i1 = 1; i1 <= vmr_ref_red.dim(1); ++i1)
        for (int i3 = 1; i3 <= vmr_ref_red.dim(3); ++i3)
            vmr_ref_red({i1, 0, i3}) = vmr_ref({i1, 1, i3});

    for (int i = 1; i <= n_gas; ++i) {
        int idx = i; // KFAF: WTF?
        for (int i1 = 1; i1 <= vmr_ref_red.dim(1); ++i1)
            for (int i3 = 1; i3 <= vmr_ref_red.dim(3); ++i3)
                vmr_ref_red({i1, i, i3}) =
                    vmr_ref({i1, idx + 1, i3}); // CvH: why +1?
    }

    this->vmr_ref = std::move(vmr_ref_red);

    // Reduce minor arrays so variables only contain minor gases that are
    // available. Reduce size of minor Arrays.
    Array<std::string, 1> minor_gases_lower_red;
    Array<std::string, 1> scaling_gas_lower_red;
    Array<std::string, 1> minor_gases_upper_red;
    Array<std::string, 1> scaling_gas_upper_red;

    reduce_minor_arrays(
        gas_names,
        gas_minor,
        identifier_minor,
        kminor_lower,
        minor_gases_lower,
        minor_limits_gpt_lower,
        minor_scales_with_density_lower,
        scaling_gas_lower,
        scale_by_complement_lower,
        kminor_start_lower,
        this->kminor_lower,
        minor_gases_lower_red,
        this->minor_limits_gpt_lower,
        this->minor_scales_with_density_lower,
        scaling_gas_lower_red,
        this->scale_by_complement_lower,
        this->kminor_start_lower);

    reduce_minor_arrays(
        gas_names,
        gas_minor,
        identifier_minor,
        kminor_upper,
        minor_gases_upper,
        minor_limits_gpt_upper,
        minor_scales_with_density_upper,
        scaling_gas_upper,
        scale_by_complement_upper,
        kminor_start_upper,
        this->kminor_upper,
        minor_gases_upper_red,
        this->minor_limits_gpt_upper,
        this->minor_scales_with_density_upper,
        scaling_gas_upper_red,
        this->scale_by_complement_upper,
        this->kminor_start_upper);

    // Arrays not reduced by the presence, or lack thereof, of a gas
    this->press_ref = press_ref;
    this->temp_ref = temp_ref;

    // Reshaping according to new dimension ordering since v1.5
    this->kmajor.set_dims(
        {kmajor.dim(4), kmajor.dim(2), kmajor.dim(3), kmajor.dim(1)});
    for (int i4 = 1; i4 <= this->kmajor.dim(4); ++i4)
        for (int i3 = 1; i3 <= this->kmajor.dim(3); ++i3)
            for (int i2 = 1; i2 <= this->kmajor.dim(2); ++i2)
                for (int i1 = 1; i1 <= this->kmajor.dim(1); ++i1)
                    this->kmajor({i1, i2, i3, i4}) = kmajor({i4, i2, i3, i1});

    // Reshaping according to new 1.5 release.
    // Create a new vector that consists of rayl_lower and rayl_upper stored in
    // one variable.
    if (rayl_lower.size() > 0) {
        this->krayl.set_dims(
            {rayl_lower.dim(3), rayl_lower.dim(2), rayl_lower.dim(1), 2});
        for (int i3 = 1; i3 <= this->krayl.dim(3); ++i3)
            for (int i2 = 1; i2 <= this->krayl.dim(2); ++i2)
                for (int i1 = 1; i1 <= this->krayl.dim(1); ++i1) {
                    this->krayl({i1, i2, i3, 1}) = rayl_lower({i3, i2, i1});
                    this->krayl({i1, i2, i3, 2}) = rayl_upper({i3, i2, i1});
                }
    }

    // ---- post processing ----
    //  creates log reference pressure
    this->press_ref_log = this->press_ref;
    for (int i1 = 1; i1 <= this->press_ref_log.dim(1); ++i1)
        this->press_ref_log({i1}) = std::log(this->press_ref_log({i1}));

    // log scale of reference pressure
    this->press_ref_trop_log = std::log(press_ref_trop);

    // Get index of gas (if present) for determining col_gas
    create_idx_minor(
        this->gas_names,
        gas_minor,
        identifier_minor,
        minor_gases_lower_red,
        this->idx_minor_lower);
    create_idx_minor(
        this->gas_names,
        gas_minor,
        identifier_minor,
        minor_gases_upper_red,
        this->idx_minor_upper);

    // Get index of gas (if present) that has special treatment in density
    // scaling
    create_idx_minor_scaling(
        this->gas_names,
        scaling_gas_lower_red,
        this->idx_minor_scaling_lower);
    create_idx_minor_scaling(
        this->gas_names,
        scaling_gas_upper_red,
        this->idx_minor_scaling_upper);

    // Create flavor list.
    // Reduce (remap) key_species list; checks that all key gases are present in
    // incoming
    Array<int, 3> key_species_red;
    Array<BOOL, 1> key_species_present_init;

    create_key_species_reduce(
        gas_names,
        this->gas_names,
        key_species,
        key_species_red,
        key_species_present_init);

    check_key_species_present_init(gas_names, key_species_present_init);

    // create flavor list
    create_flavor(key_species_red, this->flavor);

    // create gpoint flavor list
    create_gpoint_flavor(
        key_species_red,
        this->gpt2band,
        this->flavor,
        this->gpoint_flavor);

    // minimum, maximum reference temperature, pressure -- assumes low-to-high
    // ordering for T, high-to-low ordering for p
    this->temp_ref_min = this->temp_ref({1});
    this->temp_ref_max = this->temp_ref({temp_ref.dim(1)});
    this->press_ref_min = this->press_ref({press_ref.dim(1)});
    this->press_ref_max = this->press_ref({1});

    // creates press_ref_log, temp_ref_delta
    this->press_ref_log_delta =
        (std::log(this->press_ref_min) - std::log(this->press_ref_max))
        / (this->press_ref.dim(1) - 1);
    this->temp_ref_delta =
        (this->temp_ref_max - this->temp_ref_min) / (this->temp_ref.dim(1) - 1);

    // Which species are key in one or more bands?
    // this->flavor is an index into this->gas_names
    // if (allocated(this%is_key)) deallocate(this%is_key) ! Shouldn't ever
    // happen...
    Array<int, 1> is_key({this->gas_names.dim(1)}); // CvH bool, defaults to 0.?

    for (int j = 1; j <= this->flavor.dim(2); ++j)
        for (int i = 1; i <= this->flavor.dim(1); ++i)
            if (this->flavor({i, j}) != 0)
                is_key({this->flavor({i, j})}) = true;

    this->is_key = is_key;
}
