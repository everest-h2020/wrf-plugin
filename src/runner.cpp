#include "rrtmg.h"
#include "rrtmgp/Constants.h"
#include "rrtmgp/Netcdf_interface.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace std;
using namespace rrtmgp;

static Array<REAL, 1> T_lay;
static Array<REAL, 1> p_lay;
static Array<REAL, 1> n_d;
static Array<REAL, 2> n_prime;

namespace {

void load_test_data(string_view filename)
{
    Netcdf_file file(string(filename), Netcdf_mode::Read);

    const auto n_lay = file.get_dimension_size("tlay_d2");
    const auto n_col = file.get_dimension_size("tlay_d1");
    const auto n_gas = file.get_dimension_size("col_gas_d3");

    Array<REAL, 2> tlay(
        file.get_variable<REAL>("tlay", {n_lay, n_col}),
        {n_col, n_lay});
    Array<REAL, 2> play(
        file.get_variable<REAL>("play", {n_lay, n_col}),
        {n_col, n_lay});
    Array<REAL, 3> col_gas(
        file.get_variable<REAL>("col_gas", {n_gas, n_lay, n_col}),
        {n_col, n_lay, n_gas});

    // clang-format off
    // std::cout
    //     << std::setw(5) << "i_eta"
    //     << std::setw(14) << "T_lay (K)"
    //     << std::setw(14) << "p_lay (Pa)"
    //     << std::setw(14) << "n[0]"
    //     << std::setw(14) << "n[1]"
    //     << std::setw(14) << "n[2]"
    //     << std::setw(14) << "n[3]"
    //     << std::setw(14) << "n[4]"
    //     << std::setw(14) << "n[5]"
    //     << std::setw(14) << "n[6]"
    //     << std::endl;
    // for (int ilay = 0; ilay < n_lay; ++ilay)
    // {
    //     std::cout
    //         << std::setw(5) << ilay
    //         << std::fixed
    //         << std::setw(14) << tlay({1, ilay + 1})
    //         << std::setw(14) << play({1, ilay + 1})
    //         << std::scientific
    //         << std::setw(14) << col_gas({1, ilay + 1, 1})
    //         << std::setw(14) << col_gas({1, ilay + 1, 2})
    //         << std::setw(14) << col_gas({1, ilay + 1, 3})
    //         << std::setw(14) << col_gas({1, ilay + 1, 4})
    //         << std::setw(14) << col_gas({1, ilay + 1, 5})
    //         << std::setw(14) << col_gas({1, ilay + 1, 6})
    //         << std::setw(14) << col_gas({1, ilay + 1, 7})
    //         << std::endl;
    // }
    // clang-format on

    T_lay.set_dims({n_lay});
    p_lay.set_dims({n_lay});
    n_d.set_dims({n_lay});
    n_prime.set_dims({n_gas - 1, n_lay});

    int i_col = 1;
    for (int i_lay = 1; i_lay <= n_lay; ++i_lay) {
        T_lay({i_lay}) = tlay({i_col, i_lay});
        p_lay({i_lay}) = play({i_col, i_lay});
        const auto dry = col_gas({i_col, i_lay, 1});
        n_d({i_lay}) = dry;
        for (int i_gas = 1; i_gas < n_gas; ++i_gas)
            n_prime({i_gas, i_lay}) = col_gas({i_col, i_lay, i_gas + 1}) / dry;
    }
}

} // namespace

int main(int argc, const char** argv)
{
    Constants constants(
        "/home/friebel/everest/rte-rrtmgp-cpp/rte-rrtmgp/rrtmgp/data/"
        "rrtmgp-data-sw-g112-210809.nc");

    load_test_data(
        "/home/friebel/everest/wrf_kernels/rrtmgp-data/"
        "dump_interpolate_in_0_sw.nc");

    const auto n_lay = T_lay.dim(1);
    const auto n_gpt = constants.gpt2band.dim(1);
    const auto n_gas = n_prime.dim(1);
    const auto cp_d = 1.004500e+03;

    plugin_rrtmg_sw_init(1, n_lay, n_gpt, n_gas, cp_d);

    Array<REAL, 2> tau_gas({n_gpt, n_lay});
    Array<REAL, 2> tau_rayl({n_gpt, n_lay});

    plugin_rrtmg_sw_taumol(
        n_lay,
        n_gpt,
        n_gas,
        T_lay.data.data(),
        p_lay.data.data(),
        n_d.data.data(),
        n_prime.data.data(),
        tau_gas.data.data(),
        tau_rayl.data.data());

    return EXIT_SUCCESS;
}
