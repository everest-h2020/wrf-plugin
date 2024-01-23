#include "fxx/Memory.h"
#include "rrtmg.h"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <netcdf>

using namespace fxx;
using namespace std;
using namespace netCDF;

static tensor<REAL, 1> T_lay;
static tensor<REAL, 1> p_lay;
static tensor<REAL, 1> n_d;
static tensor<REAL, 2> r_abs;

namespace {

void load_test_data(string_view filename)
{
    NcFile file(string(filename), NcFile::read);

    const auto N_lay = file.getDim("tlay_d2").getSize();
    const auto N_gas = file.getDim("col_gas_d3").getSize();

    T_lay = tensor<REAL, 1>(N_lay);
    p_lay = tensor<REAL, 1>(N_lay);
    n_d = tensor<REAL, 1>(N_lay);
    r_abs = tensor<REAL, 2>(N_lay, N_gas);

    std::cerr << T_lay.size() << "\n";
    file.getVar("tlay")
        .getVar({0, 0}, {N_lay, 1}, {1, 1}, {1, 0}, T_lay.data());
    file.getVar("play").getVar({0, 0}, {N_lay, 1}, p_lay.data());
    file.getVar("col_gas").getVar({0, 0, 0}, {1, N_lay, 1}, n_d.data());
    file.getVar("col_gas").getVar(
        {1, 0, 0},
        {N_gas - 1, N_lay, 1},
        {1, 1, 1},
        {1, static_cast<ptrdiff_t>(N_gas), 0},
        r_abs.data());

    for (index_t i_lay = 0; i_lay < index_t(N_lay); ++i_lay)
        for (index_t i_gas = 0; i_gas < index_t(N_gas); ++i_gas)
            r_abs(i_lay, i_gas) /= n_d(i_lay);
}

} // namespace

int main(int, const char **)
{
    load_test_data("./data/wrf-input.nc");

    const auto N_lay = T_lay.layout().hrect().sizes()[0];
    // const auto N_gpt = 112;
    const auto N_gpt = 224;
    const auto N_gas = r_abs.layout().hrect().sizes()[1];
    const auto cp_d = 1.004500e+03;

    plugin_rrtmg_sw_init(1, N_lay, N_gpt, N_gas, cp_d);

    tensor<REAL, 2> tau_gas(N_gpt, N_lay);
    tensor<REAL, 2> tau_rayl(N_gpt, N_lay);

    plugin_rrtmg_sw_taumol(
        N_lay,
        N_gpt,
        N_gas,
        T_lay.data(),
        p_lay.data(),
        n_d.data(),
        r_abs.data(),
        tau_gas.data(),
        tau_rayl.data());

    {
        NcFile dump("output.nc", NcFile::replace);
        auto dim_gpt = dump.addDim("N_gpt", N_gpt);
        auto dim_lay = dump.addDim("N_lay", N_lay);

        auto var_gas =
            dump.addVar("tau_gas", NcType::nc_FLOAT, {dim_gpt, dim_lay});
        auto var_rayl =
            dump.addVar("tau_rayl", NcType::nc_FLOAT, {dim_gpt, dim_lay});

        var_gas.putVar(tau_gas.data());
        var_rayl.putVar(tau_rayl.data());
    }

    return EXIT_SUCCESS;
}
