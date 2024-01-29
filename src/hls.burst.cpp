#include "rrtmg/kernels.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

using namespace std;
using namespace rrtmg;

#include "data.inc"

namespace {

static constexpr REAL C_TINY = numeric_limits<REAL>::min();

} // namespace

static void int_frac(REAL x, index_t &j, REAL &f)
{
    REAL int_part;
    f = std::modf(x, &int_part);
    j = static_cast<index_t>(int_part);
}

static void compute_T_prime(
    const REAL T[N_CELL],
    index_t j_T[N_CELL],
    REAL f_T[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS pipeline II = 1
        const auto T_prime = (T[i_cell] - C_MIN_T_REF) / C_DELTA_T_REF;
        int_frac(T_prime, j_T[i_cell], f_T[i_cell]);
    }
}

static void compute_p_prime(
    const REAL p[N_CELL],
    index_t j_strato[N_CELL],
    index_t j_p[N_CELL],
    REAL f_p[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS pipeline II = 1
        j_strato[i_cell] = (p[i_cell] < C_P_TROPO) ? 1 : 0;
        const auto p_prime =
            (std::log(p[i_cell]) - C_LOG_MAX_P_REF) / C_DELTA_LOG_P_REF;
        int_frac(p_prime, j_p[i_cell], f_p[i_cell]);
    }
}

static void compute_k(const REAL T[N_CELL], const REAL p[N_CELL], REAL k[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS pipeline II = 1
        k[i_cell] = REAL(0.01) * p[i_cell] / T[i_cell];
    }
}

static void compute_eta_bnd(
    const index_t i_bnd,
    const index_t j_T[N_CELL],
    const index_t j_strato[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    REAL r_mix[N_CELL][2],
    index_t j_eta[N_CELL][2],
    REAL f_eta[N_CELL][2])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS PIPELINE II = 1
        const auto i_strato = j_strato[i_cell];
        const auto i_flav = C_BND_TO_FLAV[i_strato][i_bnd];
        const auto i_g_0 = C_FLAV_TO_ABS[i_flav][0];
        const auto i_g_1 = C_FLAV_TO_ABS[i_flav][1];

        const auto r_g_0 = i_g_0 >= 0 ? r_gas[i_g_0][i_cell] : REAL(1);
        const auto r_g_1 = i_g_1 >= 0 ? r_gas[i_g_1][i_cell] : REAL(1);
        const auto i_T = j_T[i_cell];

        for (index_t dT = 0; dT < 2; ++dT) {
#pragma HLS UNROLL
            const auto r_eta_half = C_ETA_HALF[i_strato][i_flav][i_T + dT];
            const auto f_mix = r_g_0 + r_eta_half * r_g_1;
            r_mix[i_cell][dT] = f_mix * n_d[i_cell];
            const auto alpha = f_mix > C_TINY ? (r_g_0 / f_mix) : REAL(0.5);
            const auto eta = alpha * (C_N_ETA - 1);
            int_frac(eta, j_eta[i_cell][dT], f_eta[i_cell][dT]);
        }
    }
}

static void tau_major_bnd(
    const index_t i_bnd,
    const index_t j_T[N_CELL],
    const REAL f_T[N_CELL],
    const index_t j_strato[N_CELL],
    const index_t j_p[N_CELL],
    const REAL f_p[N_CELL],
    const REAL r_mix[N_CELL][2],
    const index_t j_eta[N_CELL][2],
    const REAL f_eta[N_CELL][2],
    const REAL k_major[N_BND][14][9][60][N_GPB],
    REAL tau_g[N_CELL][N_GPB])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS PIPELINE II = 1
        const auto i_strato = j_strato[i_cell];
        const auto i_T = j_T[i_cell];
        const auto b_T = f_T[i_cell];
        const auto i_p = j_p[i_cell];
        const auto b_p = f_p[i_cell];

        REAL accu[N_GPB] = {0};
#pragma HLS ARRAY_PARTITION variable = accu type = complete

        for (index_t dT = 0; dT < 2; ++dT) {
#pragma HLS UNROLL
            const auto a_T = dT == 1 ? b_T : REAL(1) - b_T;
            const auto a_mix = r_mix[i_cell][dT];
            for (index_t deta = 0; deta < 2; ++deta) {
#pragma HLS UNROLL
                const auto i_eta = j_eta[i_cell][dT];
                const auto b_eta = f_eta[i_cell][dT];
                const auto a_eta = deta == 1 ? b_eta : REAL(1) - b_eta;
                const auto a_minor = a_T * a_eta * a_mix;
                for (index_t dp = 0; dp < 2; ++dp) {
#pragma HLS UNROLL
                    const auto a_p = dp == 1 ? b_p : REAL(1) - b_p;
                    for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
                        accu[i_gpb] += k_major[i_bnd][i_T + dT][i_eta + deta]
                                              [i_p + dp + i_strato][i_gpb]
                                       * a_minor * a_p;
                    }
                }
            }
        }

        for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
            tau_g[i_cell][i_gpb] = accu[i_gpb];
        }
    }
}

static void tau_rayleigh_bnd(
    const index_t i_bnd,
    const index_t j_T[N_CELL],
    const REAL f_T[N_CELL],
    const index_t j_strato[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    const index_t j_eta[N_CELL][2],
    const REAL f_eta[N_CELL][2],
    REAL tau_r[N_CELL][N_GPB])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS PIPELINE II = 1
        const auto i_strato = j_strato[i_cell];
        const auto i_T = j_T[i_cell];
        const auto b_T = f_T[i_cell];
        const auto a_wet = n_d[i_cell] * (REAL(1) + r_gas[0][i_cell]);

        REAL accu[N_GPB] = {0};
#pragma HLS ARRAY_PARTITION variable = accu type = complete

        for (index_t dT = 0; dT < 2; ++dT) {
#pragma HLS UNROLL
            const auto a_T = dT == 1 ? b_T : REAL(1) - b_T;
            for (index_t deta = 0; deta < 2; ++deta) {
#pragma HLS UNROLL
                const auto i_eta = j_eta[i_cell][dT];
                const auto b_eta = f_eta[i_cell][dT];
                const auto a_eta = deta == 1 ? b_eta : REAL(1) - b_eta;
                for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
                    accu[i_gpb] += C_K_RAYLEIGH[i_bnd][i_strato][i_T + dT]
                                               [i_eta + deta][i_gpb]
                                   * a_T * a_eta * a_wet;
                }
            }
        }

        for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
            tau_r[i_cell][i_gpb] = accu[i_gpb];
        }
    }
}

void compute_dry_fact(const REAL r_h2o[N_CELL], REAL dry_fact[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS PIPELINE II = 1
        dry_fact[i_cell] = REAL(1) / (1 + r_h2o[i_cell]);
    }
}

static void tau_minor_bnd(
    const index_t i_bnd,
    const index_t j_T[N_CELL],
    const REAL f_T[N_CELL],
    const index_t j_strato[N_CELL],
    const REAL k[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    const REAL dry_fact[N_CELL],
    const index_t j_eta[N_CELL][2],
    const REAL f_eta[N_CELL][2],
    REAL tau_g[N_CELL][N_GPB])
{
    const auto N_MPB = C_MINOR_PER_BND[i_bnd][0] + C_MINOR_PER_BND[i_bnd][1];
    for (index_t i_mpb = 0; i_mpb < N_MPB; ++i_mpb) {
        const auto i_minor = C_MINOR_START[i_bnd] + i_mpb;
        const auto i_strato = (i_mpb >= C_MINOR_PER_BND[i_bnd][0]) ? 1 : 0;
        const auto i_abs = C_MINOR_TO_ABS[i_minor];
        const auto scale_by = C_MINOR_SCALE_BY[i_minor];

        for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS PIPELINE II = 1
            const auto i_T = j_T[i_cell];
            const auto b_T = f_T[i_cell];
            auto r_abs = j_strato[i_cell] != i_strato
                             ? r_gas[i_abs][i_cell] * n_d[i_cell]
                             : REAL(0);
            if (scale_by != 0) {
                r_abs *= k[i_cell];
                const auto i_by = std::abs(scale_by) - 2;
                if (i_by >= 0) {
                    const auto gas_fact =
                        r_gas[i_by][i_cell] * dry_fact[i_cell];
                    r_abs *= scale_by < 0 ? (REAL(1) - gas_fact) : gas_fact;
                }
            }

            REAL accu[N_GPB] = {0};
#pragma HLS ARRAY_PARTITION variable = accu type = complete

            for (index_t dT = 0; dT < 2; ++dT) {
#pragma HLS UNROLL
                const auto a_T = dT == 1 ? b_T : REAL(1) - b_T;
                const auto a_dry = a_T * r_abs;
                for (index_t deta = 0; deta < 2; ++deta) {
#pragma HLS UNROLL
                    const auto i_eta = j_eta[i_cell][dT];
                    const auto b_eta = f_eta[i_cell][dT];
                    const auto a_eta = deta == 1 ? b_eta : REAL(1) - b_eta;
                    const auto alpha = a_eta * a_dry;

                    for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
                        accu[i_gpb] +=
                            C_K_MINOR[i_minor][i_T + dT][i_eta + deta][i_gpb]
                            * alpha;
                    }
                }
            }

            for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
                tau_g[i_cell][i_gpb] = accu[i_gpb];
            }
        }
    }
}

template<class T>
static void bcast3_CELL(
    const T in[N_CELL],
    T out0[N_CELL],
    T out1[N_CELL],
    T out2[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
        out0[i_cell] = in[i_cell];
        out1[i_cell] = in[i_cell];
        out2[i_cell] = in[i_cell];
    }
}

template<class T>
static void bcast3_1_GAS_CELL(
    const T in[N_GAS][N_CELL],
    T out0[N_GAS][N_CELL],
    T out1[N_GAS][N_CELL],
    T out2[N_GAS][N_CELL],
    T out3[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
        for (index_t i_gas = 0; i_gas < N_GAS; ++i_gas) {
            const auto r_gas = in[i_gas][i_cell];
            out0[i_gas][i_cell] = r_gas;
            out1[i_gas][i_cell] = r_gas;
            out2[i_gas][i_cell] = r_gas;
            if (i_gas == 0) out3[i_cell] = r_gas;
        }
    }
}

template<class T>
static void bcast4_CELL(
    const T in[N_CELL],
    T out0[N_CELL],
    T out1[N_CELL],
    T out2[N_CELL],
    T out3[N_CELL])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
        out0[i_cell] = in[i_cell];
        out1[i_cell] = in[i_cell];
        out2[i_cell] = in[i_cell];
        out3[i_cell] = in[i_cell];
    }
}

template<class T>
static void bcast3_CELL_2(
    const T in[N_CELL][2],
    T out0[N_CELL][2],
    T out1[N_CELL][2],
    T out2[N_CELL][2])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
        for (index_t dT = 0; dT < 2; ++dT) {
            out0[i_cell][dT] = in[i_cell][dT];
            out1[i_cell][dT] = in[i_cell][dT];
            out2[i_cell][dT] = in[i_cell][dT];
        }
    }
}

static void tau_sum(
    const REAL in0[N_CELL][N_GPB],
    const REAL in1[N_CELL][N_GPB],
    REAL out[N_CELL][N_GPB])
{
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
#pragma HLS PIPELINE II = 1
        for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
#pragma HLS UNROLL
            out[i_cell][i_gpb] = in0[i_cell][i_gpb] + in1[i_cell][i_gpb];
        }
    }
}

static void tau_df(
    const index_t j_T[N_CELL],
    const REAL f_T[N_CELL],
    const index_t j_strato[N_CELL],
    const index_t j_p[N_CELL],
    const REAL f_p[N_CELL],
    const REAL k[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    const REAL k_major[N_BND][14][9][60][N_GPB],
    REAL tau_g[N_BND][N_CELL][N_GPB],
    REAL tau_r[N_BND][N_CELL][N_GPB])
{
    for (index_t i_bnd = 0; i_bnd < C_N_BND; ++i_bnd) {
#pragma HLS DATAFLOW

        index_t j_T_1[N_CELL];
        index_t j_T_2[N_CELL];
        index_t j_T_3[N_CELL];
        index_t j_T_4[N_CELL];
        bcast4_CELL(j_T, j_T_1, j_T_2, j_T_3, j_T_4);

        REAL f_T_1[N_CELL];
        REAL f_T_2[N_CELL];
        REAL f_T_3[N_CELL];
        bcast3_CELL(f_T, f_T_1, f_T_2, f_T_3);

        index_t j_strato_1[N_CELL];
        index_t j_strato_2[N_CELL];
        index_t j_strato_3[N_CELL];
        index_t j_strato_4[N_CELL];
        bcast4_CELL(j_strato, j_strato_1, j_strato_2, j_strato_3, j_strato_4);

        REAL n_d_1[N_CELL];
        REAL n_d_2[N_CELL];
        REAL n_d_3[N_CELL];
        bcast3_CELL(n_d, n_d_1, n_d_2, n_d_3);

        REAL r_gas_1[N_GAS][N_CELL];
        REAL r_gas_2[N_GAS][N_CELL];
        REAL r_gas_3[N_GAS][N_CELL];
        REAL r_h2o[N_CELL];
        bcast3_1_GAS_CELL(r_gas, r_gas_1, r_gas_2, r_gas_3, r_h2o);

        REAL r_mix[N_CELL][2];
        index_t j_eta[N_CELL][2];
        REAL f_eta[N_CELL][2];
        compute_eta_bnd(
            i_bnd,
            j_T_1,
            j_strato_1,
            n_d_1,
            r_gas_1,
            r_mix,
            j_eta,
            f_eta);

        index_t j_eta_1[N_CELL][2];
        index_t j_eta_2[N_CELL][2];
        index_t j_eta_3[N_CELL][2];
        bcast3_CELL_2(j_eta, j_eta_1, j_eta_2, j_eta_3);

        REAL f_eta_1[N_CELL][2];
        REAL f_eta_2[N_CELL][2];
        REAL f_eta_3[N_CELL][2];
        bcast3_CELL_2(f_eta, f_eta_1, f_eta_2, f_eta_3);

        REAL tau_g_maj[N_CELL][N_GPB];
#pragma HLS ARRAY_PARTITION variable = tau_g_maj dim = 2 type = complete
        tau_major_bnd(
            i_bnd,
            j_T_2,
            f_T_1,
            j_strato_2,
            j_p,
            f_p,
            r_mix,
            j_eta_1,
            f_eta_1,
            k_major,
            tau_g_maj);

        tau_rayleigh_bnd(
            i_bnd,
            j_T_3,
            f_T_2,
            j_strato_3,
            n_d_2,
            r_gas_2,
            j_eta_2,
            f_eta_2,
            tau_r[i_bnd]);

        REAL dry_fact[N_CELL];
        compute_dry_fact(r_h2o, dry_fact);

        REAL tau_g_min[N_CELL][N_GPB];
#pragma HLS ARRAY_PARTITION variable = tau_g_min dim = 2 type = complete
        tau_minor_bnd(
            i_bnd,
            j_T_4,
            f_T_3,
            j_strato_4,
            k,
            n_d_3,
            r_gas_3,
            dry_fact,
            j_eta_3,
            f_eta_3,
            tau_g_min);

        tau_sum(tau_g_min, tau_g_maj, tau_g[i_bnd]);
    }
}

extern "C" {

void taumol_sw(
    const REAL T[N_CELL],
    const REAL p[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    const REAL k_major[N_BND][14][9][60][N_GPB],
    REAL tau_g[N_BND][N_CELL][N_GPB],
    REAL tau_r[N_BND][N_CELL][N_GPB])
{
#pragma HLS INTERFACE mode = bram port = T
#pragma HLS INTERFACE mode = bram port = p
#pragma HLS INTERFACE mode = bram port = n_d
#pragma HLS INTERFACE mode = bram port = r_gas storage_type = ram_1wnr
#pragma HLS INTERFACE mode = m_axi port = k_major bundle = gmem0
#pragma HLS INTERFACE mode = m_axi port = tau_g bundle = gmem1
#pragma HLS INTERFACE mode = m_axi port = tau_r bundle = gmem2

    index_t j_T[N_CELL];
    REAL f_T[N_CELL];
    compute_T_prime(T, j_T, f_T);

    index_t j_strato[N_CELL];
    index_t j_p[N_CELL];
    REAL f_p[N_CELL];
    compute_p_prime(p, j_strato, j_p, f_p);

    REAL k[N_CELL];
    compute_k(T, p, k);

    tau_df(j_T, f_T, j_strato, j_p, f_p, k, n_d, r_gas, k_major, tau_g, tau_r);
}

}
