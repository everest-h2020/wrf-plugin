#include "rrtmg/kernels.h"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

using namespace std;
using namespace rrtmg;

#include "data.inc"

namespace {

static constexpr REAL C_TINY = numeric_limits<REAL>::min();

static constexpr std::pair<index_t, REAL> int_frac(REAL x)
{
    REAL int_part;
    const auto frac = std::modf(x, &int_part);
    return std::make_pair(static_cast<index_t>(int_part), frac);
}

} // namespace

namespace rrtmg {

void taumol_sw(
    const REAL T[N_CELL],
    const REAL p[N_CELL],
    const REAL n_d[N_CELL],
    const REAL r_gas[N_GAS][N_CELL],
    REAL tau_g[N_BND][N_CELL][N_GPB],
    REAL tau_r[N_BND][N_CELL][N_GPB])
{
    REAL p_prime[N_CELL];
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
        p_prime[i_cell] =
            (std::log(p[i_cell]) - C_LOG_MAX_P_REF) / C_DELTA_LOG_P_REF;
    }

    REAL T_prime[N_CELL];
    for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell)
        T_prime[i_cell] = (T[i_cell] - C_MIN_T_REF) / C_DELTA_T_REF;

    for (index_t i_bnd = 0; i_bnd < C_N_BND; ++i_bnd) {
        REAL eta[N_CELL][2], r_mix[N_CELL][2];
        for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
            const index_t i_strato =
                (p_prime[i_cell] > C_P_PRIME_TROPO) ? 1 : 0;
            const auto i_flav = C_BND_TO_FLAV[i_strato][i_bnd];
            const auto i_g_0 = C_FLAV_TO_ABS[i_flav][0];
            const auto i_g_1 = C_FLAV_TO_ABS[i_flav][1];

            const auto r_g_0 = i_g_0 >= 0 ? r_gas[i_g_0][i_cell] : REAL(1);
            const auto r_g_1 = i_g_1 >= 0 ? r_gas[i_g_1][i_cell] : REAL(1);
            const auto [j_T, _] = int_frac(T_prime[i_cell]);

            for (index_t dT = 0; dT < 2; ++dT) {
                const auto r_eta_half = C_ETA_HALF[i_strato][i_flav][j_T + dT];
                const auto f_mix = r_g_0 + r_eta_half * r_g_1;
                r_mix[i_cell][dT] = f_mix;
                const auto alpha = f_mix > C_TINY ? (r_g_0 / f_mix) : REAL(0.5);
                eta[i_cell][dT] = alpha * (C_N_ETA - 1);
            }
        }

        for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
            const index_t i_strato =
                (p_prime[i_cell] > C_P_PRIME_TROPO) ? 1 : 0;
            const auto [j_T, f_T] = int_frac(T_prime[i_cell]);
            const auto [j_p, f_p] = int_frac(p_prime[i_cell]);

            for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb)
                tau_g[i_bnd][i_cell][i_gpb] = REAL(0);

            for (index_t dT = 0; dT < 2; ++dT) {
                const auto a_T = dT == 1 ? f_T : REAL(1) - f_T;
                for (index_t deta = 0; deta < 2; ++deta) {
                    const auto [j_eta, f_eta] = int_frac(eta[i_cell][dT]);
                    const auto a_eta = deta == 1 ? f_eta : REAL(1) - f_eta;
                    const auto a_minor =
                        a_T * a_eta * r_mix[i_cell][dT] * n_d[i_cell];
                    for (index_t dp = 0; dp < 2; ++dp) {
                        const auto a_p = dp == 1 ? f_p : REAL(1) - f_p;
                        for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
                            tau_g[i_bnd][i_cell][i_gpb] +=
                                C_K_MAJOR[i_bnd][j_T + dT][j_eta + deta]
                                         [j_p + dp + i_strato][i_gpb]
                                * a_minor * a_p;
                        }
                    }
                }
            }
        }

        for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
            const index_t i_strato =
                (p_prime[i_cell] > C_P_PRIME_TROPO) ? 1 : 0;
            const auto [j_T, f_T] = int_frac(T_prime[i_cell]);
            const auto a_wet = n_d[i_cell] * (REAL(1) + r_gas[0][i_cell]);

            for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb)
                tau_r[i_bnd][i_cell][i_gpb] = REAL(0);

            for (index_t dT = 0; dT < 2; ++dT) {
                const auto a_T = dT == 1 ? f_T : REAL(1) - f_T;
                for (index_t deta = 0; deta < 2; ++deta) {
                    const auto [j_eta, f_eta] = int_frac(eta[i_cell][dT]);
                    const auto a_eta = deta == 1 ? f_eta : REAL(1) - f_eta;
                    for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
                        tau_r[i_bnd][i_cell][i_gpb] +=
                            C_K_RAYLEIGH[i_bnd][i_strato][j_T + dT]
                                        [j_eta + deta][i_gpb]
                            * a_T * a_eta * a_wet;
                    }
                }
            }
        }

        const auto N_MPB =
            C_MINOR_PER_BND[i_bnd][0] + C_MINOR_PER_BND[i_bnd][1];
        for (index_t i_mpb = 0; i_mpb < N_MPB; ++i_mpb) {
            const auto i_minor = C_MINOR_START[i_bnd] + i_mpb;
            const auto i_strato = (i_mpb >= C_MINOR_PER_BND[i_bnd][0]) ? 1 : 0;
            const auto i_abs = C_MINOR_TO_ABS[i_minor];
            const auto scale_by = C_MINOR_SCALE_BY[i_minor];

            for (index_t i_cell = 0; i_cell < N_CELL; ++i_cell) {
                if ((p_prime[i_cell] > C_P_PRIME_TROPO) != i_strato) continue;

                auto r_abs = r_gas[i_abs][i_cell] * n_d[i_cell];
                if (scale_by != 0) {
                    r_abs *= REAL(0.01) * p[i_cell] / T[i_cell];
                    const auto i_by = std::abs(scale_by) - 2;
                    if (i_by >= 0) {
                        const auto dry_fact = REAL(1) / (1 + r_gas[0][i_cell]);
                        const auto gas_fact = r_gas[i_by][i_cell] * dry_fact;
                        r_abs *= scale_by < 0 ? (REAL(1) - gas_fact) : gas_fact;
                    }
                }

                const auto [j_T, f_T] = int_frac(T_prime[i_cell]);

                for (index_t dT = 0; dT < 2; ++dT) {
                    const auto a_T = dT == 1 ? f_T : REAL(1) - f_T;
                    for (index_t deta = 0; deta < 2; ++deta) {
                        const auto [j_eta, f_eta] = int_frac(eta[i_cell][dT]);
                        const auto a_eta = deta == 1 ? f_eta : REAL(1) - f_eta;
                        const auto alpha = a_T * a_eta * r_abs;

                        for (index_t i_gpb = 0; i_gpb < N_GPB; ++i_gpb) {
                            tau_g[i_bnd][i_cell][i_gpb] +=
                                C_K_MINOR[i_minor][j_T + dT][j_eta + deta]
                                         [i_gpb]
                                * alpha;
                        }
                    }
                }
            }
        }
    }
}

} // namespace rrtmg
