#include "ABI.h"
#include "rrtmgp/kernel_launchers.h"

#include <algorithm>
#include <cmath>

namespace {

void interpolation_kernel_fpga(
    const int ncol,
    const int nlay,
    const int ngas,
    const int nflav,
    const int neta,
    const int npres,
    const int ntemp,
    const REAL tmin,
    const int* __restrict__ flavor,
    const REAL* __restrict__ press_ref_log,
    const REAL* __restrict__ temp_ref,
    REAL press_ref_log_delta,
    REAL temp_ref_min,
    REAL temp_ref_delta,
    REAL press_ref_trop_log,
    const REAL* __restrict__ vmr_ref,
    const REAL* __restrict__ play,
    const REAL* __restrict__ tlay,
    const REAL* __restrict__ col_gas,
    int* __restrict__ jtemp,
    REAL* __restrict__ fmajor,
    REAL* __restrict__ fminor,
    REAL* __restrict__ col_mix,
    BOOL* __restrict__ tropo,
    int* __restrict__ jeta,
    int* __restrict__ jpress,
    const int blockIdx_x,
    const int blockIdx_y,
    const int blockIdx_z,
    const int blockDim_x,
    const int blockDim_y,
    const int blockDim_z,
    const int threadIdx_x,
    const int threadIdx_y,
    const int threadIdx_z)
{

    const int icol = blockIdx_x * blockDim_x + threadIdx_x;
    const int ilay = blockIdx_y * blockDim_y + threadIdx_y;
    const int iflav = blockIdx_z * blockDim_z + threadIdx_z;

    if ((icol < ncol) && (ilay < nlay) && (iflav < nflav)) {
        const int idx = icol + ilay * ncol;

        jtemp[idx] =
            int((tlay[idx] - (temp_ref_min - temp_ref_delta)) / temp_ref_delta);
        jtemp[idx] = std::max(1, std::min(ntemp - 1, std::max(1, jtemp[idx])));
        const REAL ftemp =
            (tlay[idx] - temp_ref[jtemp[idx] - 1]) / temp_ref_delta;

        const REAL locpress =
            REAL(1.)
            + (log(play[idx]) - press_ref_log[0]) / press_ref_log_delta;
        jpress[idx] =
            std::max(1, std::min(npres - 1, std::max(1, int(locpress))));
        const REAL fpress = locpress - REAL(jpress[idx]);

        tropo[idx] = log(play[idx]) > press_ref_trop_log;
        const int itropo = !tropo[idx];

        const int gas1 = flavor[2 * iflav];
        const int gas2 = flavor[2 * iflav + 1];

        for (int itemp = 0; itemp < 2; ++itemp) {
            const int vmr_base_idx =
                itropo + (jtemp[idx] + itemp - 1) * (ngas + 1) * 2;
            const int colmix_idx =
                itemp + 2 * (icol + ilay * ncol + iflav * ncol * nlay);
            const int colgas1_idx = icol + ilay * ncol + gas1 * nlay * ncol;
            const int colgas2_idx = icol + ilay * ncol + gas2 * nlay * ncol;
            const REAL ratio_eta_half = vmr_ref[vmr_base_idx + 2 * gas1]
                                        / vmr_ref[vmr_base_idx + 2 * gas2];
            col_mix[colmix_idx] =
                col_gas[colgas1_idx] + ratio_eta_half * col_gas[colgas2_idx];

            REAL eta;
            if (col_mix[colmix_idx] > REAL(2.) * tmin)
                eta = col_gas[colgas1_idx] / col_mix[colmix_idx];
            else
                eta = REAL(0.5);

            const REAL loceta = eta * REAL(neta - 1);
            jeta[colmix_idx] = std::max(1, std::min(int(loceta) + 1, neta - 1));
            const REAL feta = std::fmod(loceta, REAL(1.));
            const REAL ftemp_term =
                REAL(1 - itemp) + REAL(2 * itemp - 1) * ftemp;

            // Compute interpolation fractions needed for minor species.
            const int fminor_idx =
                2 * (itemp + 2 * (icol + ilay * ncol + iflav * ncol * nlay));
            fminor[fminor_idx] = (REAL(1.) - feta) * ftemp_term;
            fminor[fminor_idx + 1] = feta * ftemp_term;

            // Compute interpolation fractions needed for major species.
            const int fmajor_idx =
                2 * 2
                * (itemp + 2 * (icol + ilay * ncol + iflav * ncol * nlay));
            fmajor[fmajor_idx] = (REAL(1.) - fpress) * fminor[fminor_idx];
            fmajor[fmajor_idx + 1] =
                (REAL(1.) - fpress) * fminor[fminor_idx + 1];
            fmajor[fmajor_idx + 2] = fpress * fminor[fminor_idx];
            fmajor[fmajor_idx + 3] = fpress * fminor[fminor_idx + 1];
        }
    }
}

void gas_optical_depths_major_kernel_fpga(
    const int ncol,
    const int nlay,
    const int nband,
    const int ngpt,
    const int nflav,
    const int neta,
    const int npres,
    const int ntemp,
    const int* __restrict__ gpoint_flavor,
    const int* __restrict__ band_lims_gpt,
    const REAL* __restrict__ kmajor,
    const REAL* __restrict__ col_mix,
    const REAL* __restrict__ fmajor,
    const int* __restrict__ jeta,
    const BOOL* __restrict__ tropo,
    const int* __restrict__ jtemp,
    const int* __restrict__ jpress,
    REAL* __restrict__ tau,
    const int blockIdx_x,
    const int blockIdx_y,
    const int blockIdx_z,
    const int blockDim_x,
    const int blockDim_y,
    const int blockDim_z,
    const int threadIdx_x,
    const int threadIdx_y,
    const int threadIdx_z)
{

    const int igpt = blockIdx_x * blockDim_x + threadIdx_x;
    const int ilay = blockIdx_y * blockDim_y + threadIdx_y;
    const int icol = blockIdx_z * blockDim_z + threadIdx_z;

    if ((icol < ncol) && (ilay < nlay) && (igpt < ngpt)) {
        const int idx_collay = icol + ilay * ncol;
        const int itropo = !tropo[idx_collay];
        const int iflav = gpoint_flavor[itropo + 2 * igpt] - 1;

        const int ljtemp = jtemp[idx_collay];
        const int jpressi = jpress[idx_collay] + itropo;
        const int npress = npres + 1;

        // Major gases.
        const int idx_fcl3 =
            2 * 2 * 2 * (icol + ilay * ncol + iflav * ncol * nlay);
        const int idx_fcl1 = 2 * (icol + ilay * ncol + iflav * ncol * nlay);

        const REAL* __restrict__ ifmajor = &fmajor[idx_fcl3];
        const int idx_out = icol + ilay * ncol + igpt * ncol * nlay;

        for (int i = 0; i < 2; ++i) {
            tau[idx_out] +=
                col_mix[idx_fcl1 + i]
                * (ifmajor[i * 4 + 0]
                       * kmajor
                           [(ljtemp - 1 + i) + (jeta[idx_fcl1 + i] - 1) * ntemp
                            + (jpressi - 1) * ntemp * neta
                            + igpt * ntemp * neta * npress]
                   + ifmajor[i * 4 + 1]
                         * kmajor
                             [(ljtemp - 1 + i) + jeta[idx_fcl1 + i] * ntemp
                              + (jpressi - 1) * ntemp * neta
                              + igpt * ntemp * neta * npress]
                   + ifmajor[i * 4 + 2]
                         * kmajor
                             [(ljtemp - 1 + i)
                              + (jeta[idx_fcl1 + i] - 1) * ntemp
                              + jpressi * ntemp * neta
                              + igpt * ntemp * neta * npress]
                   + ifmajor[i * 4 + 3]
                         * kmajor
                             [(ljtemp - 1 + i) + jeta[idx_fcl1 + i] * ntemp
                              + jpressi * ntemp * neta
                              + igpt * ntemp * neta * npress]);
        }
    }
}

void gas_optical_depths_minor_kernel_fpga(
    const int ncol,
    const int nlay,
    const int ngpt,
    const int ngas,
    const int nflav,
    const int ntemp,
    const int neta,
    const int nminor,
    const int nminork,
    const int idx_h2o,
    const int idx_tropo,
    const int* __restrict__ gpoint_flavor,
    const REAL* __restrict__ kminor,
    const int* __restrict__ minor_limits_gpt,
    const BOOL* __restrict__ minor_scales_with_density,
    const BOOL* __restrict__ scale_by_complement,
    const int* __restrict__ idx_minor,
    const int* __restrict__ idx_minor_scaling,
    const int* __restrict__ kminor_start,
    const REAL* __restrict__ play,
    const REAL* __restrict__ tlay,
    const REAL* __restrict__ col_gas,
    const REAL* __restrict__ fminor,
    const int* __restrict__ jeta,
    const int* __restrict__ jtemp,
    const BOOL* __restrict__ tropo,
    REAL* __restrict__ tau,
    REAL* __restrict__ tau_minor,
    int block_size_x,
    int block_size_y,
    const int block_size_z,
    const int blockIdx_y,
    const int blockIdx_z,
    const int threadIdx_x,
    const int threadIdx_y,
    const int threadIdx_z)
{

    const int ilay = blockIdx_y * block_size_y + threadIdx_y;
    const int icol = blockIdx_z * block_size_z + threadIdx_z;

    if ((icol < ncol) && (ilay < nlay)) {
        const int idx_collay = icol + ilay * ncol;

        if (tropo[idx_collay] == idx_tropo) {
            for (int imnr = 0; imnr < nminor; ++imnr) {
                REAL scaling = REAL(0.);

                const int ncl = ncol * nlay;
                scaling = col_gas[idx_collay + idx_minor[imnr] * ncl];
                if (minor_scales_with_density[imnr]) {
                    const REAL PaTohPa = 0.01;
                    scaling *= PaTohPa * play[idx_collay] / tlay[idx_collay];
                    if (idx_minor_scaling[imnr] > 0) {
                        const int idx_collaywv =
                            icol + ilay * ncol + idx_h2o * ncl;
                        REAL vmr_fact = REAL(1.) / col_gas[idx_collay];
                        REAL dry_fact =
                            REAL(1.)
                            / (REAL(1.) + col_gas[idx_collaywv] * vmr_fact);
                        if (scale_by_complement[imnr])
                            scaling *=
                                (REAL(1.)
                                 - col_gas
                                           [idx_collay
                                            + idx_minor_scaling[imnr] * ncl]
                                       * vmr_fact * dry_fact);
                        else
                            scaling *=
                                col_gas
                                    [idx_collay + idx_minor_scaling[imnr] * ncl]
                                * vmr_fact * dry_fact;
                    }
                }

                const int gpt_start = minor_limits_gpt[2 * imnr] - 1;
                const int gpt_end = minor_limits_gpt[2 * imnr + 1];
                const int gpt_offs = 1 - idx_tropo;
                const int iflav = gpoint_flavor[2 * gpt_start + gpt_offs] - 1;

                const int idx_fcl2 =
                    2 * 2 * (icol + ilay * ncol + iflav * ncol * nlay);
                const int idx_fcl1 =
                    2 * (icol + ilay * ncol + iflav * ncol * nlay);

                const REAL* kfminor = &fminor[idx_fcl2];
                const REAL* kin = &kminor[0];

                const int j0 = jeta[idx_fcl1];
                const int j1 = jeta[idx_fcl1 + 1];
                const int kjtemp = jtemp[idx_collay];
                const int band_gpt = gpt_end - gpt_start;
                const int gpt_offset = kminor_start[imnr] - 1;

                for (int igpt = threadIdx_x; igpt < band_gpt;
                     igpt += block_size_x) {
                    REAL ltau_minor =
                        kfminor[0]
                            * kin
                                [(kjtemp - 1) + (j0 - 1) * ntemp
                                 + (igpt + gpt_offset) * ntemp * neta]
                        + kfminor[1]
                              * kin
                                  [(kjtemp - 1) + j0 * ntemp
                                   + (igpt + gpt_offset) * ntemp * neta]
                        + kfminor[2]
                              * kin
                                  [kjtemp + (j1 - 1) * ntemp
                                   + (igpt + gpt_offset) * ntemp * neta]
                        + kfminor[3]
                              * kin
                                  [kjtemp + j1 * ntemp
                                   + (igpt + gpt_offset) * ntemp * neta];

                    const int idx_out =
                        icol + ilay * ncol + (igpt + gpt_start) * ncol * nlay;
                    tau[idx_out] += ltau_minor * scaling;
                }
            }
        }
    }
}

void compute_tau_rayleigh_kernel_fpga(
    const int ncol,
    const int nlay,
    const int nbnd,
    const int ngpt,
    const int ngas,
    const int nflav,
    const int neta,
    const int npres,
    const int ntemp,
    const int* __restrict__ gpoint_flavor,
    const int* __restrict__ gpoint_bands,
    const int* __restrict__ band_lims_gpt,
    const REAL* __restrict__ krayl,
    int idx_h2o,
    const REAL* __restrict__ col_dry,
    const REAL* __restrict__ col_gas,
    const REAL* __restrict__ fminor,
    const int* __restrict__ jeta,
    const BOOL* __restrict__ tropo,
    const int* __restrict__ jtemp,
    REAL* __restrict__ tau_rayleigh,
    const int blockIdx_x,
    const int blockIdx_y,
    const int blockIdx_z,
    const int blockDim_x,
    const int blockDim_y,
    const int blockDim_z,
    const int threadIdx_x,
    const int threadIdx_y,
    const int threadIdx_z)
{

    // Fetch the three coordinates.
    const int icol = blockIdx_x * blockDim_x + threadIdx_x;
    const int ilay = blockIdx_y * blockDim_y + threadIdx_y;
    const int igpt = blockIdx_z * blockDim_z + threadIdx_z;

    if ((icol < ncol) && (ilay < nlay) && (igpt < ngpt)) {
        const int ibnd = gpoint_bands[igpt] - 1;

        const int idx_collay = icol + ilay * ncol;
        const int idx_collaywv = icol + ilay * ncol + idx_h2o * nlay * ncol;
        const int itropo = !tropo[idx_collay];

        const int gpt_start = band_lims_gpt[2 * ibnd] - 1;
        const int iflav = gpoint_flavor[itropo + 2 * gpt_start] - 1;

        const int idx_fcl2 = 2 * 2 * (icol + ilay * ncol + iflav * ncol * nlay);
        const int idx_fcl1 = 2 * (icol + ilay * ncol + iflav * ncol * nlay);

        const int idx_krayl = itropo * ntemp * neta * ngpt;

        const int j0 = jeta[idx_fcl1];
        const int j1 = jeta[idx_fcl1 + 1];
        const int jtempl = jtemp[idx_collay];

        const REAL kloc =
            fminor[idx_fcl2 + 0]
                * krayl
                    [idx_krayl + (jtempl - 1) + (j0 - 1) * ntemp
                     + igpt * ntemp * neta]
            + fminor[idx_fcl2 + 1]
                  * krayl
                      [idx_krayl + (jtempl - 1) + j0 * ntemp
                       + igpt * ntemp * neta]
            + fminor[idx_fcl2 + 2]
                  * krayl
                      [idx_krayl + (jtempl) + (j1 - 1) * ntemp
                       + igpt * ntemp * neta]
            + fminor[idx_fcl2 + 3]
                  * krayl
                      [idx_krayl + (jtempl) + j1 * ntemp + igpt * ntemp * neta];

        const int idx_out = icol + ilay * ncol + igpt * ncol * nlay;
        tau_rayleigh[idx_out] =
            kloc * (col_gas[idx_collaywv] + col_dry[idx_collay]);
    }
}

void combine_abs_and_rayleigh_kernel_fpga(
    const int ncol,
    const int nlay,
    const int ngpt,
    const REAL tmin,
    const REAL* __restrict__ tau_abs,
    const REAL* __restrict__ tau_rayleigh,
    REAL* __restrict__ tau,
    REAL* __restrict__ ssa,
    REAL* __restrict__ g,
    const int blockIdx_x,
    const int blockIdx_y,
    const int blockIdx_z,
    const int blockDim_x,
    const int blockDim_y,
    const int blockDim_z,
    const int threadIdx_x,
    const int threadIdx_y,
    const int threadIdx_z)
{

    // Fetch the three coordinates.
    const int icol = blockIdx_x * blockDim_x + threadIdx_x;
    const int ilay = blockIdx_y * blockDim_y + threadIdx_y;
    const int igpt = blockIdx_z * blockDim_z + threadIdx_z;

    if ((icol < ncol) && (ilay < nlay) && (igpt < ngpt)) {
        const int idx = icol + ilay * ncol + igpt * ncol * nlay;

        const REAL tau_tot = tau_abs[idx] + tau_rayleigh[idx];

        tau[idx] = tau_tot;
        g[idx] = REAL(0.);

        if (tau_tot > (REAL(2.) * tmin))
            ssa[idx] = tau_rayleigh[idx] / tau_tot;
        else
            ssa[idx] = REAL(0.);
    }
}
} // namespace

namespace rrtmgp {

void interpolation_fpga(
    const int ncol,
    const int nlay,
    const int ngas,
    const int nflav,
    const int neta,
    const int npres,
    const int ntemp,
    const int flavor[20],
    const REAL press_ref_log[59],
    const REAL temp_ref[14],
    REAL press_ref_log_delta,
    REAL temp_ref_min,
    REAL temp_ref_delta,
    REAL press_ref_trop_log,
    const REAL vmr_ref[252],
    const REAL play[5376],
    const REAL tlay[5376],
    const REAL col_gas[48384],
    int jtemp[5376],
    REAL fmajor[430080],
    REAL fminor[215040],
    REAL col_mix[107520],
    BOOL tropo[5376],
    int jeta[107520],
    int jpress[5376])
{

    const int block_col = 4;
    const int block_lay = 2;
    const int block_flav = 16;

    const int grid_col = ncol / block_col + (ncol % block_col > 0);
    const int grid_lay = nlay / block_lay + (nlay % block_lay > 0);
    const int grid_flav = nflav / block_flav + (nflav % block_flav > 0);

    REAL tmin = std::numeric_limits<REAL>::min();

    for (int i = 0; i < grid_flav; i++)
        for (int j = 0; j < grid_lay; j++)
            for (int k = 0; k < grid_col; k++)
                for (int l = 0; l < block_flav; l++)
                    for (int m = 0; m < block_lay; m++)
                        for (int n = 0; n < block_col; n++) {
                            interpolation_kernel_fpga(
                                ncol,
                                nlay,
                                ngas,
                                nflav,
                                neta,
                                npres,
                                ntemp,
                                tmin,
                                flavor,
                                press_ref_log,
                                temp_ref,
                                press_ref_log_delta,
                                temp_ref_min,
                                temp_ref_delta,
                                press_ref_trop_log,
                                vmr_ref,
                                play,
                                tlay,
                                col_gas,
                                jtemp,
                                fmajor,
                                fminor,
                                col_mix,
                                tropo,
                                jeta,
                                jpress,
                                k,
                                j,
                                i,
                                block_col,
                                block_lay,
                                block_flav,
                                n,
                                m,
                                l);
                        }
}

void compute_tau_absorption_fpga(
    const int ncol,
    const int nlay,
    const int nband,
    const int ngpt,
    const int ngas,
    const int nflav,
    const int neta,
    const int npres,
    const int ntemp,
    const int nminorlower,
    const int nminorklower,
    const int nminorupper,
    const int nminorkupper,
    const int idx_h2o,
    const int gpoint_flavor[448],
    const int band_lims_gpt[28],
    const REAL kmajor[1693440],
    const REAL kminor_lower[64512],
    const REAL kminor_upper[44352],
    const int minor_limits_gpt_lower[64],
    const int minor_limits_gpt_upper[44],
    const BOOL minor_scales_with_density_lower[32],
    const BOOL minor_scales_with_density_upper[22],
    const BOOL scale_by_complement_lower[32],
    const BOOL scale_by_complement_upper[22],
    const int idx_minor_lower[32],
    const int idx_minor_upper[22],
    const int idx_minor_scaling_lower[32],
    const int idx_minor_scaling_upper[22],
    const int kminor_start_lower[32],
    const int kminor_start_upper[22],
    const BOOL tropo[5376],
    const REAL col_mix[107520],
    const REAL fmajor[430080],
    const REAL fminor[215040],
    const REAL play[5376],
    const REAL tlay[5376],
    const REAL col_gas[48384],
    const int jeta[107520],
    const int jtemp[5376],
    const int jpress[5376],
    REAL tau[1204224])
{

    // from tuning:
    int grid_x = 64;
    int grid_y = 42;
    int grid_z = 3;
    int block_x = 4;
    int block_y = 1;
    int block_z = 48;

    for (int i = 0; i < grid_z; i++)
        for (int j = 0; j < grid_y; j++)
            for (int k = 0; k < grid_x; k++)
                for (int l = 0; l < block_z; l++)
                    for (int m = 0; m < block_y; m++)
                        for (int n = 0; n < block_x; n++) {
                            gas_optical_depths_major_kernel_fpga(
                                ncol,
                                nlay,
                                nband,
                                ngpt,
                                nflav,
                                neta,
                                npres,
                                ntemp,
                                gpoint_flavor,
                                band_lims_gpt,
                                kmajor,
                                col_mix,
                                fmajor,
                                jeta,
                                tropo,
                                jtemp,
                                jpress,
                                tau,
                                k,
                                j,
                                i,
                                block_x,
                                block_y,
                                block_z,
                                n,
                                m,
                                l);
                        }

    // Lower
    int idx_tropo = 1;

    // from tuning:
    grid_x = 1;
    grid_y = 42;
    grid_z = 8;
    block_x = 8;
    block_y = 1;
    block_z = 16;

    for (int i = 0; i < grid_z; i++)
        for (int j = 0; j < grid_y; j++)
            for (int k = 0; k < grid_x; k++)
                for (int l = 0; l < block_z; l++)
                    for (int m = 0; m < block_y; m++)
                        for (int n = 0; n < block_x; n++) {
                            gas_optical_depths_minor_kernel_fpga(
                                ncol,
                                nlay,
                                ngpt,
                                ngas,
                                nflav,
                                ntemp,
                                neta,
                                nminorlower,
                                nminorklower,
                                idx_h2o,
                                idx_tropo,
                                gpoint_flavor,
                                kminor_lower,
                                minor_limits_gpt_lower,
                                minor_scales_with_density_lower,
                                scale_by_complement_lower,
                                idx_minor_lower,
                                idx_minor_scaling_lower,
                                kminor_start_lower,
                                play,
                                tlay,
                                col_gas,
                                fminor,
                                jeta,
                                jtemp,
                                tropo,
                                tau,
                                nullptr,
                                block_x,
                                block_y,
                                block_z,
                                j,
                                i,
                                n,
                                m,
                                l);
                        }

    // Upper
    idx_tropo = 0;

    // from tuning:
    grid_x = 1;
    grid_y = 42;
    grid_z = 4;
    block_x = 8;
    block_y = 1;
    block_z = 32;

    for (int i = 0; i < grid_z; i++)
        for (int j = 0; j < grid_y; j++)
            for (int k = 0; k < grid_x; k++)
                for (int l = 0; l < block_z; l++)
                    for (int m = 0; m < block_y; m++)
                        for (int n = 0; n < block_x; n++) {
                            gas_optical_depths_minor_kernel_fpga(
                                ncol,
                                nlay,
                                ngpt,
                                ngas,
                                nflav,
                                ntemp,
                                neta,
                                nminorupper,
                                nminorkupper,
                                idx_h2o,
                                idx_tropo,
                                gpoint_flavor,
                                kminor_upper,
                                minor_limits_gpt_upper,
                                minor_scales_with_density_upper,
                                scale_by_complement_upper,
                                idx_minor_upper,
                                idx_minor_scaling_upper,
                                kminor_start_upper,
                                play,
                                tlay,
                                col_gas,
                                fminor,
                                jeta,
                                jtemp,
                                tropo,
                                tau,
                                nullptr,
                                block_x,
                                block_y,
                                block_z,
                                j,
                                i,
                                n,
                                m,
                                l);
                        }
}

void compute_tau_rayleigh_fpga(
    const int ncol,
    const int nlay,
    const int nbnd,
    const int ngpt,
    const int ngas,
    const int nflav,
    const int neta,
    const int npres,
    const int ntemp,
    const int gpoint_flavor[448],
    const int gpoint_bands[224],
    const int band_lims_gpt[28],
    const REAL krayl[56448],
    int idx_h2o,
    const REAL col_dry[5376],
    const REAL col_gas[48384],
    const REAL fminor[193536],
    const int jeta[96768],
    const BOOL tropo[5376],
    const int jtemp[5376],
    REAL tau_rayleigh[1204224])
{

    // from tuning:
    int grid_x = 32;
    int grid_y = 42;
    int grid_z = 14;
    int block_x = 4;
    int block_y = 1;
    int block_z = 16;

    for (int i = 0; i < grid_z; i++)
        for (int j = 0; j < grid_y; j++)
            for (int k = 0; k < grid_x; k++)
                for (int l = 0; l < block_z; l++)
                    for (int m = 0; m < block_y; m++)
                        for (int n = 0; n < block_x; n++) {
                            compute_tau_rayleigh_kernel_fpga(
                                ncol,
                                nlay,
                                nbnd,
                                ngpt,
                                ngas,
                                nflav,
                                neta,
                                npres,
                                ntemp,
                                gpoint_flavor,
                                gpoint_bands,
                                band_lims_gpt,
                                krayl,
                                idx_h2o,
                                col_dry,
                                col_gas,
                                fminor,
                                jeta,
                                tropo,
                                jtemp,
                                tau_rayleigh,
                                k,
                                j,
                                i,
                                block_x,
                                block_y,
                                block_z,
                                n,
                                m,
                                l);
                        }
}

void combine_abs_and_rayleigh_fpga(
    const int ncol,
    const int nlay,
    const int ngpt,
    const REAL tau_abs[1204224],
    const REAL tau_rayleigh[1204224],
    REAL tau[1204224],
    REAL ssa[1204224],
    REAL g[1204224])
{

    REAL tmin = std::numeric_limits<REAL>::min();

    // from tuning:
    // dim3 grid(4, 4, 2);
    // dim3 block(32, 11, 112);

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
                for (int l = 0; l < 112; l++)
                    for (int m = 0; m < 11; m++)
                        for (int n = 0; n < 32; n++) {
                            combine_abs_and_rayleigh_kernel_fpga(
                                ncol,
                                nlay,
                                ngpt,
                                tmin,
                                tau_abs,
                                tau_rayleigh,
                                tau,
                                ssa,
                                g,
                                k,
                                j,
                                i,
                                32,
                                11,
                                112,
                                n,
                                m,
                                l);
                        }
}

} // namespace rrtmgp