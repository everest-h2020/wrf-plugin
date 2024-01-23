from rrtmgp.parameters import *

def interpolate(profile: ReferenceProfile, T, p, n_prime):
    """Projects volume properties to linear fractions of the profiles."""

    # Normalize temperature and pressure to the reference profiles.
    T_prime     = (T - profile.T_min) / profile.T_delta
    T_prime     = np.maximum(np.minimum(T_prime, profile.lim_T), 0)
    p_prime     = (np.log(p) - profile.p_log_max) / profile.p_log_delta
    p_prime     = np.maximum(np.minimum(p_prime, profile.lim_p), 0)

    # Calculate troposphere / stratosphere boundary.
    i_strato    = (p < profile.p_tropo) * 1

    # Compute the binary species parameter (eta).
    n_prime_g0  = n_prime[..., profile.absorbers.flav_to_abs[:,0]]
    n_prime_g1  = n_prime[..., profile.absorbers.flav_to_abs[:,1]]
    T_eta       = np.stack((T_prime, T_prime + 1)).astype(np.int32)
    r_eta       = np.transpose(profile.eta_half[i_strato,T_eta,:], (1, 2, 0))
    n_prime_mix = n_prime_g0[...,None] + r_eta * n_prime_g1[...,None]
    eta         = np.select(
        [n_prime_mix > profile.tiny, n_prime_mix <= profile.tiny],
        [n_prime_g0[..., None] / n_prime_mix, 0.5]) * (profile.N_eta-1)
    eta         = np.maximum(np.minimum(eta, profile.lim_eta), 0)

    return T_prime, p_prime, eta, i_strato, n_prime_mix

def tau_major(spectra: AbsorptionSpectra, i_strato, j_p, j_T, j_eta, f_major, n_prime_mix):
    i_lay       = np.arange(j_T.shape[0])
    i_p         = j_p + i_strato
    i_p         = np.stack((i_p,i_p+1), -1)[:,:,None,None]
    i_T         = np.stack((j_T,j_T+1), -1)[:,None,:,None]

    result      = np.zeros((spectra.N_gpt, j_T.shape[0]))
    bnd_start   = np.cumsum(spectra.bnd_width) - spectra.bnd_width[0]
    for i_bnd in range(spectra.bnd_to_flav.shape[1]):
        gptS, gptE  = bnd_start[i_bnd], bnd_start[i_bnd] + spectra.bnd_width[i_bnd]

        i_flav  = spectra.bnd_to_flav[i_strato][:,i_bnd]
        i_eta   = j_eta[i_lay,i_flav,...]
        i_eta   = np.stack((i_eta,i_eta+1), -1)[:,None,:,:]

        a       = n_prime_mix[None,i_lay,i_flav,None,:,None]
        b       = f_major[None,i_lay,i_flav,:,:,:]
        c       = spectra.k_major[gptS:gptE,i_p,i_T,i_eta]

        result[gptS:gptE,:] = np.sum(a * b * c, (2,3,4))

    return result

def tau_minor(spectra: AbsorptionSpectra, p, T, i_strato, j_T, j_eta, f_minor, n_prime):
    absorbers   = spectra.absorbers

    i_lay       = np.arange(j_T.shape[0])
    i_T         = np.stack((j_T,j_T+1), -1)[:,:,None]

    result      = np.zeros((spectra.N_gpt, j_T.shape[0]))
    bnd_start   = np.cumsum(spectra.bnd_width) - spectra.bnd_width[0]
    def _layer(layer):
        for i_mabsi in range(absorbers.N_mabsi[layer]):
            i_bnd   = spectra.mabsi_to_bnd[layer][i_mabsi]
            gptS, gptE  = bnd_start[i_bnd], bnd_start[i_bnd] + spectra.bnd_width[i_bnd]

            i_flav  = spectra.bnd_to_flav[layer][i_bnd]
            mciS    = spectra.mabsi_to_mci[layer][i_mabsi]
            mciE    = mciS + spectra.bnd_width[i_bnd]

            i_eta   = j_eta[i_lay,i_flav,...]
            i_eta   = np.stack((i_eta,i_eta+1), -1)

            i_abs   = absorbers.mabsi_to_abs[layer][i_mabsi]
            f_mabs  = np.copy(n_prime[:,i_abs])

            if spectra.mabsi_density[layer][i_mabsi]:
                f_mabs = f_mabs * (0.01*p/T)
                scale_by = spectra.mabsi_scale_by[layer][i_mabsi] or 0
                if scale_by > 0:
                    vmr_fact = 1 / n_prime[:,0]
                    dry_fact = 1 / (1 + n_prime[:,absorbers.i_h2o] * vmr_fact)
                    gas_fact = n_prime[:,scale_by] * vmr_fact * dry_fact
                    if spectra.mabsi_compl[layer][i_mabsi]:
                        f_mabs = f_mabs * (1 - gas_fact)
                    else:
                        f_mabs = f_mabs * gas_fact

            f_mabs[i_strato != layer] = 0

            a   = f_mabs[None,:,None,None]
            b   = f_minor[None,i_lay,i_flav,:,:]
            c   = spectra.k_minor[layer][mciS:mciE,i_T,i_eta]

            result[gptS:gptE,:] = result[gptS:gptE,:] + np.sum(a * b * c, (2,3))

    _layer(0)
    _layer(1)

    return result

def tau_rayleigh(spectra: AbsorptionSpectra, i_strato, j_T, j_eta, f_minor, n_prime):
    absorbers   = spectra.absorbers

    i_lay       = np.arange(j_T.shape[0])
    i_T         = np.stack((j_T,j_T+1), -1)[:,:,None]
    n_wet       = n_prime[:,absorbers.i_h2o]+n_prime[:,0]

    result      = np.zeros((spectra.N_gpt, j_T.shape[0]))
    bnd_start   = np.cumsum(spectra.bnd_width) - spectra.bnd_width[0]
    for i_bnd in range(spectra.bnd_to_flav.shape[1]):
        gptS, gptE  = bnd_start[i_bnd], bnd_start[i_bnd] + spectra.bnd_width[i_bnd]

        i_flav  = spectra.bnd_to_flav[i_strato][:,i_bnd]
        i_eta   = j_eta[i_lay,i_flav,...]
        i_eta   = np.stack((i_eta,i_eta+1), -1)

        a       = n_wet[None,i_lay,None,None]
        b       = f_minor[None,i_lay,i_flav,:,:]
        c       = spectra.k_rayleigh[i_strato[:,None,None],gptS:gptE,i_T,i_eta]
        c       = np.transpose(c, (3, 0, 1, 2))

        result[gptS:gptE,:] = np.sum(a * b * c, (2,3))

    return result
