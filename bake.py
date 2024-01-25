from rrtmgp.reference import *
from rrtmgp.solar import *
import io

REAL    = np.float32

N_GAS = 7
N_GPB = 16
N_BND = 16

params      = Dataset('./data/rrtmgp-data-sw-g224-2018-12-04.nc', 'r')
absorbers   = AbsorberRegistry(params)
profile     = ReferenceProfile(params, absorbers)
spectra     = AbsorptionSpectra(params, absorbers)
source      = SolarSource(params)

output      = io.open('./data/rrtmgp-data-sw-g224-2018-12-04.baked.inc', 'w')

def bake_value(slice, is_integer=False):
    def fmt_value(val):
        return f'{val:d}' if is_integer else f'{val:G}'

    if slice.ndim == 0:
        output.write(fmt_value(slice[()]))
        return

    output.write('{')
    for x in np.nditer(slice, order='C'):
        output.write(f'{fmt_value(x)}, ')
    output.write('}')

ctypes = {
    'int32': 'INTEGER',
    'int64': 'index_t',
    'float32': 'REAL',
    'float64': 'REAL'
}

def bake_const(var, name):
    var = np.array(var)
    output.write(f'static constexpr {ctypes[var.dtype.name]}')
    extents = "".join(map(lambda x: f'[{x}]', var.shape))
    output.write(f' {name}{extents}')
    output.write(' = ')
    bake_value(var, np.issubdtype(var.dtype, np.integer))
    output.write(';\n')

# 1. Reference profile

bake_const(profile.N_T, 'C_N_T_REF')
bake_const(profile.T_min, 'C_MIN_T_REF')
bake_const(profile.T_delta, 'C_DELTA_T_REF')

bake_const(profile.N_p, 'C_N_P_REF')
bake_const(profile.p_log_max, 'C_LOG_MAX_P_REF')
bake_const(profile.p_log_delta, 'C_DELTA_LOG_P_REF')

C_P_PRIME_TROPO = (np.log(profile.p_tropo) - profile.p_log_max) / profile.p_log_delta
bake_const(C_P_PRIME_TROPO, 'C_P_PRIME_TROPO')

bake_const(profile.N_eta, 'C_N_ETA')
C_ETA_HALF = np.transpose(profile.eta_half, (0, 2, 1));
bake_const(C_ETA_HALF, 'C_ETA_HALF')

# 2. Absorber registry

# The WRF species are fixed and should be equal to what we have.
wrf_species = ["h2o", "co2", "o3", "n2o", "co=0", "ch4", "o2"]
for i, abs in enumerate(wrf_species):
    assert abs.endswith("=0") or absorbers.abs_names[i + 1] == abs
def wrf_has_species(idx):
    return idx <= len(wrf_species) and not wrf_species[idx - 1].endswith("=0")

bake_const(spectra.N_gpt, 'C_N_GPT')
bake_const(spectra.N_bnd, 'C_N_BND')
bake_const(spectra.bnd_width, 'C_BND_WIDTH')
bake_const(spectra.bnd_to_flav, 'C_BND_TO_FLAV')
C_FLAV_TO_ABS = absorbers.flav_to_abs - 1
bake_const(C_FLAV_TO_ABS, 'C_FLAV_TO_ABS')

# 3. Absorption spectra

#C_K_PRESCALE_LD = 77
C_K_PRESCALE_LD = 0
C_K_PRESCALE = np.exp2(C_K_PRESCALE_LD)
#bake_const(C_K_PRESCALE_LD, 'C_K_PRESCALE_LD')

C_K_MAJOR = np.zeros((N_BND, profile.N_T, profile.N_eta, profile.N_p + 1, N_GPB))
for i_bnd in range(spectra.N_bnd):
    for i_gpb in range(spectra.bnd_width[i_bnd]):
        i_gpt = spectra.bnd_limits_gpt[i_bnd][0] + i_gpb - 1
        C_K_MAJOR[i_bnd, :, :, :, i_gpb] = np.transpose(spectra.k_major[i_gpt, :, :, :], (1, 2, 0)) * C_K_PRESCALE
bake_const(C_K_MAJOR, 'C_K_MAJOR')

def filter_mabsi(i_strat: int):
    def wrf_has_mabsi(i_mabsi: int):
        if not wrf_has_species(absorbers.mabsi_to_abs[i_strat][i_mabsi]):
            return False
        if not spectra.mabsi_density[i_strat][i_mabsi]:
            return True
        i_abs = spectra.mabsi_scale_by[i_strat][i_mabsi]
        return i_abs is None or wrf_has_species(i_abs)

    mabsi = [i_mabsi for i_mabsi in range(absorbers.N_mabsi[i_strat]) if wrf_has_mabsi(i_mabsi)]
    mabsi.sort(key=lambda i_mabsi: spectra.mabsi_to_bnd[i_strat][i_mabsi])
    return np.array(mabsi)

mabsi_lower, mabsi_upper = filter_mabsi(0), filter_mabsi(1)

C_N_MINOR = np.array((mabsi_lower.shape[0], mabsi_upper.shape[0]))
bake_const(C_N_MINOR, 'C_N_MINOR')

C_MINOR_PER_BND = np.zeros((spectra.N_bnd, 2), dtype=np.int64)
C_MINOR_TO_ABS = np.zeros((np.sum(C_N_MINOR)), dtype=np.int64)
C_MINOR_SCALE_BY = np.zeros((np.sum(C_N_MINOR)), dtype=np.int64)
C_K_MINOR = np.zeros((np.sum(C_N_MINOR), profile.N_T, profile.N_eta, N_GPB))

i_minor = 0
for i_bnd in range(spectra.N_bnd):
    lower = list(filter(lambda x: spectra.mabsi_to_bnd[0][x] == i_bnd, mabsi_lower))
    upper = list(filter(lambda x: spectra.mabsi_to_bnd[1][x] == i_bnd, mabsi_upper))

    C_MINOR_PER_BND[i_bnd][0] = len(lower)
    C_MINOR_PER_BND[i_bnd][1] = len(upper)

    def emit_minor(i_minor: int, i_strat: int, i_mabsi: int):
        C_MINOR_TO_ABS[i_minor] = absorbers.mabsi_to_abs[i_strat][i_mabsi] - 1
        for i_gpb in range(spectra.bnd_width[i_bnd]):
            i_mci = spectra.mabsi_to_mci[i_strat][i_mabsi] + i_gpb
            C_K_MINOR[i_minor,:,:,i_gpb] = spectra.k_minor[i_strat][i_mci,:,:]

        scale_by = 0
        if spectra.mabsi_density[i_strat][i_mabsi]:
            scale_by = (spectra.mabsi_scale_by[i_strat][i_mabsi] or 0) + 1
            if spectra.mabsi_compl[i_strat][i_mabsi]:
                scale_by = -scale_by
        C_MINOR_SCALE_BY[i_minor] = scale_by

    for i_mabsi in lower:
        emit_minor(i_minor, 0, i_mabsi)
        i_minor = i_minor + 1
    for i_mabsi in upper:
        emit_minor(i_minor, 1, i_mabsi)
        i_minor = i_minor + 1

C_MINOR_START = np.zeros((spectra.N_bnd), dtype=np.int64)
C_MINOR_START[1:] = np.cumsum(np.sum(C_MINOR_PER_BND, axis=1))[:-1]
bake_const(C_MINOR_START, 'C_MINOR_START')
bake_const(C_MINOR_PER_BND, 'C_MINOR_PER_BND')
bake_const(C_MINOR_TO_ABS, 'C_MINOR_TO_ABS')
bake_const(C_MINOR_SCALE_BY, 'C_MINOR_SCALE_BY')
bake_const(C_K_MINOR, 'C_K_MINOR')

C_K_RAYLEIGH = np.zeros((N_BND, 2, profile.N_T, profile.N_eta, N_GPB))
for i_bnd in range(spectra.N_bnd):
    for i_gpb in range(spectra.bnd_width[i_bnd]):
        i_gpt = spectra.bnd_limits_gpt[i_bnd][0] + i_gpb - 1
        C_K_RAYLEIGH[i_bnd, :, :, :, i_gpb] = spectra.k_rayleigh[:, i_gpt, :, :] * C_K_PRESCALE
bake_const(C_K_RAYLEIGH, 'C_K_RAYLEIGH')

# 4. Solar source

C_MG_DEFAULT = params.variables["mg_default"][()]
C_SB_DEFAULT = params.variables["sb_default"][()]
C_TSI_DEFAULT = params.variables["tsi_default"][()]

C_E_SOLAR = source.query(C_MG_DEFAULT, C_SB_DEFAULT)
C_E_SOLAR = (C_E_SOLAR / np.sum(C_E_SOLAR)) * C_TSI_DEFAULT
bake_const(C_E_SOLAR, 'C_E_SOLAR')

output.close()
