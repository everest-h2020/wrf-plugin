from rrtmgp.reference import *
import matplotlib.pyplot as plt

REAL    = np.float32

params      = Dataset('./data/rrtmgp-data-sw-g224-2018-12-04.nc', 'r')
absorbers   = AbsorberRegistry(params)
profile     = ReferenceProfile(params, absorbers)
spectra     = AbsorptionSpectra(params, absorbers)

input   = Dataset('./data/wrf-input.nc', 'r')
T       = input.variables['tlay'][:,0].astype(REAL)
p       = input.variables['play'][:,0].astype(REAL)
n_prime = np.transpose(input.variables['col_gas'][:,:,0], (1, 0)).astype(REAL)

n_prime[:,8:] = 0

# Perform interpolation.
T_prime, p_prime, eta, i_strato, n_prime_mix = interpolate(profile, T, p, n_prime)

# Split interpolation indices from fractions.
f_T, j_T        = np.fmod(T_prime, 1), T_prime.astype(np.int32)
f_p, j_p        = np.fmod(p_prime, 1), p_prime.astype(np.int32)
f_eta, j_eta    = np.fmod(eta, 1), eta.astype(np.int32)

# NOTE: Weird numerically unstable schemes from original code:
if False:
    f_T = (T[:] - profile.T[j_T[:]]) / profile.T_delta
    j_p = 1 + (np.log(p) - profile.p_log_max) / profile.p_log_delta
    f_p = j_p - j_p.astype(np.int32)
    j_p = j_p.astype(np.int32) - 1

# Compute the interpolation coefficients.
f_minor     = (np.stack((1-f_T,f_T), -1)[:,None,:,None]
               * np.stack((1-f_eta,f_eta), -1))
f_major     = (np.stack((1-f_p,f_p), -1)[:,None,:,None,None] * f_minor[...,None,:,:])

# Compute the major absorber contributions.
tau_maj     = tau_major(spectra, i_strato, j_p, j_T, j_eta, f_major, n_prime_mix)

# Compute the minor absorber contributions.
tau_min     = tau_minor(spectra, p, T, i_strato, j_T, j_eta, f_minor, n_prime)

# Combine gas taus.
tau_gas     = tau_maj + tau_min

output      = Dataset("./data/ref-output.nc", "w")

output.createDimension('nlay', j_T.shape[0])
output.createDimension('ngpt', tau_gas.shape[0])

output.createVariable('tau_maj', tau_maj.dtype, ('ngpt', 'nlay'))[...] = tau_maj
plt.plot(np.reshape(tau_maj, (-1)), label='maj')
output.createVariable('tau_min', tau_min.dtype, ('ngpt', 'nlay'))[...] = tau_min
plt.plot(np.reshape(tau_min, (-1)), label='min')

if hasattr(spectra, 'k_rayleigh'):
    # Compute rayleigh scattering tau.
    tau_rayl    = tau_rayleigh(spectra, i_strato, j_T, j_eta, f_minor, n_prime)
    output.createVariable('tau_rayl', tau_rayl.dtype, ('ngpt', 'nlay'))[...] = tau_rayl
    plt.plot(np.reshape(tau_rayl, (-1)), label='rayl')

plt.yscale('log')
plt.legend()

output.close()
plt.show(block=True)
