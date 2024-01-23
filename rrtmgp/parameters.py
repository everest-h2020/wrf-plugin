import numpy as np
from netCDF4 import Dataset

def ndarray_to_str(a: np.ndarray):
    """Convert a right-padded array of ASCII chars to a string."""
    if len(a.shape) == 1:
        return a.tobytes().decode('ASCII').strip()
    return [ndarray_to_str(x) for x in a]

class Identifiers:
    """Stores an ordered set of string identifiers with associated indices."""
    def __init__(self, items: 'iter[str]', start: int = 0):
        self.items      = {}
        """Lookup of identifiers to indices."""
        self.next_id    = start
        """Index of the next identifier to be added."""
        self.get_or_add(items)

    def get_or_add(self, id):
        """Gets the index of `id`, adding it at the end if necessary."""
        if isinstance(id, str):
            next_id = self.next_id
            id = self.items.setdefault(id, next_id)
            if id == next_id:
                self.next_id = next_id + 1
            return id
        return [self.get_or_add(x) for x in id]

    def __str__(self):
        """Gets a string representation of all identifiers."""
        return self.items.__str__()

    def __repr__(self):
        """Gets a string representation of all identifiers."""
        return self.items.__repr__()

    def __getitem__(self, key):
        """Gets the id or index for the given input."""
        if np.issubdtype(type(key), np.integer):
            for id, idx in self.items.items():
                if idx == key:
                    return id
            return None
        if isinstance(key, str):
            return self.items.get(key, None)
        return [self[x] for x in key]

    def __iter__(self):
        """Iterates over all ids."""
        return self.items.__iter__()

    def __len__(self):
        """Gets the number of identifiers."""
        return len(self.items)

class AbsorberRegistry:
    """Registry of major and minor absorbers defined in a constants file."""
    def __init__(self, const: Dataset):
        # 1. Query dimensions

        self.N_abs          = const.dimensions['absorber'].size
        """Total number of absorbers (gases)."""
        self.N_mabs         = const.dimensions['minor_absorber'].size
        """Number of minor absorbers (trace gases)."""
        self.N_mabsi        = (
            const.dimensions['minor_absorber_intervals_lower'].size,
            const.dimensions['minor_absorber_intervals_upper'].size)
        """Number of minor absorber intervals ([lower, upper] atmosphere)"""

        # 2. Query identifier sets

        self.abs_names      = Identifiers(
            ndarray_to_str(const.variables['gas_names']), start=1)
        """Set of absorber identifiers."""
        self.mabs_to_abs    = ndarray_to_str(const.variables['gas_minor'])
        """Lookup of minor absorber indices to absorber names."""
        self.mabsi_to_mabs  = Identifiers(
            ndarray_to_str(const.variables['identifier_minor']))
        """Set of minor absorber identifiers."""
        self.mabsi_names    = (
            ndarray_to_str(const.variables['minor_gases_lower']),
            ndarray_to_str(const.variables['minor_gases_upper']))
        """Lookup of minor absorber interval indices to minor absorber names."""

        self.i_h2o          = self.abs_names['h2o']
        """Absorber index of water vapor."""

        # 3. Resolve minor absorber intervals to absorber identifiers

        # Resolve minor absorber intervals to minor absorber index.
        i_mabs              = (
            self.mabsi_to_mabs[self.mabsi_names[0]],
            self.mabsi_to_mabs[self.mabsi_names[1]])
        self.mabsi_to_abs   = (
            self.abs_names[[self.mabs_to_abs[x] for x in i_mabs[0]]],
            self.abs_names[[self.mabs_to_abs[x] for x in i_mabs[1]]])
        """Lookup of minor absorber intervals to absorber indices."""

        # 4. Build flavor set

        # NOTE: For comparison, it is important that the flavor collection is
        #       stable over the key_species list.
        flavors             = list(dict.fromkeys([
           (tuple(x) if np.any(x != 0) else (2, 2))
           for x in np.reshape(const.variables['key_species'], (-1, 2))[:,:]]))
        self.flav_to_abs    = np.array(flavors, np.int32)
        """Lookup of flavor indices to absorber indices."""
        self.N_flav         = len(self.flav_to_abs)
        """Number of flavors."""

    def get_flavor(self, flav, default: int = -1):
        """Obtains the index of the flavor for the given absorbers."""
        if isinstance(flav, tuple):
            a, b = flav
            if isinstance(a, str):
                a = self.abs_names[a]
            if isinstance(b, str):
                b = self.abs_names[b]
            if (a, b) == (0, 0):
                a, b = 2, 2
            for i_flav in range(self.N_flav):
                if tuple(self.flav_to_abs[i_flav]) == (a, b):
                    return i_flav
            return default
        return np.apply_along_axis(
            lambda x: self.get_flavor(tuple(x)), -1, flav)

    def is_major(self, abs):
        """Determines whether the given absorbers are major species."""
        if np.issubdtype(type(abs), np.integer):
            return (self.flav_to_abs == abs).any()
        if isinstance(abs, str):
            return self.is_key(self.abs_names[abs])
        if abs is None:
            return False
        return [self.is_key(x) for x in abs]

    def __iter__(self):
        """Iterates over all absorber names."""
        return self.abs_names.__iter__()

    def __len__(self):
        """Gets the number of absorbers."""
        return len(self.abs_names)

class ReferenceProfile:
    """Parameterization of the reference profile for the constant tables."""
    def __init__(self, const: Dataset, abs: AbsorberRegistry):
        self.absorbers  = abs

        # 1. Query dimensions

        self.N_T        = const.dimensions['temperature'].size
        """Number of temperature reference points."""
        self.N_p        = const.dimensions['pressure'].size
        """Number of pressure reference points."""
        self.N_eta      = const.dimensions['mixing_fraction'].size
        """Number of mixing fraction reference points."""

        # If i is a projected index, i AND (i+1) must be dereferencable. Thus,
        # the inclusive upper limit is the predecessor of (N-1).
        # TODO: The original code used (N-2) as the upper limit, but only on the
        #       integer part. As a result, a value of (N-1) turned into (N-2)
        #       plus a fraction of 1.0, whereas this approach yields (N-2) plus
        #       a fraction of 1.0-eps. What is the difference?

        self.lim_T      = np.nextafter(self.N_T - 1, 0)
        """Upper limit (inclusive) for projected temperatures."""
        self.lim_p      = np.nextafter(self.N_p - 1, 0)
        """Upper limit (inclusive) for projected pressures."""
        self.lim_eta    = np.nextafter(self.N_eta - 1, 0)
        """Upper limit (inclusive) for the mixing fraction."""

        # 2. Parameterize reference profiles

        # Parameterize the temperature profile (which must be linear).
        self.T              = const.variables['temp_ref'][:]
        """Reference temperature profile."""
        self.T_min          = self.T[0]
        """Minimum reference temperature."""
        self.T_delta        = (self.T[-1] - self.T[0]) / (self.N_T - 1)
        """Step between reference temperatures."""
        T_expect            = self.T_min + np.arange(self.N_T) * self.T_delta
        assert (np.abs(self.T - T_expect) == 0).all()

        # Parameterize the pressure profile (which must be exponential).
        self.p              = const.variables['press_ref'][:]
        """Reference pressure profile."""
        self.p_log_max      = np.log(self.p[0])
        """Maximum logairthm of reference pressure."""
        self.p_log_delta    = (np.log(self.p[-1]) - self.p_log_max) / (self.N_p - 1)
        """Logarithmical step between reference pressures."""
        self.p_tropo        = const.variables['press_ref_trop'][0]
        """Troposphere cutoff (inclusive lower bound) pressure."""
        # NOTE: The pressure profile given in the constant files significantly
        #       deviates from the assumption... But it is still used in the
        #       reference code!
        p_expect            = np.exp(np.arange(self.N_p) * self.p_log_delta + self.p_log_max)
        assert (np.abs(self.p - p_expect) < 1e-8).all()

        # 3. Precompute binary species parameter values

        self.tiny           = np.finfo(np.double).eps * 2.0
        """Lower bound for divisible floating-point numbers."""

        self.r_ref          = const.variables['vmr_ref'][:,:,:]
        """Reference volume mixing ratios for all absorbers."""
        self.eta_half       = np.transpose(
            self.r_ref[:,self.absorbers.flav_to_abs[:,0],:]
            / self.r_ref[:,self.absorbers.flav_to_abs[:,1],:]
        , (2, 0, 1))
        """Reference mixing ratio between major absorbers of all flavors."""

class AbsorptionSpectra:
    """Lookup tables for absorption coefficients."""
    def __init__(self, const: Dataset, abs: AbsorberRegistry):
        self.absorbers = abs

        # 1. Query dimensions

        self.N_gpt          = const.dimensions['gpt'].size
        """Number of g-points (quadrature points)."""
        self.N_bnd          = const.dimensions['bnd'].size
        """Number of bands (wavelength intervals)."""

        # 2. Recover g-point parameterization

        self.bnd_limits_gpt = const.variables['bnd_limits_gpt'][...]

        self.gpt_to_bnd     = np.zeros((self.N_gpt), dtype=np.int32)
        """Maps from g-points to band numbers."""
        self.bnd_width      = np.zeros((self.N_bnd), dtype=np.int32)
        """Maps from band number to number of g-points."""
        self.gpt_to_wvn     = np.zeros((self.N_gpt))
        """Maps from g-points to wavenumbers."""
        for i_bnd in range(self.N_bnd):
            gpt_rg = self.bnd_limits_gpt[i_bnd]
            i_gpt = range(gpt_rg[0]-1, gpt_rg[1])
            self.gpt_to_bnd[i_gpt] = i_bnd
            self.bnd_width[i_bnd] = len(i_gpt)
            wvn_rg = const.variables['bnd_limits_wavenumber'][i_bnd]
            self.gpt_to_wvn[i_gpt] = np.linspace(wvn_rg[0], wvn_rg[1], len(i_gpt))

        # 3. Recover binary species parametrization

        # Map from layer and g-points to flavor indices.
        self.bnd_to_flav    = np.stack((
            abs.get_flavor(const.variables['key_species'][:,0]),
            abs.get_flavor(const.variables['key_species'][:,1])))
        """Maps from band numbers to flavor indices."""

        # 4. Recover minor absorber compression

        self.mabsi_to_bnd   = (
            np.array([
                np.where(self.bnd_limits_gpt == x)[0][0]
                for x in const.variables['minor_limits_gpt_lower']],
                dtype=np.int32),
            np.array([
                np.where(self.bnd_limits_gpt == x)[0][0]
                for x in const.variables['minor_limits_gpt_upper']],
                dtype=np.int32))
        """Maps from minor absorber interval to band number."""

        self.mabsi_limits = (
            const.variables['minor_limits_gpt_lower'][...],
            const.variables['minor_limits_gpt_upper'][...])

        # 5. Query lookup tables

        self.k_major        = np.transpose(
            const.variables['kmajor'][...],
            (3, 1, 0, 2))
        """Major species absorption coefficients."""
        self.k_minor        = (
            np.transpose(const.variables['kminor_lower'][...], (2, 0, 1)),
            np.transpose(const.variables['kminor_upper'][...], (2, 0, 1)))
        """Minor species absorption coefficients."""
        self.mabsi_to_mci   = (
            const.variables['kminor_start_lower'][...]-1,
            const.variables['kminor_start_upper'][...]-1)
        """Maps minor absorber interval index to contributor offset."""

        self.mabsi_compl    = (
            const.variables['scale_by_complement_lower'][...],
            const.variables['scale_by_complement_upper'][...])
        """Indicates whether a minor absorber scales by complement."""
        self.mabsi_density  = (
            const.variables['minor_scales_with_density_lower'][...],
            const.variables['minor_scales_with_density_upper'][...])
        """Indicates whether a minor absorber scales with density."""
        self.mabsi_scale_by = (
            abs.abs_names[ndarray_to_str(const.variables['scaling_gas_lower'])],
            abs.abs_names[ndarray_to_str(const.variables['scaling_gas_upper'])])
        """Maps minor absorber to companion gas it scales by."""

        if 'rayl_lower' in const.variables:
            self.k_rayleigh = np.stack((
                np.transpose(const.variables['rayl_lower'][...], (2, 0, 1)),
                np.transpose(const.variables['rayl_upper'][...], (2, 0, 1))))
            """Rayleigh scattering absorption coefficients."""
