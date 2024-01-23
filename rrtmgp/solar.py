import numpy as np
from netCDF4 import Dataset
from datetime import date, timedelta

class SolarCycle:
    def __init__(self, const: Dataset):
        self.lookup = const.variables['solar_var_avgcyc']
        self.ref_date = date(2008, 1, 4)
        self.ref_len = timedelta(days=365.25)

        self.N_points = self.lookup.shape[0]
        self.interval = 1.0 / (self.N_points - 2)
        self.half_interval = self.interval / 2.0

    def to_cycle_frac(self, now: date):
        delta = now - self.ref_date
        return (delta / self.ref_len) % 1

    def project(self, cycle_frac):
        if cycle_frac <= 0:
            return self.lookup[0,:]
        if cycle_frac >= 1:
            return self.lookup[-1,:]

        if cycle_frac <= self.half_interval:
            sfid = 1
            fraclo = 0
            frachi = self.half_interval
        elif cycle_frac >= (1-self.half_interval):
            sfid = self.N_points - 1
            fraclo = 1 - self.half_interval
            frachi = 1
        else:
            sfid = int((cycle_frac - self.half_interval) * (self.N_points - 2)) + 2
            fraclo = (sfid-2) * self.interval + self.half_interval
            frachi = fraclo + self.interval

        intfrac = (cycle_frac - fraclo) / (frachi - fraclo)
        lo, hi = self.lookup[sfid-1,:], self.lookup[sfid,:]
        mg_index = lo[0] + intfrac * (hi[0] - lo[0])
        sb_index = lo[1] + intfrac * (hi[1] - lo[1])
        return np.array((mg_index, sb_index))

class SolarSource:
    def __init__(self, const: Dataset):
        self.quiet = const.variables['solar_source_quiet'][...]
        self.facular = const.variables['solar_source_facular'][...]
        self.sunspot = const.variables['solar_source_sunspot'][...]

        self.mg_default = const.variables['mg_default'][()]
        self.sb_default = const.variables['sb_default'][()]

        self.a_offset = 0.1495954
        self.b_offset = 0.00066696

    def query(self, mg=None, sb=None):
        mg = mg or self.mg_default
        sb = sb or self.sb_default
        return (self.quiet
                + (mg - self.a_offset) * self.facular
                + (sb - self.b_offset) * self.sunspot)
