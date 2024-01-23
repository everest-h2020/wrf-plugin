import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

REAL    = np.float32

expect  = Dataset("./data/ref-output.nc", "r")
got     = Dataset("./output.nc", "r")

tau_maj_ref = expect.variables["tau_maj"][...]
tau_min_ref = expect.variables["tau_min"][...]
tau_gas_ref = tau_maj_ref + tau_min_ref
tau_gas_got = got.variables["tau_gas"][...]

tau_gas_err = tau_gas_got - tau_gas_ref

plt.plot(np.reshape(tau_gas_err / tau_gas_ref, (-1)), label='gas')

if "tau_rayl" in got.variables:
    tau_rayl_ref = expect.variables["tau_rayl"][...]
    tau_rayl_got = got.variables["tau_rayl"][...]
    tau_rayl_err = tau_rayl_got - tau_rayl_ref

    plt.plot(np.reshape(tau_rayl_err / tau_rayl_ref, (-1)), label='rayl')

plt.legend()
plt.show()
