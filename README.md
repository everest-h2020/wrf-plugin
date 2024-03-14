![EVEREST logo](img/logo_horiz_positive.png)

This software bundle is part of the [EVEREST][1] project, funded under an [EU grant][2].

# WRF RRTMG plugin

This is the main repository of the `plugin_rrtmg` project, which produces a dynamic library of the same name that dispatches class to the RRTMG submodule in WRF.

## Building

A two-step build process is required, which involves the use of **Python** (version `3.10` or newer). First, setup the virtual environment and install the Python dependencies.

```sh
# Setup the python virtual environment.
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To build the plugin, the constant data must be baked. This is achieved by running the following command in the virtual environment:

```sh
python bake.py
```

The `plugin_rrtmg` project is built using **CMake** (version `3.20` or newer).
Make sure to provide all dependencies required by the project, either by installing them to system-default locations, or by setting the appropriate search location hints!

| Dependency       | `apt` Package name |
| ---------------- | --- |
| `libnetcdf`      | `libnetcdf-dev` |
| `libnetcdf_c++4` | `libnetcdf-c++4-dev` |

```sh
# Configure.
cmake -S . -B build -G Ninja

# Build.
cmake --build build
```

## Testing

An additional **CMake** target called `runner` is provided, which invokes the plugin on pre-recorded test data (`data/wrf-input.nc`). The `runner` executable should be executed from within the project root directory, where it will produce an `output.nc` file.

Assuming the virtual environment is activated, the following commands will perform verification:

```sh
# Produces data/ref-output.nc using the Python reference.
python make_reference.py

# Execute the runner.
build/runner

# Compare the actual with the expected result.
python compare.py
```

## License

This project is licensed under the ISC license.

---

![EU notice](img/eu_banner.png)

[1]: https://everest-h2020.eu/
[2]: https://cordis.europa.eu/project/id/957269
