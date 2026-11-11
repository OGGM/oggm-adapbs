# oggm-adapbs

Bayesian calibration of the [Open Global Glacier Model (OGGM)](https://oggm.org)
with the adaptive particle batch smoother (AdaPBS).

Instead of one "best" parameter set, the calibration returns weighted parameter
samples of the posterior distribution, given a prior and observations with
errors. The AdaPBS is adaptive multiple importance sampling: it runs the model on
batches of parameter samples ("particles") and moves the proposal distribution
towards the posterior until the weights are no longer degenerate.

**Status: experimental and temporary.** The AdaPBS code is adapted from
[MuSA](https://github.com/ealonsogzl/MuSA), which is licensed under the GPL-3.0.
OGGM is BSD-3-Clause, so the AdaPBS lives in this separate GPL package for now.
We expect to replace it with a lightweight, permissively licensed package by the
MuSA authors when it becomes available.

## Citation

The AdaPBS algorithm and its implementation are the work of the MuSA authors.
If you use this package, please cite:

- Aalstad, K., Alonso-González, E., Pirk, N., Westermann, S., Willmes, C., and
  Yang, R.: Evolving beyond collapse: An adaptive particle batch smoother for
  cryospheric data assimilation, Geosci. Model Dev., 19, 8565–8596,
  https://doi.org/10.5194/gmd-19-8565-2026, 2026.
- Alonso-González, E., Aalstad, K., Baba, M. W., Revuelto, J., López-Moreno,
  J. I., Fiddes, J., Essery, R., and Gascoin, S.: The Multiple Snow Data
  Assimilation System (MuSA v1.0), Geosci. Model Dev., 15, 9127–9155,
  https://doi.org/10.5194/gmd-15-9127-2022, 2022.

and OGGM (see https://docs.oggm.org/en/stable/citing-oggm.html).

## Installation

This package needs the development version of OGGM (it uses features which are
not in a release yet). In an environment with OGGM's dependencies:

```
pip install git+https://github.com/OGGM/oggm.git
pip install git+https://github.com/OGGM/oggm-adapbs.git
```

or, for development, clone this repository and run `pip install -e .`

## Contents

- `oggm_adapbs.adapbs`: the AdaPBS algorithm (`run_adapbs`), with no OGGM
  dependency. You provide a `forward(theta, iteration)` function which returns
  the predicted observations for a batch of parameter sets. Differences with
  MuSA's implementation are listed in the module docstring.
- `oggm_adapbs.massbalance`: OGGM tools for the mass balance parameters
  (`melt_f`, `prcp_fac`, `temp_bias`) with a fixed geometry: the default prior
  (`mb_prior`), the MB model on the inversion flowlines (`static_mb_model`),
  winter, summer and annual MB of hydrological years (`seasonal_specific_mb`),
  and an entity task calibrating against geodetic MB
  (`mb_calibration_from_geodetic_mb_adapbs`).
- `example_workflows/`: notebooks.
  - `wgms_static_calibration.ipynb`: the AdaPBS version of OGGM's
    `mb_calibration_to_rmsd`, against WGMS mean, annual, winter and summer MB.

Example:

```python
from oggm_adapbs.massbalance import mb_calibration_from_geodetic_mb_adapbs

ds = mb_calibration_from_geodetic_mb_adapbs(gdir, temp_bias_file_path=...,
                                            n_particles=200, seed=0)
# ds: all the particles, their posterior weights and predicted MB
```

## Tests

The tests use OGGM's test fixtures and options (e.g. the Hintereisferner glacier
directory, `--run-slow`):

```
pytest --run-slow oggm_adapbs
```

The test of the AMIS weights against MuSA's own function needs `MUSA_DIR`: a
clone of MuSA, or a folder made with `ci/get_musa.sh <dir> [<ref>]`. Locally it
is skipped without it; on CI it must run. CI tests against the MuSA commit the
AdaPBS was ported from, plus a job against MuSA master which is allowed to fail.

## License

GPL-3.0, see [LICENSE](LICENSE).
