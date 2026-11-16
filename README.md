# oggm-adapbs

Bayesian calibration of the [Open Global Glacier Model (OGGM)](https://oggm.org)
with the adaptive particle batch smoother (AdaPBS).

Instead of one "best" parameter set, the calibration returns weighted parameter
samples of the posterior distribution, given a prior and observations with
errors. AdaPBS is adaptive multiple importance sampling: it runs the model on
batches of parameter samples ("particles") and moves the proposal distribution
towards the posterior until the weights are no longer degenerate.

**Status: experimental and temporary.** The AdaPBS code is adapted from
[MuSA](https://github.com/ealonsogzl/MuSA), which is licensed under the GPL-3.0.
OGGM is BSD-3-Clause, so AdaPBS lives in this separate GPL package for now.
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
- `oggm_adapbs.dynamic` (experimental): how many parameter sets of OGGM's
  usual workflow (MB calibration, ice thickness inversion, dynamic spinup) are
  consistent with the observations, and how different are their glaciers?
  The MB parameters, Glen A and the initial state of the spinup vary
  together; AdaPBS keeps the plausible parameter sets (as GLUE would, but
  more efficiently). Each particle inverts the ice thickness with its own MB
  parameters and Glen A, makes a starting glacier in a constant climate with a
  temperature offset (the "initial state period"), then runs with the
  historical climate. Observations: geodetic MB, RGI area and IceBoost mean
  thickness or volume (`dynamic_equifinality_adapbs`, with a pool of
  processes). The plausible runs can be rerun with OGGM's hydrological output
  (`select_particles`, `dynamic_hydro_rerun`) for runoff metrics
  (`runoff_metrics`).
- `example_workflows/`: notebooks.
  - `wgms_static_calibration.ipynb`: the AdaPBS version of OGGM's
    `mb_calibration_to_rmsd` (annual MB), then with the winter and summer MB.
  - `geodetic_equifinality.ipynb`: calibration of three parameters against
    one geodetic MB value, with a flat prior: equifinality, information gain,
    and the role of the prior.
  - `dynamic_equifinality.ipynb` (experimental): the plausible parameter sets
    of a glacier (MB parameters, Glen A, initial state of the spinup) given
    the geodetic MB, the RGI area and the IceBoost mean thickness, with wide
    priors and dynamical runs in parallel; compared with OGGM's dynamic
    calibration.
  - `dynamic_runoff_sensitivity.ipynb` (experimental): the plausible runs
    of the previous notebook rerun with OGGM's hydrological output; runoff
    metrics (mean runoff, seasonality, share of glacier melt) and a simple
    analysis of which parameters drive their spread.
- `scripts/`: the same workflow for one glacier from the command line
  (`run_glacier.py`), and slurm scripts to run many glaciers on a cluster
  (see `scripts/README.md`).

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
