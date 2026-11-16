"""Equifinality of OGGM's dynamical runs, explored with AdaPBS (experimental).

The question: how many different parameter sets of OGGM are consistent with
the observations we have, and how different are the glaciers (and their
runoff) they produce?

OGGM prepares a glacier for a simulation in three steps, each with its own
parameters: the MB calibration (``melt_f``, ``prcp_fac``, ``temp_bias``), the
ice thickness inversion (Glen's A) and the dynamic spinup, which needs a
glacier state at the start of the historical run. Here, we let all these
parameters vary at once and keep the parameter sets whose glacier matches
three observations: the geodetic MB (Hugonnet et al., 2021), the RGI area and
the IceBoost v2 mean thickness (volume / area) or volume, at ``state_year``
(the RGI date + 1, i.e. the end of the RGI year).

This is the idea of GLUE (keep the "plausible" or "behavioural" runs), which
can be done with a Latin hypercube sample of the parameters and a threshold
on the misfit. AdaPBS does the same job more efficiently: with a wide,
nearly uniform prior, its posterior weights say how plausible each parameter
set is, and its iterations avoid spending runs where the model cannot fit the
observations. Parameters which the observations do not constrain are
expected: they are what we want to find.

Each particle (parameter set) of :py:func:`run_dynamic_particle`:

1. inverts the ice thickness at the RGI date, with its MB parameters (for the
   shape of the MB profile) and its own Glen A (no sliding);
2. makes a glacier state for the start of the historical run (the "initial
   state period"): the inverted glacier runs for ``initial_state_years`` in a
   constant climate (the mean climate of the spinup period) with its
   temperature bias plus an offset (``initial_state_temp_offset``), as in
   OGGM's dynamic spinup (where this temperature bias is found by
   minimisation instead). The offset makes the starting glacier larger or
   smaller. This is not a reconstruction of the past glacier, which present
   geometry and past climate cannot determine (Eis et al., 2019,
   https://doi.org/10.5194/tc-13-3317-2019; Eis et al., 2021,
   https://doi.org/10.3389/feart.2021.595755): it stands for our ignorance of
   the past state;
3. runs with the historical climate from ``state_year - spinup_years`` to the
   end of the geodetic MB period.

Parameters (any subset of ``DYNAMIC_PARAMS``): log10_melt_f and
log10_prcp_fac (multiplicative factors vary in log10 space; melt_f and
prcp_fac themselves are possible too), temp_bias, log10_glen_a_factor
(A = ``cfg.PARAMS['inversion_glen_a']`` * 10 ** log10_glen_a_factor, used for
the inversion and the dynamics) and initial_state_temp_offset (K, added to
temp_bias during the initial state period only).

Please cite the AdaPBS and MuSA papers (see :py:mod:`oggm_adapbs.adapbs`).
"""
import ast
import copy
import glob
import logging
import multiprocessing as mp
import os
import time
import warnings
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

from oggm import cfg, entity_task
from oggm.utils import get_geodetic_mb_dataframe, file_downloader
from oggm.exceptions import InvalidParamsError
from oggm.core.massbalance import (MonthlyTIModel, ConstantMassBalance,
                                   MultipleFlowlineMassBalance,
                                   apparent_mb_from_any_mb, calving_mb)
from oggm.core.inversion import (prepare_for_inversion,
                                 mass_conservation_inversion,
                                 filter_inversion_output)
from oggm.core.flowline import (init_present_time_glacier,
                                decide_evolution_model, run_with_hydro,
                                run_from_climate_data)

from oggm_adapbs.adapbs import run_adapbs, glogit
from oggm_adapbs.massbalance import MB_PARAMS, mb_prior

log = logging.getLogger(__name__)

# The parameters which vary. melt_f and prcp_fac are multiplicative factors,
# as Glen's A: by default they vary in log10 space (log10_melt_f,
# log10_prcp_fac), so that e.g. half and twice a value are equally far from it
LOG_MB_PARAMS = {'log10_melt_f': 'melt_f', 'log10_prcp_fac': 'prcp_fac'}
DYNAMIC_PARAMS = ('log10_melt_f', 'log10_prcp_fac', 'temp_bias',
                  'log10_glen_a_factor', 'initial_state_temp_offset')
# Also possible: melt_f and prcp_fac themselves instead of their log10
ALL_DYNAMIC_PARAMS = DYNAMIC_PARAMS + ('melt_f', 'prcp_fac')


def mb_param_values(params):
    """melt_f, prcp_fac and temp_bias of a particle (from their log10 if
    the particle has them)."""
    out = {}
    for p in MB_PARAMS:
        lp = f'log10_{p}'
        out[p] = float(10 ** params[lp]) if lp in params else float(params[p])
    return out

# IceBoost v2 tables with the volume errors (vol_err_km3). The tables OGGM
# uses for the inversion (oggm.workflow.ICEBOOST_V2_FILES, v20260701) have the
# same volumes but no errors.
ICEBOOST_URL = ('https://cluster.klima.uni-bremen.de/~oggm/ice_thickness/'
                'iceboost_v2/')
ICEBOOST_FILES = {'6': 'iceboost_compiled_rgi60_v20260705.parquet',
                  '7G': 'iceboost_compiled_rgi70G_v20260705.parquet',
                  '7C': 'iceboostv2_compiled_rgi70C_v20260705.parquet'}

# The diagnostics a particle needs, and the thickness threshold of the area
DIAGNOSTIC_VARIABLES = ['volume', 'area', 'area_min_h', 'mass']
MIN_ICE_THICK_FOR_AREA = 2.

# The predictions of a particle (the task uses either volume or thickness)
PARTICLE_OBS = ['dmdtda', 'area', 'volume', 'thickness']

# OGGM's hydrological output (run_with_hydro) kept by the hydro reruns: the
# glacier and off-glacier areas of the catchment (m2), and the water fluxes
# in kg per year (or per month). Runoff is the sum of RUNOFF_VARIABLES.
HYDRO_VARIABLES = ['on_area', 'off_area',
                   'melt_on_glacier', 'melt_off_glacier',
                   'liq_prcp_on_glacier', 'liq_prcp_off_glacier',
                   'snowfall_on_glacier', 'snowfall_off_glacier']
RUNOFF_VARIABLES = ['melt_on_glacier', 'melt_off_glacier',
                    'liq_prcp_on_glacier', 'liq_prcp_off_glacier']


def _period_years(period):
    """'2000-01-01_2020-01-01' -> (2000, 2020)"""
    return tuple(int(p.split('-')[0]) for p in period.split('_'))


def iceboost_volume(gdir, file_path=None, return_area=False):
    """IceBoost v2 volume of a glacier and its error (km3).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    file_path : str, optional
        a table (parquet or csv, path or URL) indexed by RGI id, with
        ``vol_km3`` and ``vol_err_km3`` columns (and ``area_km2`` for
        return_area). Default: the IceBoost v2 table of the glacier's RGI
        version.
    return_area : bool
        also return the glacier area of the table (km2)

    Returns
    -------
    (volume, error) in km3, and the area in km2 if return_area
    """
    if file_path is None:
        v = gdir.rgi_version
        key = '6' if v[0] == '6' else v[0] + v[-1]
        if key not in ICEBOOST_FILES:
            raise InvalidParamsError(f'No IceBoost table for RGI version {v}.')
        file_path = ICEBOOST_URL + ICEBOOST_FILES[key]
    if '://' in str(file_path):
        file_path = file_downloader(file_path)
    if str(file_path).endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, index_col=0)
    cols = ['vol_km3', 'vol_err_km3'] + (['area_km2'] if return_area else [])
    for c in cols:
        if c not in df.columns:
            raise InvalidParamsError(f'The volume table has no {c} column: '
                                     f'{file_path}')
    if gdir.rgi_id not in df.index:
        raise InvalidParamsError(f'{gdir.rgi_id} not in {file_path}.')
    row = df.loc[gdir.rgi_id]
    out = (float(row['vol_km3']), float(row['vol_err_km3']))
    if return_area:
        out += (float(row['area_km2']),)
    return out


def terminus_step_km2(fls, n_last=5, min_h=MIN_ICE_THICK_FOR_AREA):
    """The area of one grid point at the terminus (km2).

    The model area changes one grid point at a time, so it cannot match an
    area target more closely than that. This is the largest area among the
    last ``n_last`` grid points thicker than ``min_h`` of the main (last)
    flowline.

    Parameters
    ----------
    fls : list of flowlines
        model flowlines (e.g. ``gdir.read_store('model_flowlines')``)
    n_last : int
        number of grid points at the terminus to consider
    min_h : float
        thickness threshold (m), as for the model area (``area_min_h``)
    """
    fl = fls[-1]
    a = fl.bin_area_m2[fl.thick > min_h] * 1e-6
    return float(a[-n_last:].max()) if len(a) else 0.


def area_error_km2(area_km2, step_km2, area_err=0.05, area_k=0.25):
    """Error (1 sigma) of the area target (km2).

    sigma^2 = (area_err * area)^2 + area_k * step^2: a relative error (outline
    and date uncertainty) and a discretisation term (``terminus_step_km2``).
    With area_k = 0.25, the discretisation sigma is half a grid point.
    """
    return float(np.sqrt((area_err * area_km2) ** 2 + area_k * step_km2 ** 2))


def dynamic_prior(gdir, calibrate_params=DYNAMIC_PARAMS, *,
                  prior_centre=None,
                  prior_bounds=None,
                  temp_bias_file_path=None,
                  prcp_fac_range=2.,
                  temp_bias_range=3.,
                  glen_a_range=1.,
                  initial_state_temp_range=2.):
    """The default prior of :py:func:`dynamic_equifinality_adapbs`.

    The MB parameters as in :py:func:`oggm_adapbs.massbalance.mb_prior`
    (log10_melt_f and log10_prcp_fac: the log10 of its centres and bounds),
    and:

    - log10_glen_a_factor: centre 0 (OGGM's default
      ``cfg.PARAMS['inversion_glen_a']``, not the regionally calibrated value
      of the glacier settings), bounds +- glen_a_range (default: A / 10 to
      A * 10)
    - initial_state_temp_offset: centre 0, bounds +-
      initial_state_temp_range (K)

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    calibrate_params : list of str
        the parameters which vary (any of ``ALL_DYNAMIC_PARAMS``, but not
        e.g. melt_f and log10_melt_f together)
    prior_centre : dict, optional
        override the centre of some parameters (in log10 for the log10
        parameters)
    prior_bounds : dict, optional
        override the bounds of some parameters (in log10 for the log10
        parameters)
    temp_bias_file_path : str, optional
        temperature bias file, for the centre of the temp_bias prior
    prcp_fac_range, temp_bias_range : float
        see :py:func:`oggm_adapbs.massbalance.mb_prior`
    glen_a_range : float
        range of the log10_glen_a_factor bounds around the centre
    initial_state_temp_range : float
        range of the initial_state_temp_offset bounds around the centre (K)

    Returns
    -------
    a dict as :py:func:`oggm_adapbs.massbalance.mb_prior`, plus
    ``glen_a_ref`` (the A of log10_glen_a_factor = 0). ``centre`` has one
    form of melt_f and prcp_fac: the linear one if it varies, else the
    log10 one.
    """
    calibrate_params = list(calibrate_params)
    for p in calibrate_params:
        if p not in ALL_DYNAMIC_PARAMS:
            raise InvalidParamsError(f'Unknown parameter: {p}. Use one of '
                                     f'{ALL_DYNAMIC_PARAMS}.')
    for lp, p in LOG_MB_PARAMS.items():
        if lp in calibrate_params and p in calibrate_params:
            raise InvalidParamsError(f'{p} and {lp} cannot both vary.')
    prior_centre = dict(prior_centre or {})
    prior_bounds = dict(prior_bounds or {})

    mb = mb_prior(gdir, MB_PARAMS,
                  prior_centre={k: v for k, v in prior_centre.items()
                                if k in MB_PARAMS},
                  prior_bounds={k: v for k, v in prior_bounds.items()
                                if k in MB_PARAMS},
                  temp_bias_file_path=temp_bias_file_path,
                  prcp_fac_range=prcp_fac_range,
                  temp_bias_range=temp_bias_range)
    centre = dict(mb['centre'])
    bounds = {p: (mb['x_min'][k], mb['x_max'][k])
              for k, p in enumerate(MB_PARAMS)}
    for lp, p in LOG_MB_PARAMS.items():
        centre[lp] = float(np.log10(centre[p]))
        bounds[lp] = tuple(float(b) for b in np.log10(bounds[p]))

    centre['log10_glen_a_factor'] = 0.
    centre['initial_state_temp_offset'] = 0.
    for k, v in prior_centre.items():
        if k not in MB_PARAMS:
            centre[k] = float(v)
    bounds['log10_glen_a_factor'] = (centre['log10_glen_a_factor'] -
                                     glen_a_range,
                                     centre['log10_glen_a_factor'] +
                                     glen_a_range)
    c = centre['initial_state_temp_offset']
    bounds['initial_state_temp_offset'] = (c - initial_state_temp_range,
                                           c + initial_state_temp_range)
    for k, v in prior_bounds.items():
        if k not in MB_PARAMS:
            bounds[k] = v
    # One form of melt_f and prcp_fac for the particles
    for lp, p in LOG_MB_PARAMS.items():
        drop = lp if p in calibrate_params else p
        centre.pop(drop)
        bounds.pop(drop)

    x_min = np.array([bounds[p][0] for p in calibrate_params], dtype=float)
    x_max = np.array([bounds[p][1] for p in calibrate_params], dtype=float)
    x_c = np.array([centre[p] for p in calibrate_params], dtype=float)
    if np.any(x_c <= x_min) or np.any(x_c >= x_max):
        raise InvalidParamsError(f'Prior centre {x_c} not within the bounds '
                                 f'{x_min}, {x_max}.')
    return dict(params=calibrate_params, centre=centre,
                x_centre=x_c, x_min=x_min, x_max=x_max,
                glen_a_ref=float(cfg.PARAMS['inversion_glen_a']))


def _particle_files(gdir, filesuffix):
    return glob.glob(os.path.join(gdir.dir, f'*{filesuffix}.*'))


def _particle_start(gdir, params, s, *, state_year, spinup_years,
                    initial_state_years, ref_period, glen_a_ref,
                    settings_filesuffix='', volume_bounds=None,
                    extra_settings=None):
    """Steps 1 and 2 of a particle: inversion and initial state period.

    Writes the particle's settings and inversion files (suffix ``s``) and
    sets ``gdir.settings_filesuffix`` to ``s``: the caller resets it and
    deletes the files.

    Returns a dict with ``fls`` (the glacier at the start of the historical
    run), ``glen_a``, ``evolution_model``, ``model_kwargs``, ``ds_init``
    (diagnostics of the initial state period, or None),
    ``volume_inversion_km3``, ``volume_initial_state_km3`` and ``rejected``
    (the inverted volume is outside ``volume_bounds``: nothing else is done).
    """
    y0, y1 = ref_period
    ys = state_year - spinup_years
    glen_a = glen_a_ref * 10 ** params['log10_glen_a_factor']
    task_kwargs = dict(settings_filesuffix=s, add_to_log_file=False)
    mbp = mb_param_values(params)
    data = {**mbp,
            'store_diagnostic_variables': DIAGNOSTIC_VARIABLES,
            'min_ice_thick_for_area': MIN_ICE_THICK_FOR_AREA}
    data.update(extra_settings or {})
    gdir.create_new_settings(s, data=data,
                             parent_filesuffix=settings_filesuffix,
                             ignore_existing=True, overwrite=True)
    out = dict(fls=None, glen_a=glen_a, ds_init=None,
               volume_initial_state_km3=np.nan, rejected=False)

    # 1. Inversion at the RGI date with this particle's parameters
    fls_inv = gdir.read_store('inversion_flowlines',
                              filesuffix=settings_filesuffix)
    mb_inv = MultipleFlowlineMassBalance(
        gdir, settings_filesuffix=s, fls=fls_inv,
        mb_model_class=MonthlyTIModel, check_calib_params=False)
    apparent_mb_from_any_mb(gdir, input_filesuffix=settings_filesuffix,
                            output_filesuffix=s, mb_model=mb_inv,
                            mb_years=(y0, y1), **task_kwargs)
    prepare_for_inversion(gdir, input_filesuffix=s,
                          output_filesuffix=s, **task_kwargs)
    mass_conservation_inversion(gdir, input_filesuffix=s,
                                output_filesuffix=s, glen_a=glen_a,
                                fs=0., **task_kwargs)
    filter_inversion_output(gdir, input_filesuffix=s,
                            output_filesuffix=s, **task_kwargs)
    init_present_time_glacier(gdir, input_filesuffix=s,
                              output_filesuffix=s, **task_kwargs)
    fls = gdir.read_store('model_flowlines', filesuffix=s)
    v_inv = sum(fl.volume_m3 for fl in fls) * 1e-9
    out['volume_inversion_km3'] = v_inv
    if volume_bounds is not None and not (volume_bounds[0] <= v_inv
                                          <= volume_bounds[1]):
        out['rejected'] = True
        return out

    gdir.settings_filesuffix = s
    evolution_model = decide_evolution_model(gdir)
    model_kwargs = dict(glen_a=glen_a, fs=0., gdir=gdir,
                        settings_filesuffix=s)

    # 2. Initial state period: a constant climate (the mean of the spinup
    # period, as in OGGM's run_dynamic_spinup) with an offset on the
    # temperature bias makes the glacier at the start of the historical run
    if initial_state_years > 0:
        yc = (ys + state_year) / 2
        mb_init = MultipleFlowlineMassBalance(
            gdir, settings_filesuffix=s, fls=fls,
            mb_model_class=partial(ConstantMassBalance,
                                   mb_model_class=MonthlyTIModel),
            y0=yc, halfsize=state_year - yc, check_calib_params=False)
        mb_init.temp_bias = (mbp['temp_bias'] +
                             params['initial_state_temp_offset'])
        # (the constant climate does not depend on the year: the initial
        # state years are just before the historical run)
        model = evolution_model(copy.deepcopy(fls), mb_model=mb_init,
                                y0=ys - initial_state_years, **model_kwargs)
        ds_init = model.run_until_and_store(ys, diag_path=None,
                                            geom_path=False,
                                            fl_diag_path=False)
        if isinstance(ds_init, tuple):
            ds_init = ds_init[0]
        out['ds_init'] = ds_init
        fls = model.fls
    out['volume_initial_state_km3'] = sum(fl.volume_m3 for fl in fls) * 1e-9
    out.update(fls=fls, evolution_model=evolution_model,
               model_kwargs=model_kwargs)
    return out


def run_dynamic_particle(gdir, params, filesuffix, *,
                         state_year,
                         spinup_years,
                         initial_state_years,
                         ref_period,
                         glen_a_ref,
                         volume_bounds=None,
                         settings_filesuffix='',
                         keep_files=False):
    """One forward run of the model (one particle, i.e. one parameter set).

    Inversion, initial state period, historical run (see
    :py:mod:`oggm_adapbs.dynamic`).
    The particle writes its own settings and inversion files (with
    ``filesuffix``), so that several particles can run in parallel on the same
    glacier directory, and deletes them at the end. The model runs are kept
    in memory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    params : dict
        values of all ``DYNAMIC_PARAMS``
    filesuffix : str
        unique suffix of the particle's files (not empty)
    state_year : int
        the year of the area and volume targets (e.g. RGI date + 1)
    spinup_years : int
        the historical run starts in ``state_year - spinup_years``
    initial_state_years : int
        length of the initial state period (0: none, the historical run
        starts from the inverted glacier)
    ref_period : (int, int)
        the geodetic MB period (start year, end year, e.g. (2000, 2020))
    glen_a_ref : float
        the A of log10_glen_a_factor = 0
    volume_bounds : (float, float), optional
        if the inverted volume (km3) is outside these bounds, the particle
        is rejected before the dynamical runs (NaN predictions)
    settings_filesuffix : str
        the settings the particle's settings inherit from
    keep_files : bool
        keep the particle's files (for debugging)

    Returns
    -------
    a dict with ``pred`` (dmdtda in kg m-2 yr-1, area_min_h in km2, volume
    in km3 and mean thickness in m (volume / total area) at state_year,
    see ``PARTICLE_OBS``), ``volume_inversion_km3``,
    ``volume_initial_state_km3`` (at the end of the initial state period, i.e.
    at the start of the historical run), annual ``volume_km3``, ``area_km2``
    (area_min_h) and ``mass_kg`` (ice mass, for the specific MB) from the
    start of the initial state period (``initial_state_years`` before the
    historical run: these years have no calendar meaning, the climate is
    constant) to the end of the historical run, ``status`` ('ok',
    'rejected', 'disappeared' if there is no glacier left at state_year,
    or the error) and ``run_time`` (s). A glacier which disappears after
    state_year is a valid run.
    """
    if not filesuffix:
        raise InvalidParamsError('A particle needs a filesuffix.')
    t0 = time.time()
    y0, y1 = ref_period
    ys = state_year - spinup_years
    ye = max(state_year, y1)
    nyears = ye - ys + 1 + initial_state_years
    out = dict(pred=np.full(len(PARTICLE_OBS), np.nan),
               volume_inversion_km3=np.nan, volume_initial_state_km3=np.nan,
               volume_km3=np.full(nyears, np.nan),
               area_km2=np.full(nyears, np.nan),
               mass_kg=np.full(nyears, np.nan), status='ok')
    s = filesuffix
    orig_suffix = gdir.settings_filesuffix
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            st = _particle_start(gdir, params, s, state_year=state_year,
                                 spinup_years=spinup_years,
                                 initial_state_years=initial_state_years,
                                 ref_period=ref_period, glen_a_ref=glen_a_ref,
                                 settings_filesuffix=settings_filesuffix,
                                 volume_bounds=volume_bounds)
            out['volume_inversion_km3'] = st['volume_inversion_km3']
            if st['rejected']:
                out['status'] = 'rejected'
                return out
            out['volume_initial_state_km3'] = st['volume_initial_state_km3']

            # 3. Historical run
            fls = st['fls']
            mb_hist = MultipleFlowlineMassBalance(
                gdir, settings_filesuffix=s, fls=fls,
                mb_model_class=MonthlyTIModel, check_calib_params=False)
            model = st['evolution_model'](fls, mb_model=mb_hist, y0=ys,
                                          **st['model_kwargs'])
            ds = model.run_until_and_store(ye, diag_path=None, geom_path=False,
                                           fl_diag_path=False)
            if isinstance(ds, tuple):
                ds = ds[0]

        ds = ds.assign_coords(time=np.round(ds.time.values).astype(int))
        mass = ds['mass_kg']
        dm = float(mass.sel(time=y1) - mass.sel(time=y0))
        vol = float(ds['volume_m3'].sel(time=state_year))
        area = float(ds['area_m2'].sel(time=state_year))
        series = ds
        if st['ds_init'] is not None:
            # The last year of the initial state period is the first of
            # the historical run
            series = xr.concat([st['ds_init'].isel(time=slice(0, -1)), ds],
                               dim='time')
        out['volume_km3'] = series['volume_m3'].values * 1e-9
        out['area_km2'] = series['area_min_h_m2'].values * 1e-6
        out['mass_kg'] = series['mass_kg'].values
        if area == 0:
            # No glacier at the date of the observations: no thickness, and
            # no weight (a glacier which disappears later is a valid run)
            out['status'] = 'disappeared'
            return out
        out['pred'] = np.array([
            dm / gdir.rgi_area_m2 / (y1 - y0),
            float(ds['area_min_h_m2'].sel(time=state_year)) * 1e-6,
            vol * 1e-9,
            vol / area])
    except Exception as e:  # domain overflow, numerical errors...
        out['pred'] = np.full(len(PARTICLE_OBS), np.nan)
        out['status'] = f'{type(e).__name__}: {e}'[:200]
    finally:
        gdir.settings_filesuffix = orig_suffix
        if not keep_files:
            for f in _particle_files(gdir, s):
                os.remove(f)
        out['run_time'] = time.time() - t0
    return out


class _QuietLogger:
    """Silence an OGGM logger for a while (OGGM's run tasks warn when the
    run starts before the RGI date, which is what we want here)."""

    def __init__(self, name, level=logging.ERROR):
        self.logger = logging.getLogger(name)
        self.level = level

    def __enter__(self):
        self.orig = self.logger.level
        self.logger.setLevel(self.level)

    def __exit__(self, *args):
        self.logger.setLevel(self.orig)


def run_dynamic_particle_hydro(gdir, params, filesuffix, *,
                               state_year,
                               spinup_years,
                               initial_state_years,
                               ref_period,
                               glen_a_ref,
                               ref_area_yr=2000,
                               settings_filesuffix='',
                               keep_files=False):
    """Rerun one particle with OGGM's hydrological output (monthly).

    The same inversion and initial state period as
    :py:func:`run_dynamic_particle`, then the historical run through OGGM's
    ``run_with_hydro`` (``run_from_climate_data``, monthly output). The
    glacier evolution is the same as in :py:func:`run_dynamic_particle`, but
    this is about 2.5 times slower (the hydrological output recomputes the
    monthly MB), and writes files (deleted at the end).

    Runoff is OGGM's: melt and liquid precipitation, on and off the glacier,
    over a fixed catchment: the glacier area of ``ref_area_yr`` (by default
    2000, not the largest area of the run, which depends on the particle's
    initial state). In OGGM, the catchment of each elevation band is the
    largest of its reference area and its glacier area: in years when the
    glacier is larger than in ``ref_area_yr`` (often before), the catchment
    is a little larger too.

    Parameters
    ----------
    as :py:func:`run_dynamic_particle` (without ``volume_bounds``), and:
    ref_area_yr : int
        the year of the glacier area which defines the catchment

    Returns
    -------
    a dict with the annual ``volume_km3`` and ``area_km2`` (area_min_h) of
    the historical run, ``hydro`` (a dict of the annual and monthly
    hydrological variables, ``HYDRO_VARIABLES`` and the ``_monthly``
    versions of the fluxes with 12 calendar months, in kg per year or month;
    the last year is NaN), ``status`` and ``run_time``.
    """
    if not filesuffix:
        raise InvalidParamsError('A particle needs a filesuffix.')
    t0 = time.time()
    y0, y1 = ref_period
    ys = state_year - spinup_years
    ye = max(state_year, y1)
    if not ys <= ref_area_yr < ye:
        raise InvalidParamsError(f'ref_area_yr ({ref_area_yr}) must be within '
                                 f'the historical run ({ys}-{ye}).')
    nyears = ye - ys + 1
    out = dict(volume_km3=np.full(nyears, np.nan),
               area_km2=np.full(nyears, np.nan),
               hydro={**{v: np.full(nyears, np.nan)
                         for v in HYDRO_VARIABLES},
                      **{f'{v}_monthly': np.full((nyears, 12), np.nan)
                         for v in HYDRO_VARIABLES if 'area' not in v}},
               status='ok')
    s = filesuffix
    orig_suffix = gdir.settings_filesuffix
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            st = _particle_start(
                gdir, params, s, state_year=state_year,
                spinup_years=spinup_years,
                initial_state_years=initial_state_years,
                ref_period=ref_period, glen_a_ref=glen_a_ref,
                settings_filesuffix=settings_filesuffix,
                extra_settings={'store_model_geometry': True,
                                'store_diagnostic_variables':
                                    DIAGNOSTIC_VARIABLES + HYDRO_VARIABLES})
            fls = st['fls']
            mb_hist = MultipleFlowlineMassBalance(
                gdir, settings_filesuffix=s, fls=fls,
                mb_model_class=MonthlyTIModel, check_calib_params=False)
            with _QuietLogger('oggm.core.flowline'):
                run_with_hydro(gdir, settings_filesuffix=s,
                               run_task=run_from_climate_data,
                               store_monthly_hydro=True,
                               ref_area_yr=ref_area_yr,
                               init_model_fls=fls, mb_model=mb_hist,
                               ys=ys, ye=ye, output_filesuffix=s,
                               glen_a=st['glen_a'], fs=0.,
                               add_to_log_file=False)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix=s)) as ds:
            ds = ds.load()
        out['volume_km3'] = ds['volume_m3'].values * 1e-9
        out['area_km2'] = ds['area_min_h_m2'].values * 1e-6
        for v in out['hydro']:
            out['hydro'][v] = ds[v].values
    except Exception as e:  # domain overflow, numerical errors...
        out['status'] = f'{type(e).__name__}: {e}'[:200]
    finally:
        gdir.settings_filesuffix = orig_suffix
        if not keep_files:
            for f in _particle_files(gdir, s):
                os.remove(f)
        out['run_time'] = time.time() - t0
    return out


def _init_worker(cfg_contents):
    cfg.unpack_config(cfg_contents)
    logging.getLogger('oggm').setLevel(logging.ERROR)
    warnings.simplefilter('ignore', category=RuntimeWarning)
    # Preprocessed directories have pickle files: one warning per worker
    warnings.filterwarnings('ignore', message='Store data not found')


def _make_pool(n_workers):
    """A pool of processes for the particles, or None (run in this process)."""
    if n_workers is None or n_workers <= 1:
        return None
    return mp.get_context('spawn').Pool(n_workers, initializer=_init_worker,
                                        initargs=(cfg.pack_config(),))


def _run_job(args):
    func, gdir, params, filesuffix, kwargs = args
    return func(gdir, params, filesuffix, **kwargs)


def _map_jobs(pool, jobs, n_workers):
    if pool is None:
        return [_run_job(j) for j in jobs]
    chunksize = max(1, len(jobs) // (4 * n_workers))
    return pool.map(_run_job, jobs, chunksize=chunksize)


@entity_task(log)
def dynamic_equifinality_adapbs(gdir, *,
                                settings_filesuffix='',
                                state_year=None,
                                spinup_years=30,
                                initial_state_years=20,
                                ref_mb_period=None,
                                ref_mb=None,
                                ref_mb_err=None,
                                mb_file_path=None,
                                mb_err_factor=0.2,
                                ref_area=None,
                                area_err=0.05,
                                area_k=0.25,
                                volume_target='thickness',
                                ref_volume=None,
                                ref_volume_err=None,
                                volume_file_path=None,
                                volume_rel_err=0.2,
                                volume_err_factor=0.5,
                                reject_sigma=5.,
                                calibrate_params=DYNAMIC_PARAMS,
                                prior_centre=None,
                                prior_bounds=None,
                                prior_sd=1.,
                                temp_bias_file_path=None,
                                n_particles=200,
                                max_iterations=10,
                                neff_threshold=0.3,
                                seed=None,
                                n_workers=None,
                                ):
    """The plausible parameter sets of one glacier, with AdaPBS (experimental).

    The MB parameters, Glen A and the initial state of the dynamic spinup
    vary together, and AdaPBS weights each parameter set by how well its
    glacier matches the observations (see :py:mod:`oggm_adapbs.dynamic` for
    the purpose and the forward model): the geodetic MB, the area
    (``area_min_h``, ice thicker than 2 m) and the mean thickness (volume /
    total area) or the volume at ``state_year``. Nothing is written to the
    glacier directory (the particles' files are deleted).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory (with inversion and model flowlines, e.g. a
        preprocessed level 3 directory)
    settings_filesuffix : str
        the settings (and inversion flowlines) to start from
    state_year : int, optional
        the year of the area and volume targets. Default: RGI date + 1
    spinup_years : int
        length of the historical run before state_year
    initial_state_years : int
        length of the initial state period (constant climate) before the
        historical run. 0: none, the historical run starts from the inverted
        glacier (then initial_state_temp_offset cannot vary)
    ref_mb_period : str, default: the glacier's ``geodetic_mb_period``
        the geodetic MB period (full years)
    ref_mb, ref_mb_err : float, optional
        use this MB (kg m-2 yr-1, per RGI area) and its error instead of the
        geodetic MB file
    mb_file_path : str, optional
        custom geodetic MB file (see utils.get_geodetic_mb_dataframe)
    mb_err_factor : float
        factor on the MB error (default 0.2: Hugonnet's errors are too large
        for one glacier)
    ref_area : float, optional
        area target (km2). Default: the RGI area
    area_err, area_k : float
        the area error, see :py:func:`area_error_km2`
    volume_target : str
        'thickness' (default): the mean thickness, IceBoost volume / IceBoost
        area against model volume / model area (all the ice). Less dependent
        on the glacier size, and not as correlated with the area target as
        the volume. 'volume': the volume.
    ref_volume, ref_volume_err : float, optional
        use this volume and error (km3) instead of IceBoost. The thickness
        is then ref_volume / ref_area.
    volume_file_path : str, optional
        volume table, see :py:func:`iceboost_volume`
    volume_rel_err : float or None
        the error of the volume (or thickness) target, as a fraction of it
        (default 0.2, the same for all glaciers: a pragmatic choice). None:
        the relative IceBoost error of the glacier (or ref_volume_err /
        ref_volume) times ``volume_err_factor`` instead. The IceBoost error is
        not an accuracy measure: it is the spread of IceBoost under perturbed
        inputs, summed as if fully correlated (an upper bound, Maffezzoli et
        al. 2026), and smallest in steep terrain
    volume_err_factor : float
        factor on the relative IceBoost error, if volume_rel_err is None
    reject_sigma : float or None
        reject particles whose inverted volume (or volume / ref_area) is
        further than reject_sigma errors from the target, before the
        dynamical runs (they get a zero weight). None: no rejection
    calibrate_params : list of str
        the parameters which vary (any of ``DYNAMIC_PARAMS``). The others
        stay at their prior centre
    prior_centre, prior_bounds : dict, optional
        see :py:func:`dynamic_prior`
    prior_sd : float or dict
        prior standard deviation in normal space
    temp_bias_file_path : str, optional
        temperature bias file, for the centre of the temp_bias prior
    n_particles : int
        particles per AdaPBS iteration
    max_iterations : int
        maximum number of AdaPBS iterations
    neff_threshold : float
        AdaPBS adapts while Neff / n_particles is below this
    seed : int, optional
        random seed
    n_workers : int, optional
        run the particles in a pool of this many processes. Default (None or
        1): in this process (use that when OGGM's multiprocessing runs
        several glaciers in parallel)

    Returns
    -------
    an xarray Dataset with all the particles (dimension ``particle``):
    parameter values, Glen A, posterior ``weight``, predictions (``pred``),
    inverted volume, volume at the start of the historical run, volume, area
    and mass time series (dimension ``time``, starting
    ``initial_state_years`` before the historical run), status and run time;
    equally weighted resampled particles (``posterior_*``); per iteration
    diagnostics; prior and observations in the attributes.
    """
    if volume_target not in ('thickness', 'volume'):
        raise InvalidParamsError(f'Unknown volume_target: {volume_target}')
    if calving_mb(gdir) != 0:
        raise NotImplementedError('Calving glaciers are not implemented.')
    calibrate_params = list(calibrate_params)
    if (initial_state_years == 0 and
            'initial_state_temp_offset' in calibrate_params):
        raise InvalidParamsError('initial_state_temp_offset cannot vary '
                                 'without an initial state period.')
    if state_year is None:
        state_year = gdir.rgi_date + 1
    state_year = int(state_year)

    # Observations: geodetic MB
    if ref_mb_period is None:
        ref_mb_period = gdir.settings['geodetic_mb_period']
    y0, y1 = _period_years(ref_mb_period)
    if ref_mb is None:
        df = get_geodetic_mb_dataframe(file_path=mb_file_path,
                                       rgi_version=gdir.rgi_version
                                       ).loc[[gdir.rgi_id]]
        df = df.loc[df['period'] == ref_mb_period]
        if len(df) != 1:
            raise InvalidParamsError(f'{gdir.rgi_id}: no geodetic MB for '
                                     f'period {ref_mb_period}.')
        ref_mb = df['dmdtda'].iloc[0] * 1000  # m w.e. yr-1 -> kg m-2 yr-1
        ref_mb_err = df['err_dmdtda'].iloc[0] * 1000
    elif ref_mb_err is None:
        raise InvalidParamsError('ref_mb_err is required with ref_mb.')

    # Area
    if ref_area is None:
        ref_area = gdir.rgi_area_km2
    fls = gdir.read_store('model_flowlines', filesuffix=settings_filesuffix)
    step = terminus_step_km2(fls)
    sigma_area = area_error_km2(ref_area, step, area_err=area_err,
                                area_k=area_k)

    # Volume
    # Volume or mean thickness
    if ref_volume is None:
        ref_volume, ref_volume_err, vol_area = iceboost_volume(
            gdir, volume_file_path, return_area=True)
    elif ref_volume_err is None and volume_rel_err is None:
        raise InvalidParamsError('ref_volume_err or volume_rel_err is '
                                 'required with ref_volume.')
    else:
        vol_area = ref_area
    if volume_target == 'volume':
        ref_v, to_volume = ref_volume, 1.  # km3
    else:
        ref_v = ref_volume / vol_area * 1e3  # km3 / km2 -> m
        to_volume = ref_area * 1e-3  # m -> km3 (approximately, for rejection)
    if volume_rel_err is not None:
        sigma_v = volume_rel_err * ref_v
    else:
        sigma_v = ref_volume_err / ref_volume * ref_v * volume_err_factor
    obs_names = ['dmdtda', 'area', volume_target]
    i_pred = [PARTICLE_OBS.index(o) for o in obs_names]

    obs = np.array([ref_mb, ref_area, ref_v], dtype=float)
    obs_err = np.array([ref_mb_err * mb_err_factor, sigma_area, sigma_v],
                       dtype=float)
    volume_bounds = None
    if reject_sigma is not None:
        volume_bounds = ((obs[2] - reject_sigma * obs_err[2]) * to_volume,
                         (obs[2] + reject_sigma * obs_err[2]) * to_volume)

    # Is the climate long enough?
    ys = state_year - spinup_years
    ye = max(state_year, y1)
    mb_mod = MonthlyTIModel(gdir, settings_filesuffix=settings_filesuffix,
                            check_calib_params=False)
    if ys < mb_mod.ys or ye - 1 > mb_mod.ye:
        raise InvalidParamsError(f'The runs need climate data from {ys} to '
                                 f'{ye - 1}, available: {mb_mod.ys} to '
                                 f'{mb_mod.ye}.')

    # Prior
    prior = dynamic_prior(gdir, calibrate_params, prior_centre=prior_centre,
                          prior_bounds=prior_bounds,
                          temp_bias_file_path=temp_bias_file_path)
    x_c, x_min, x_max = prior['x_centre'], prior['x_min'], prior['x_max']
    if isinstance(prior_sd, dict):
        prior_sd = np.array([prior_sd[p] for p in calibrate_params],
                            dtype=float)

    # Forward model
    run_kwargs = dict(state_year=state_year, spinup_years=spinup_years,
                      initial_state_years=initial_state_years,
                      ref_period=(y0, y1),
                      glen_a_ref=prior['glen_a_ref'],
                      volume_bounds=volume_bounds,
                      settings_filesuffix=settings_filesuffix)
    pool = _make_pool(n_workers)
    extra_keys = ['volume_inversion_km3', 'volume_initial_state_km3',
                  'volume_km3', 'area_km2', 'mass_kg', 'status', 'run_time']

    def forward(theta, iteration):
        t0 = time.time()
        jobs = []
        for k, th in enumerate(theta):
            params = dict(prior['centre'])
            params.update(zip(calibrate_params, th))
            jobs.append((run_dynamic_particle, gdir, params,
                         f'_adapbs_dyn_it{iteration}_p{k}', run_kwargs))
        res = _map_jobs(pool, jobs, n_workers)
        status = np.array([r['status'] for r in res])
        n_rej = int((status == 'rejected').sum())
        n_dis = int((status == 'disappeared').sum())
        n_err = int((~np.isin(status, ['ok', 'rejected', 'disappeared'])
                     ).sum())
        log.info(f'{gdir.rgi_id} AdaPBS iteration {iteration + 1}: '
                 f'{len(jobs)} runs in {time.time() - t0:.0f} s, '
                 f'{n_rej} rejected, {n_dis} disappeared, {n_err} failed')
        pred = np.array([r['pred'][i_pred] for r in res])
        return pred, {k: np.array([r[k] for r in res]) for k in extra_keys}

    try:
        res = run_adapbs(forward, obs, obs_err, x_min, x_max,
                         prior_mean=glogit(x_c, x_min, x_max),
                         prior_sd=prior_sd, n_particles=n_particles,
                         max_iterations=max_iterations,
                         neff_threshold=neff_threshold,
                         rng=np.random.default_rng(seed))
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Output
    ex = res['extras']
    ds = xr.Dataset(coords={'particle': np.arange(len(res['weights'])),
                            'resampled': np.arange(n_particles),
                            'obs': obs_names,
                            'time': np.arange(ys - initial_state_years,
                                              ye + 1)})
    for k, p in enumerate(calibrate_params):
        ds[p] = ('particle', res['theta'][:, k])
        ds[f'posterior_{p}'] = ('resampled', res['posterior_theta'][:, k])
    for lp, p in LOG_MB_PARAMS.items():
        if lp in calibrate_params:
            # The values themselves, for convenience
            ds[p] = 10 ** ds[lp]
            ds[f'posterior_{p}'] = 10 ** ds[f'posterior_{lp}']
            ds[p].attrs['description'] = f'10 ** {lp}'
    lgf = (ds['log10_glen_a_factor'].values
           if 'log10_glen_a_factor' in calibrate_params
           else prior['centre']['log10_glen_a_factor'])
    ds['glen_a'] = ('particle', np.broadcast_to(
        prior['glen_a_ref'] * 10 ** np.asarray(lgf), ds.particle.shape))
    ds['weight'] = ('particle', res['weights'])
    ds['iteration'] = ('particle', res['iteration'])
    ds['pred'] = (('particle', 'obs'), res['pred'])
    ds['ref_obs'] = ('obs', obs)
    ds['ref_obs_err'] = ('obs', obs_err)
    ds['volume_inversion_km3'] = ('particle', ex['volume_inversion_km3'])
    ds['volume_initial_state_km3'] = ('particle',
                                      ex['volume_initial_state_km3'])
    ds['volume_km3'] = (('particle', 'time'), ex['volume_km3'])
    ds['area_km2'] = (('particle', 'time'), ex['area_km2'])
    ds['mass_kg'] = (('particle', 'time'), ex['mass_kg'])
    ds['mass_kg'].attrs['description'] = ('glacier mass: the annual specific '
                                          'MB (per RGI area) is its '
                                          'difference / gdir.rgi_area_m2')
    ds['status'] = ('particle', ex['status'].astype(str))
    ds['run_time'] = ('particle', ex['run_time'])
    ds['area_km2'].attrs['description'] = (f'area of the ice thicker than '
                                           f'{MIN_ICE_THICK_FOR_AREA} m')
    ds['pred'].attrs['units'] = ('dmdtda: kg m-2 yr-1, area: km2, '
                                 'volume: km3, thickness: m')
    diag = res['diagnostics'].set_index('iteration')
    for c in diag.columns:
        ds[f'adapbs_{c}'] = ('adapbs_iteration', diag[c].values)
    ds = ds.assign_coords(adapbs_iteration=diag.index.values)
    ds.attrs.update(rgi_id=gdir.rgi_id,
                    calibrate_params=calibrate_params,
                    prior_centre=x_c, prior_min=x_min, prior_max=x_max,
                    prior_sd=np.broadcast_to(prior_sd, x_c.shape).astype(float),
                    fixed_params=str({p: v for p, v
                                      in prior['centre'].items()
                                      if p not in calibrate_params}),
                    glen_a_ref=prior['glen_a_ref'],
                    state_year=state_year, spinup_years=spinup_years,
                    initial_state_years=initial_state_years,
                    ref_mb_period=ref_mb_period,
                    settings_filesuffix=settings_filesuffix,
                    area_step_km2=step, mb_err_factor=mb_err_factor,
                    area_err=area_err, area_k=area_k,
                    volume_target=volume_target,
                    volume_rel_err=np.nan if volume_rel_err is None
                    else volume_rel_err,
                    volume_err_factor=volume_err_factor,
                    reject_sigma=np.nan if reject_sigma is None
                    else reject_sigma,
                    n_particles=n_particles, max_iterations=max_iterations,
                    neff_threshold=neff_threshold,
                    log_evidence=res['log_evidence'])
    return ds


def save_netcdf(ds, path, float64_vars=('weight',)):
    """Save a Dataset of this module, compressed as OGGM's compiled outputs.

    The floating point variables are saved as float32 with zlib compression
    (level 5), except ``float64_vars`` (the weights: some are tiny, and their
    sums matter). The hydro reruns are several times smaller this way.

    Parameters
    ----------
    ds : xarray.Dataset
        e.g. the output of :py:func:`dynamic_equifinality_adapbs` or
        :py:func:`dynamic_hydro_rerun`
    path : str
        the netCDF file to write
    float64_vars : list of str
        variables kept in float64 (compressed as well)
    """
    enc = {}
    for v in ds.data_vars:
        if not np.issubdtype(ds[v].dtype, np.floating):
            continue
        enc[v] = {'zlib': True, 'complevel': 5}
        if v not in float64_vars:
            enc[v]['dtype'] = 'float32'
    ds.to_netcdf(path, encoding=enc)


def select_particles(ds, weight_fraction=0.999, n_resample=None, seed=None):
    """Choose the particles to rerun (e.g. with hydrological output).

    AdaPBS keeps all its runs, most of them with a negligible weight. Two
    ways to choose the plausible ones:

    - by default, the particles which hold ``weight_fraction`` of the total
      weight (the largest weights first). They keep their weights
      (renormalised): no information is lost, but the weights are unequal;
    - with ``n_resample``: draw this many particles at random with the
      posterior weights. A particle drawn several times is rerun once; its
      weight is the number of draws / n_resample. Equally weighted samples
      are convenient for GLUE-style analyses, and this is cheaper, but adds
      sampling noise.

    Parameters
    ----------
    ds : xarray.Dataset
        output of :py:func:`dynamic_equifinality_adapbs`
    weight_fraction : float
        fraction of the total weight to keep (default selection)
    n_resample : int, optional
        number of draws (the second selection)
    seed : int, optional
        random seed of the draws

    Returns
    -------
    (indices, weights): the particle indices (into ``ds.particle``) and
    their weights (sum 1)
    """
    w = np.nan_to_num(ds['weight'].values)
    if n_resample is not None:
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(w), n_resample, p=w / w.sum())
        idx, counts = np.unique(draws, return_counts=True)
        return idx, counts / n_resample
    order = np.argsort(-w)
    n = int(np.searchsorted(np.cumsum(w[order]) / w.sum(), weight_fraction))
    idx = np.sort(order[:min(n + 1, len(w))])
    return idx, w[idx] / w[idx].sum()


def dynamic_hydro_rerun(gdir, ds, particles=None, weights=None, *,
                        ref_area_yr=2000, n_workers=None):
    """Rerun particles of :py:func:`dynamic_equifinality_adapbs` with
    OGGM's hydrological output (monthly).

    The AdaPBS runs keep no hydrological output: computing it for every run
    would cost about 2.5 times more, mostly for runs with a negligible weight.
    Instead, we rerun the plausible particles (see :py:func:`select_particles`)
    with :py:func:`run_dynamic_particle_hydro`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory of the AdaPBS run (same settings)
    ds : xarray.Dataset
        output of :py:func:`dynamic_equifinality_adapbs`
    particles : array of int, optional
        indices of the particles to rerun. Default: see
        :py:func:`select_particles`
    weights : array of float, optional
        their weights (default: their AdaPBS weights, renormalised)
    ref_area_yr : int
        the year of the glacier area which defines the catchment of the
        runoff (default 2000)
    n_workers : int, optional
        number of processes (as in :py:func:`dynamic_equifinality_adapbs`)

    Returns
    -------
    an xarray Dataset with dimensions ``particle`` (the indices of ``ds``),
    ``time`` (calendar years of the historical run) and ``month`` (calendar
    months): the parameters, ``weight``, annual ``volume_km3`` and
    ``area_km2``, the ``HYDRO_VARIABLES`` (annual and ``_monthly``), the
    ``runoff`` (annual and monthly, kg) and the ``status`` of the runs.
    """
    if particles is None:
        particles, weights = select_particles(ds)
    particles = np.asarray(particles)
    if weights is None:
        weights = np.nan_to_num(ds['weight'].values[particles])
        weights = weights / weights.sum()
    if ds.attrs['rgi_id'] != gdir.rgi_id:
        raise InvalidParamsError(f'ds is for {ds.attrs["rgi_id"]}, not '
                                 f'{gdir.rgi_id}.')

    # Everything the particles need, from the AdaPBS run
    a = ds.attrs
    calibrate_params = list(np.atleast_1d(a['calibrate_params']))
    fixed = ast.literal_eval(a['fixed_params'])
    y0, y1 = _period_years(a['ref_mb_period'])
    run_kwargs = dict(state_year=int(a['state_year']),
                      spinup_years=int(a['spinup_years']),
                      initial_state_years=int(a['initial_state_years']),
                      ref_period=(y0, y1), glen_a_ref=float(a['glen_a_ref']),
                      settings_filesuffix=a.get('settings_filesuffix', ''),
                      ref_area_yr=ref_area_yr)
    jobs = []
    for i in particles:
        params = {**fixed, **{p: float(ds[p].values[i])
                              for p in calibrate_params}}
        jobs.append((run_dynamic_particle_hydro, gdir, params,
                     f'_adapbs_hydro_p{i}', run_kwargs))

    t0 = time.time()
    pool = _make_pool(n_workers)
    try:
        res = _map_jobs(pool, jobs, n_workers)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    status = np.array([r['status'] for r in res])
    log.info(f'{gdir.rgi_id} hydro rerun: {len(jobs)} runs in '
             f'{time.time() - t0:.0f} s, {int((status != "ok").sum())} '
             f'failed')

    ys = run_kwargs['state_year'] - run_kwargs['spinup_years']
    ye = max(run_kwargs['state_year'], y1)
    out = xr.Dataset(coords={'particle': particles,
                             'time': np.arange(ys, ye + 1),
                             'month': np.arange(1, 13)})
    for p in ALL_DYNAMIC_PARAMS:
        if p in ds:
            out[p] = ('particle', ds[p].values[particles])
    out['weight'] = ('particle', np.asarray(weights, dtype=float))
    out['volume_km3'] = (('particle', 'time'),
                         np.array([r['volume_km3'] for r in res]))
    out['area_km2'] = (('particle', 'time'),
                       np.array([r['area_km2'] for r in res]))
    for v in res[0]['hydro']:
        dims = (('particle', 'time', 'month') if v.endswith('_monthly')
                else ('particle', 'time'))
        out[v] = (dims, np.array([r['hydro'][v] for r in res]))
    out['runoff'] = sum(out[v] for v in RUNOFF_VARIABLES)
    out['runoff_monthly'] = sum(out[f'{v}_monthly'] for v in RUNOFF_VARIABLES)
    for v in ['runoff', 'runoff_monthly'] + [
            v for v in out.data_vars if v.split('_monthly')[0] in
            HYDRO_VARIABLES and 'area' not in v]:
        out[v].attrs['units'] = ('kg month-1' if v.endswith('_monthly')
                                 else 'kg yr-1')
    out['runoff'].attrs['description'] = (
        'melt and liquid precipitation on and off the glacier, over the '
        f'glacier area of {ref_area_yr} (larger in years when the glacier '
        'is larger; the last year is NaN)')
    out['status'] = ('particle', status.astype(str))
    out.attrs.update(rgi_id=gdir.rgi_id, ref_area_yr=ref_area_yr,
                     calibrate_params=calibrate_params,
                     **{k: a[k] for k in ['state_year', 'spinup_years',
                                          'initial_state_years',
                                          'ref_mb_period']})
    return out


def runoff_metrics(dh, period=(2000, 2020)):
    """Simple runoff metrics of the hydro reruns, per particle.

    Means over the calendar years ``period[0]`` to ``period[1] - 1``
    (default 2000-2019, the period of the geodetic MB):

    - ``runoff``: mean annual runoff (Mt yr-1, i.e. million m3 of water per
      year);
    - ``runoff_monthly``: mean runoff of each calendar month (Mt month-1),
      the seasonal cycle;
    - ``summer_fraction``: the fraction of the annual runoff in July, August
      and September;
    - ``peak_month``: the month with the largest mean runoff;
    - ``glacier_melt_share``: melt on the glacier (snow and ice) / runoff.

    Parameters
    ----------
    dh : xarray.Dataset
        output of :py:func:`dynamic_hydro_rerun`
    period : (int, int)
        start and end year (the end year is excluded)

    Returns
    -------
    an xarray Dataset (dimension ``particle``, and ``month`` for the
    seasonal cycle), with the parameters and weights of ``dh``
    """
    sel = dh.sel(time=slice(period[0], period[1] - 1))
    if sel.sizes['time'] != period[1] - period[0]:
        raise InvalidParamsError(f'The reruns do not cover {period}.')
    out = dh[[p for p in ALL_DYNAMIC_PARAMS if p in dh] + ['weight']].copy()
    out['runoff'] = sel['runoff'].mean('time') * 1e-9
    out['runoff'].attrs['units'] = 'Mt yr-1'
    clim = sel['runoff_monthly'].mean('time') * 1e-9
    out['runoff_monthly'] = clim
    out['runoff_monthly'].attrs['units'] = 'Mt month-1'
    out['summer_fraction'] = (clim.sel(month=[7, 8, 9]).sum('month') /
                              clim.sum('month'))
    out['peak_month'] = clim.idxmax('month')
    out['glacier_melt_share'] = (sel['melt_on_glacier'].sum('time') /
                                 sel['runoff'].sum('time'))
    out.attrs.update(dh.attrs)
    out.attrs['period'] = f'{period[0]}-{period[1] - 1}'
    return out
