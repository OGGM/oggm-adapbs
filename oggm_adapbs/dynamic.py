"""OGGM dynamic calibration with AdaPBS (experimental).

In OGGM's standard workflow, the MB parameters are calibrated on a fixed
geometry, the ice thickness is inverted with a Glen A calibrated for a whole
region, and a dynamic spinup finds a glacier state which matches the RGI
area at the RGI date. Here, one AdaPBS calibration of a single glacier
replaces all three. Each particle (parameter set):

1. inverts the ice thickness at the RGI date, with its MB parameters (for the
   shape of the MB profile) and its own Glen A (no sliding)
2. lets the inverted glacier grow (or shrink) for ``growth_years`` in a
   constant climate (the mean climate of the spinup period) with its
   temperature bias plus a growth temperature offset, as in OGGM's dynamic
   spinup (where the growth temperature bias is found by minimisation
   instead)
3. runs with the historical climate from ``state_year - spinup_years`` to the
   end of the geodetic MB period

and is compared to three targets: the geodetic MB (Hugonnet et al., 2021),
the RGI area and the IceBoost v2 mean thickness (volume / area) or volume,
both at ``state_year`` (the RGI date + 1, i.e. the end of the RGI year).

Calibrated parameters (any subset of ``DYNAMIC_PARAMS``): melt_f, prcp_fac,
temp_bias, log10_glen_a_factor (A = ``cfg.PARAMS['inversion_glen_a']`` *
10 ** log10_glen_a_factor, used for the inversion and the dynamics) and
growth_temp_offset (K, added to temp_bias during the growth period only).

Please cite the AdaPBS and MuSA papers (see :py:mod:`oggm_adapbs.adapbs`).
"""
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
                                decide_evolution_model)

from oggm_adapbs.adapbs import run_adapbs, glogit
from oggm_adapbs.massbalance import MB_PARAMS, mb_prior

log = logging.getLogger(__name__)

DYNAMIC_PARAMS = MB_PARAMS + ('log10_glen_a_factor', 'growth_temp_offset')

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
                  growth_temp_range=3.):
    """The default prior of the dynamic calibration.

    The MB parameters as in :py:func:`oggm_adapbs.massbalance.mb_prior`,
    and:

    - log10_glen_a_factor: centre 0 (OGGM's default
      ``cfg.PARAMS['inversion_glen_a']``, not the regionally calibrated value
      of the glacier settings), bounds +- glen_a_range (default: A / 10 to
      A * 10)
    - growth_temp_offset: centre 0, bounds +- growth_temp_range (K)

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    calibrate_params : list of str
        the parameters to calibrate (any of ``DYNAMIC_PARAMS``)
    prior_centre : dict, optional
        override the centre of some parameters
    prior_bounds : dict, optional
        override the bounds of some parameters
    temp_bias_file_path : str, optional
        temperature bias file, for the centre of the temp_bias prior
    prcp_fac_range, temp_bias_range : float
        see :py:func:`oggm_adapbs.massbalance.mb_prior`
    glen_a_range : float
        range of the log10_glen_a_factor bounds around the centre
    growth_temp_range : float
        range of the growth_temp_offset bounds around the centre (K)

    Returns
    -------
    a dict as :py:func:`oggm_adapbs.massbalance.mb_prior`, plus
    ``glen_a_ref`` (the A of log10_glen_a_factor = 0).
    """
    calibrate_params = list(calibrate_params)
    for p in calibrate_params:
        if p not in DYNAMIC_PARAMS:
            raise InvalidParamsError(f'Unknown parameter: {p}. Use one of '
                                     f'{DYNAMIC_PARAMS}.')
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

    centre['log10_glen_a_factor'] = 0.
    centre['growth_temp_offset'] = 0.
    for k, v in prior_centre.items():
        if k not in MB_PARAMS:
            centre[k] = float(v)
    bounds['log10_glen_a_factor'] = (centre['log10_glen_a_factor'] -
                                     glen_a_range,
                                     centre['log10_glen_a_factor'] +
                                     glen_a_range)
    bounds['growth_temp_offset'] = (centre['growth_temp_offset'] -
                                    growth_temp_range,
                                    centre['growth_temp_offset'] +
                                    growth_temp_range)
    for k, v in prior_bounds.items():
        if k not in MB_PARAMS:
            bounds[k] = v

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


def run_dynamic_particle(gdir, params, filesuffix, *,
                         state_year,
                         spinup_years,
                         growth_years,
                         ref_period,
                         glen_a_ref,
                         volume_bounds=None,
                         settings_filesuffix='',
                         keep_files=False):
    """One forward run of the dynamic calibration (one particle).

    Inversion, growth, historical run (see :py:mod:`oggm_adapbs.dynamic`).
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
    growth_years : int
        length of the growth period (0: no growth period)
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
    see ``PARTICLE_OBS``), ``volume_inversion_km3``, ``volume_growth_km3``
    (at the end of the growth period), annual ``volume_km3``, ``area_km2``
    (area_min_h) and ``mass_kg`` (ice mass, for the specific MB) from the start of the growth period (``growth_years`` before
    the historical run: these years have no calendar meaning, the climate is
    constant) to the end of the historical run, ``status`` ('ok',
    'rejected' or the error) and ``run_time`` (s).
    """
    if not filesuffix:
        raise InvalidParamsError('A particle needs a filesuffix.')
    t0 = time.time()
    y0, y1 = ref_period
    ys = state_year - spinup_years
    ye = max(state_year, y1)
    nyears = ye - ys + 1 + growth_years
    out = dict(pred=np.full(len(PARTICLE_OBS), np.nan),
               volume_inversion_km3=np.nan, volume_growth_km3=np.nan,
               volume_km3=np.full(nyears, np.nan),
               area_km2=np.full(nyears, np.nan),
               mass_kg=np.full(nyears, np.nan), status='ok')
    s = filesuffix
    glen_a = glen_a_ref * 10 ** params['log10_glen_a_factor']
    task_kwargs = dict(settings_filesuffix=s, add_to_log_file=False)
    orig_suffix = gdir.settings_filesuffix
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            gdir.create_new_settings(
                s, data={'melt_f': float(params['melt_f']),
                         'prcp_fac': float(params['prcp_fac']),
                         'temp_bias': float(params['temp_bias']),
                         'store_diagnostic_variables': DIAGNOSTIC_VARIABLES,
                         'min_ice_thick_for_area': MIN_ICE_THICK_FOR_AREA},
                parent_filesuffix=settings_filesuffix,
                ignore_existing=True, overwrite=True)

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
                out['status'] = 'rejected'
                return out

            gdir.settings_filesuffix = s
            evolution_model = decide_evolution_model(gdir)
            model_kwargs = dict(glen_a=glen_a, fs=0., gdir=gdir,
                                settings_filesuffix=s)

            # 2. Growth period, constant climate of the spinup period (as in
            # OGGM's run_dynamic_spinup)
            if growth_years > 0:
                yc = (ys + state_year) / 2
                mb_grow = MultipleFlowlineMassBalance(
                    gdir, settings_filesuffix=s, fls=fls,
                    mb_model_class=partial(ConstantMassBalance,
                                           mb_model_class=MonthlyTIModel),
                    y0=yc, halfsize=state_year - yc,
                    check_calib_params=False)
                mb_grow.temp_bias = (params['temp_bias'] +
                                     params['growth_temp_offset'])
                # (the constant climate does not depend on the year: the
                # growth years are just before the historical run)
                model = evolution_model(copy.deepcopy(fls), mb_model=mb_grow,
                                        y0=ys - growth_years, **model_kwargs)
                ds_grow = model.run_until_and_store(ys, diag_path=None,
                                                    geom_path=False,
                                                    fl_diag_path=False)
                if isinstance(ds_grow, tuple):
                    ds_grow = ds_grow[0]
                fls = model.fls
            out['volume_growth_km3'] = sum(fl.volume_m3 for fl in fls) * 1e-9

            # 3. Historical run
            mb_hist = MultipleFlowlineMassBalance(
                gdir, settings_filesuffix=s, fls=fls,
                mb_model_class=MonthlyTIModel, check_calib_params=False)
            model = evolution_model(fls, mb_model=mb_hist, y0=ys,
                                    **model_kwargs)
            ds = model.run_until_and_store(ye, diag_path=None, geom_path=False,
                                           fl_diag_path=False)
            if isinstance(ds, tuple):
                ds = ds[0]

        ds = ds.assign_coords(time=np.round(ds.time.values).astype(int))
        mass = ds['mass_kg']
        dm = float(mass.sel(time=y1) - mass.sel(time=y0))
        vol = float(ds['volume_m3'].sel(time=state_year))
        out['pred'] = np.array([
            dm / gdir.rgi_area_m2 / (y1 - y0),
            float(ds['area_min_h_m2'].sel(time=state_year)) * 1e-6,
            vol * 1e-9,
            vol / float(ds['area_m2'].sel(time=state_year))])
        if growth_years > 0:
            # The last year of the growth period is the first of the run
            ds = xr.concat([ds_grow.isel(time=slice(0, -1)), ds], dim='time')
        out['volume_km3'] = ds['volume_m3'].values * 1e-9
        out['area_km2'] = ds['area_min_h_m2'].values * 1e-6
        out['mass_kg'] = ds['mass_kg'].values
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


def _init_worker(cfg_contents):
    cfg.unpack_config(cfg_contents)
    logging.getLogger('oggm').setLevel(logging.ERROR)
    warnings.simplefilter('ignore', category=RuntimeWarning)
    # Preprocessed directories have pickle files: one warning per worker
    warnings.filterwarnings('ignore', message='Store data not found')


def _run_job(args):
    gdir, params, filesuffix, kwargs = args
    return run_dynamic_particle(gdir, params, filesuffix, **kwargs)


@entity_task(log)
def dynamic_calibration_adapbs(gdir, *,
                               settings_filesuffix='',
                               state_year=None,
                               spinup_years=30,
                               growth_years=20,
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
                               volume_err_factor=1.,
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
    """Dynamic calibration of one glacier with AdaPBS (experimental).

    Replaces the static MB calibration, the regional inversion and the
    dynamic spinup (see :py:mod:`oggm_adapbs.dynamic` for the forward model).
    Targets: the geodetic MB, the area (``area_min_h``, ice thicker than 2 m)
    and the mean thickness (volume / total area) or the volume at
    ``state_year``. Nothing is written to the glacier
    directory (the particles' files are deleted).

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
    growth_years : int
        length of the growth period (constant climate) before the historical
        run. 0: no growth period (then growth_temp_offset cannot be
        calibrated)
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
        error of the volume (or thickness) target, as a fraction of it
        (default: 0.2). None: the IceBoost error (or ref_volume_err) times
        volume_err_factor (IceBoost errors are large, median 49% for RGI6)
    volume_err_factor : float
        factor on the volume error, if volume_rel_err is None
    reject_sigma : float or None
        reject particles whose inverted volume (or volume / ref_area) is
        further than reject_sigma errors from the target, before the
        dynamical runs (they get a zero weight). None: no rejection
    calibrate_params : list of str
        the parameters to calibrate (any of ``DYNAMIC_PARAMS``). The others
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
    inverted volume, volume after the growth period, volume and area time
    series (dimension ``time``, starting ``growth_years`` before the
    historical run with the growth period), status and run time; equally weighted resampled particles
    (``posterior_*``); per iteration diagnostics; prior and observations in
    the attributes.
    """
    if volume_target not in ('thickness', 'volume'):
        raise InvalidParamsError(f'Unknown volume_target: {volume_target}')
    if calving_mb(gdir) != 0:
        raise NotImplementedError('Calving glaciers are not implemented.')
    calibrate_params = list(calibrate_params)
    if growth_years == 0 and 'growth_temp_offset' in calibrate_params:
        raise InvalidParamsError('growth_temp_offset cannot be calibrated '
                                 'without a growth period.')
    if state_year is None:
        state_year = gdir.rgi_date + 1
    state_year = int(state_year)

    # Observations: geodetic MB
    if ref_mb_period is None:
        ref_mb_period = gdir.settings['geodetic_mb_period']
    y0, y1 = (int(p.split('-')[0]) for p in ref_mb_period.split('_'))
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
                      growth_years=growth_years, ref_period=(y0, y1),
                      glen_a_ref=prior['glen_a_ref'],
                      volume_bounds=volume_bounds,
                      settings_filesuffix=settings_filesuffix)
    pool = None
    if n_workers is not None and n_workers > 1:
        pool = mp.get_context('spawn').Pool(n_workers,
                                            initializer=_init_worker,
                                            initargs=(cfg.pack_config(),))

    extra_keys = ['volume_inversion_km3', 'volume_growth_km3', 'volume_km3',
                  'area_km2', 'mass_kg', 'status', 'run_time']

    def forward(theta, iteration):
        t0 = time.time()
        jobs = []
        for k, th in enumerate(theta):
            params = dict(prior['centre'])
            params.update(zip(calibrate_params, th))
            jobs.append((gdir, params, f'_adapbs_dyn_it{iteration}_p{k}',
                         run_kwargs))
        if pool is None:
            res = [_run_job(j) for j in jobs]
        else:
            chunksize = max(1, len(jobs) // (4 * n_workers))
            res = pool.map(_run_job, jobs, chunksize=chunksize)
        status = np.array([r['status'] for r in res])
        n_rej = int((status == 'rejected').sum())
        n_err = int(((status != 'ok') & (status != 'rejected')).sum())
        log.info(f'{gdir.rgi_id} AdaPBS iteration {iteration + 1}: '
                 f'{len(jobs)} runs in {time.time() - t0:.0f} s, '
                 f'{n_rej} rejected, {n_err} failed')
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
                            'time': np.arange(ys - growth_years, ye + 1)})
    for k, p in enumerate(calibrate_params):
        ds[p] = ('particle', res['theta'][:, k])
        ds[f'posterior_{p}'] = ('resampled', res['posterior_theta'][:, k])
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
    ds['volume_growth_km3'] = ('particle', ex['volume_growth_km3'])
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
                    fixed_params=str({p: prior['centre'][p]
                                      for p in DYNAMIC_PARAMS
                                      if p not in calibrate_params}),
                    glen_a_ref=prior['glen_a_ref'],
                    state_year=state_year, spinup_years=spinup_years,
                    growth_years=growth_years, ref_mb_period=ref_mb_period,
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
