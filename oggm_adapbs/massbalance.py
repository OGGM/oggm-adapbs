"""OGGM mass balance calibration with the AdaPBS, static geometry.

Tools to build the prior and the forward model of the mass balance
parameters (melt_f, prcp_fac, temp_bias) of OGGM's ``MonthlyTIModel`` on
fixed (inversion) flowlines, and an entity task calibrating them against
geodetic MB data.
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from oggm import cfg, entity_task
from oggm.utils import (get_geodetic_mb_dataframe, get_temp_bias_dataframe,
                        date_to_floatyear)
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.core.massbalance import (MonthlyTIModel,
                                   MultipleFlowlineMassBalance,
                                   decide_winter_precip_factor, calving_mb,
                                   _floatyears_from_ref_mb_period)

from oggm_adapbs.adapbs import run_adapbs, glogit

log = logging.getLogger(__name__)

MB_PARAMS = ('melt_f', 'prcp_fac', 'temp_bias')


def informed_temp_bias(gdir, temp_bias_file_path):
    """The temperature bias prior of this glacier, read from the bias file.

    Same as in OGGM's ``mb_calibration_from_geodetic_mb`` with
    ``informed_threestep=True`` (code copied from OGGM, BSD-3-Clause).
    """

    bias_df = get_temp_bias_dataframe(temp_bias_file_path)
    climinfo = gdir.get_climate_info()

    # Is this file made for this run? RGI versions are not
    # interchangeable. Recent files say what they were made for, older ones
    # don't: for those the file name is all we have to go by.
    file_version = None
    if 'rgi_version' in bias_df:
        file_version = bias_df['rgi_version'].iloc[0]
    if file_version is not None and not pd.isnull(file_version):
        if str(file_version) != gdir.rgi_version:
            raise InvalidWorkflowError(
                f'The temperature bias file was made for RGI version '
                f'{file_version}, but this run uses {gdir.rgi_version}: '
                f'set `temp_bias_file_path` to the file matching your '
                f'setup. File: {temp_bias_file_path}')
        file_source = bias_df.get('baseline_climate_source')
        file_source = None if file_source is None else file_source.iloc[0]
        source = climinfo['baseline_climate_source']
        if file_source is not None and not pd.isnull(file_source):
            if str(file_source) != str(source):
                log.warning(f'The temperature bias file was made with '
                            f'the {file_source} climate data, but this '
                            f'run uses {source}: the prior is unlikely '
                            f'to be a good one. '
                            f'File: {temp_bias_file_path}')
    else:
        name = os.path.basename(str(temp_bias_file_path)).lower()
        guess = {'rgi70g': '70G', 'rgi70c': '70C', 'rgi6': '60'}
        guess = next((v for k, v in guess.items() if k in name), None)
        if guess is not None and guess != gdir.rgi_version:
            log.warning(f'The name of the temperature bias file suggests '
                        f'that it was made for RGI{guess}, but this run '
                        f'uses RGI version {gdir.rgi_version}. Is it the '
                        f'right file? File: {temp_bias_file_path}')

    # The glacier's climate grid point is either in the file or it is not:
    # accept one grid cell of distance at most.
    ref_lon = climinfo['baseline_climate_ref_pix_lon']
    ref_lat = climinfo['baseline_climate_ref_pix_lat']
    lon_val = bias_df.lon_val.values
    lat_val = bias_df.lat_val.values
    nx = np.ptp(bias_df.lon_id.values)
    ny = np.ptp(bias_df.lat_id.values)
    dlon = np.ptp(lon_val) / nx if nx else None
    dlat = np.ptp(lat_val) / ny if ny else None
    dlon = dlat if dlon is None else dlon
    dlat = dlon if dlat is None else dlat

    d_lon = np.abs(lon_val - ref_lon)
    d_lon = np.minimum(d_lon, 360 - d_lon)  # wrap-around at the dateline
    dis = np.maximum(d_lon / dlon, np.abs(lat_val - ref_lat) / dlat)
    imin = np.argmin(dis)
    if dis[imin] > 1:
        raise InvalidWorkflowError(
            f'The climate grid point of this glacier ({ref_lon:.2f}°E '
            f'{ref_lat:.2f}°N) is not in the temperature bias file: the '
            f'nearest one is {dis[imin]:.1f} grid cells away. Is it the '
            f'right file for this run? File: {temp_bias_file_path}')
    temp_bias = bias_df.iloc[imin]['median_temp_bias_w_err_grouped']
    assert np.isfinite(temp_bias), 'Temp bias not finite?'
    return float(temp_bias)


def mb_prior(gdir, calibrate_params=MB_PARAMS, *,
             prior_centre=None,
             prior_bounds=None,
             temp_bias_file_path=None,
             prcp_fac_range=2.,
             temp_bias_range=3.):
    """The default prior of the MB parameters of a glacier.

    The prior of each parameter is logit-normal (Gaussian in a transformed
    space, bounded). This defines its centre and bounds:

    - melt_f: centre ``cfg.PARAMS['melt_f']``, bounds ``melt_f_min/max``
      from the glacier settings
    - prcp_fac: centre ``cfg.PARAMS['prcp_fac']`` if set, otherwise from the
      winter precipitation (``decide_winter_precip_factor``), bounds
      [centre / prcp_fac_range, centre * prcp_fac_range] (clipped to
      ``prcp_fac_min/max``)
    - temp_bias: centre from the temperature bias file if
      ``temp_bias_file_path`` is given (otherwise 0), bounds the centre
      +- temp_bias_range

    The centres are not read from the glacier settings: after a calibration
    (or on preprocessed directories) these hold calibrated values.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    calibrate_params : list of str
        the parameters to calibrate (any of melt_f, prcp_fac, temp_bias)
    prior_centre : dict, optional
        override the centre of some parameters, e.g. {'melt_f': 6}
    prior_bounds : dict, optional
        override the bounds of some parameters, e.g. {'temp_bias': (-2, 2)}
    temp_bias_file_path : str, optional
        temperature bias file, for the centre of the temp_bias prior
    prcp_fac_range : float
        multiplicative range of the prcp_fac bounds around the centre
    temp_bias_range : float
        range of the temp_bias bounds around the centre (K)

    Returns
    -------
    a dict with ``params`` (the calibrated parameters), ``centre`` (dict, the
    centre of all three parameters, the values of the parameters which are
    not calibrated), and ``x_centre``, ``x_min``, ``x_max`` (arrays, one
    value per calibrated parameter).
    """
    calibrate_params = list(calibrate_params)
    for p in calibrate_params:
        if p not in MB_PARAMS:
            raise InvalidParamsError(f'Unknown parameter: {p}. Use one of '
                                     f'{MB_PARAMS}.')

    centre = {'melt_f': cfg.PARAMS['melt_f'],
              'prcp_fac': cfg.PARAMS['prcp_fac'],
              'temp_bias': 0.}
    if centre['prcp_fac'] is None:
        centre['prcp_fac'] = decide_winter_precip_factor(gdir)
    if temp_bias_file_path is not None:
        centre['temp_bias'] = informed_temp_bias(gdir, temp_bias_file_path)
    centre.update(prior_centre or {})
    centre = {k: float(v) for k, v in centre.items()}

    bounds = {
        'melt_f': (gdir.settings['melt_f_min'], gdir.settings['melt_f_max']),
        'prcp_fac': (max(centre['prcp_fac'] / prcp_fac_range,
                         gdir.settings['prcp_fac_min']),
                     min(centre['prcp_fac'] * prcp_fac_range,
                         gdir.settings['prcp_fac_max'])),
        'temp_bias': (centre['temp_bias'] - temp_bias_range,
                      centre['temp_bias'] + temp_bias_range),
    }
    bounds.update(prior_bounds or {})

    x_min = np.array([bounds[p][0] for p in calibrate_params], dtype=float)
    x_max = np.array([bounds[p][1] for p in calibrate_params], dtype=float)
    x_c = np.array([centre[p] for p in calibrate_params], dtype=float)
    if np.any(x_c <= x_min) or np.any(x_c >= x_max):
        raise InvalidParamsError(f'Prior centre {x_c} not within the bounds '
                                 f'{x_min}, {x_max}.')
    return dict(params=calibrate_params, centre=centre,
                x_centre=x_c, x_min=x_min, x_max=x_max)


def static_mb_model(gdir, params, *,
                    settings_filesuffix='',
                    mb_model_class=MonthlyTIModel,
                    **kwargs):
    """The MB model and the inversion flowlines of a glacier (fixed geometry).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    params : dict
        melt_f, prcp_fac and temp_bias to start with (e.g. the prior centre).
        Change them later with ``setattr(mb_mod, name, value)``.
    settings_filesuffix : str
        the settings to use
    mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
        the MassBalanceModel to use (with melt_f, temp_bias, prcp_fac)
    kwargs : dict
        kwargs to pass to the mb_model_class instance

    Returns
    -------
    (mb_mod, fls)
    """
    fls = gdir.read_store('inversion_flowlines')
    mb_kwargs = dict(melt_f=params['melt_f'], temp_bias=params['temp_bias'],
                     prcp_fac=params['prcp_fac'], check_calib_params=False,
                     settings_filesuffix=settings_filesuffix, **kwargs)
    if len(fls) > 1:
        mb_mod = MultipleFlowlineMassBalance(gdir=gdir, fls=fls,
                                             mb_model_class=mb_model_class,
                                             **mb_kwargs)
    else:
        mb_mod = mb_model_class(gdir=gdir, **mb_kwargs)
    return mb_mod, fls


def seasonal_specific_mb(mb_mod, fls, years, *,
                         start_month=10,
                         summer_start_month=5):
    """Winter, summer and annual specific MB of hydrological years.

    OGGM's ``get_specific_mb`` works in calendar years. Glaciological
    (WGMS) mass balances are given for hydrological years (October to
    September in the northern hemisphere), with a winter (October to April)
    and summer (May to September) balance. This sums the monthly specific MB
    accordingly.

    Parameters
    ----------
    mb_mod : MassBalanceModel
        the MB model
    fls : list of flowlines
        the flowlines
    years : array of int
        the hydrological years. Year Y starts in ``start_month`` of the
        calendar year Y - 1 (if start_month > 1).
    start_month : int
        first month of the hydrological year (default 10, October)
    summer_start_month : int
        first month of the summer season (default 5, May). For the southern
        hemisphere, use e.g. ``start_month=4, summer_start_month=10``.

    Returns
    -------
    a DataFrame indexed by year with columns winter, summer, annual
    (kg m-2 = mm w.e.)
    """
    years = np.atleast_1d(years).astype(int)
    months = (np.arange(12) + start_month - 1) % 12 + 1
    offset = np.zeros(12, dtype=int)
    if start_month > 1:
        offset[months >= start_month] = -1
    cal_years = (years[:, None] + offset[None, :]).ravel()
    fyears = date_to_floatyear(cal_years, np.tile(months, len(years)))
    mb = mb_mod.get_specific_mb(fls=fls, year=fyears,
                                time_resolution='monthly')
    mb = np.asarray(mb).reshape(len(years), 12)
    n_winter = (summer_start_month - start_month) % 12
    winter = mb[:, :n_winter].sum(axis=1)
    summer = mb[:, n_winter:].sum(axis=1)
    return pd.DataFrame({'winter': winter, 'summer': summer,
                         'annual': winter + summer},
                        index=pd.Index(years, name='year'))


@entity_task(log)
def mb_calibration_from_geodetic_mb_adapbs(gdir, *,
                                           settings_filesuffix='',
                                           ref_mb_period=None,
                                           ref_mb=None,
                                           ref_mb_err=None,
                                           file_path=None,
                                           temp_bias_file_path=None,
                                           obs_err_factor=1.,
                                           calibrate_params=MB_PARAMS,
                                           prior_centre=None,
                                           prior_bounds=None,
                                           prior_sd=1.,
                                           centre_prior=True,
                                           n_particles=200,
                                           max_iterations=6,
                                           neff_threshold=0.3,
                                           seed=None,
                                           mb_model_class=MonthlyTIModel,
                                           **kwargs: dict,
                                           ):
    """Bayesian calibration to geodetic MB data with the AdaPBS.

    Experimental. Instead of one parameter set like OGGM's
    ``mb_calibration_from_geodetic_mb``, this returns weighted parameter
    particles (samples of the posterior distribution), with the adaptive
    particle batch smoother of MuSA (see :py:mod:`oggm_adapbs.adapbs`).
    The geometry is fixed (inversion flowlines). Nothing is written to the
    glacier directory.

    The prior is the default one of :py:func:`mb_prior`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to calibrate
    settings_filesuffix: str
        the settings to use
    ref_mb_period : str or list of str, default: PARAMS['geodetic_mb_period']
        the geodetic MB period(s), one observation per period. Do not combine
        overlapping periods (their errors are not independent).
    ref_mb : float or list of float, optional
        use these MB values (kg m-2 yr-1, one per period) instead of the
        geodetic MB file (e.g. for in-situ data).
    ref_mb_err : float or list of float, optional
        the errors (1 sigma) of ref_mb, required with ref_mb.
    file_path : str, optional
        custom geodetic mass-balance file (see utils.get_geodetic_mb_dataframe)
    temp_bias_file_path : str, optional
        temperature bias file, for the centre of the temp_bias prior
    obs_err_factor : float
        multiply the observation errors (err_dmdtda) by this factor
    calibrate_params : list of str
        the parameters to calibrate. The others stay at their prior centre.
    prior_centre : dict, optional
        override the prior centre of some parameters, e.g. {'melt_f': 6}
    prior_bounds : dict, optional
        override the bounds of some parameters, e.g. {'temp_bias': (-2, 2)}
    prior_sd : float or dict
        prior standard deviation in normal space (default: 1). Above ~1.8 the
        logit-normal puts most of its mass near the bounds.
    centre_prior : bool
        if False, the prior is centred in the middle of the bounds (in normal
        space, as in MuSA) instead of on the prior centre
    n_particles : int
        particles per AdaPBS iteration
    max_iterations : int
        maximum number of AdaPBS iterations
    neff_threshold : float
        AdaPBS adapts while the effective sample size is below this fraction
        of n_particles (default 0.3, MuSA uses 0.1)
    seed : int, optional
        random seed
    mb_model_class : MassBalanceModel, default ``MonthlyTIModel``
        the MassBalanceModel to use (with melt_f, temp_bias, prcp_fac)
    kwargs : dict
        kwargs to pass to the mb_model_class instance

    Returns
    -------
    an xarray Dataset with all the particles (dimension ``particle``):
    parameter values, posterior ``weight``, predicted MB (``mb``) and AdaPBS
    ``iteration``; equally weighted resampled particles (``posterior_*``);
    per iteration diagnostics; prior and observations in the attributes.
    """
    # Observations
    if ref_mb_period is None:
        ref_mb_period = gdir.settings['geodetic_mb_period']
    periods = ([ref_mb_period] if isinstance(ref_mb_period, str)
               else list(ref_mb_period))
    if ref_mb is not None:
        if ref_mb_err is None:
            raise InvalidParamsError('ref_mb_err is required with ref_mb.')
        obs = np.atleast_1d(np.asarray(ref_mb, dtype=float))
        obs_err = np.atleast_1d(np.asarray(ref_mb_err, dtype=float))
        if not (len(obs) == len(obs_err) == len(periods)):
            raise InvalidParamsError('Need one ref_mb and ref_mb_err per '
                                     'period.')
    else:
        ref_mb_df = get_geodetic_mb_dataframe(file_path=file_path,
                                              rgi_version=gdir.rgi_version
                                              ).loc[[gdir.rgi_id]]
        obs, obs_err = [], []
        for period in periods:
            df = ref_mb_df.loc[ref_mb_df['period'] == period]
            if len(df) != 1:
                raise InvalidParamsError(f'{gdir.rgi_id}: no geodetic MB '
                                         f'for period {period}.')
            # m w.e. yr-1 -> kg m-2 yr-1
            obs.append(df['dmdtda'].iloc[0] * 1000)
            obs_err.append(df['err_dmdtda'].iloc[0] * 1000)
        obs, obs_err = np.array(obs), np.array(obs_err)
    obs_err = obs_err * obs_err_factor
    years = []
    for period in periods:
        time_resolution, yrs = _floatyears_from_ref_mb_period(period)
        if time_resolution != 'annual':
            raise InvalidParamsError('Only periods of full years for now.')
        years.append(yrs)

    cmb = calving_mb(gdir)
    if cmb != 0:
        raise NotImplementedError('Calving with geodetic MB is not '
                                  'implemented yet.')

    # Prior
    prior = mb_prior(gdir, calibrate_params, prior_centre=prior_centre,
                     prior_bounds=prior_bounds,
                     temp_bias_file_path=temp_bias_file_path)
    calibrate_params = prior['params']
    x_c, x_min, x_max = prior['x_centre'], prior['x_min'], prior['x_max']
    if isinstance(prior_sd, dict):
        prior_sd = np.array([prior_sd[p] for p in calibrate_params],
                            dtype=float)
    prior_mean = glogit(x_c, x_min, x_max) if centre_prior else None

    # Forward model: fixed geometry
    mb_mod, fls = static_mb_model(gdir, prior['centre'],
                                  settings_filesuffix=settings_filesuffix,
                                  mb_model_class=mb_model_class, **kwargs)
    for y in np.concatenate(years):
        if not mb_mod.is_year_valid(y):
            raise ValueError(f'year {y} out of the valid time bounds: '
                             f'[{mb_mod.ys_float}, {mb_mod.ye_float}]')

    def forward(theta, iteration):
        out = np.full((len(theta), len(obs)), np.nan)
        for i, th in enumerate(theta):
            for p, v in zip(calibrate_params, th):
                setattr(mb_mod, p, v)
            for k, yrs in enumerate(years):
                out[i, k] = mb_mod.get_specific_mb(fls=fls, year=yrs).mean()
        return out

    res = run_adapbs(forward, obs, obs_err, x_min, x_max,
                     prior_mean=prior_mean, prior_sd=prior_sd,
                     n_particles=n_particles, max_iterations=max_iterations,
                     neff_threshold=neff_threshold,
                     rng=np.random.default_rng(seed))

    # Output
    ds = xr.Dataset(coords={'particle': np.arange(len(res['weights'])),
                            'resampled': np.arange(n_particles),
                            'obs': periods})
    for k, p in enumerate(calibrate_params):
        ds[p] = ('particle', res['theta'][:, k])
        ds[f'posterior_{p}'] = ('resampled', res['posterior_theta'][:, k])
    ds['weight'] = ('particle', res['weights'])
    ds['iteration'] = ('particle', res['iteration'])
    ds['mb'] = (('particle', 'obs'), res['pred'])
    ds['mb'].attrs['units'] = 'kg m-2 yr-1'
    ds['ref_mb'] = ('obs', obs)
    ds['ref_mb_err'] = ('obs', obs_err)
    diag = res['diagnostics'].set_index('iteration')
    for c in diag.columns:
        ds[f'adapbs_{c}'] = ('adapbs_iteration', diag[c].values)
    ds = ds.assign_coords(adapbs_iteration=diag.index.values)
    ds.attrs.update(rgi_id=gdir.rgi_id,
                    calibrate_params=calibrate_params,
                    prior_centre=x_c, prior_min=x_min, prior_max=x_max,
                    prior_sd=np.broadcast_to(prior_sd, x_c.shape).astype(float),
                    centre_prior=int(centre_prior),
                    fixed_params=str({p: prior['centre'][p] for p in MB_PARAMS
                                      if p not in calibrate_params}),
                    n_particles=n_particles, max_iterations=max_iterations,
                    neff_threshold=neff_threshold,
                    obs_err_factor=obs_err_factor,
                    log_evidence=res['log_evidence'])
    return ds
