"""The plausible runs of one glacier with AdaPBS, then their runoff.

The command line version of the notebooks ``dynamic_equifinality.ipynb`` and
``dynamic_runoff_sensitivity.ipynb``, for many glaciers or on a cluster (see
``scripts/README.md``). The defaults are the settings of the notebooks.

Example, on a laptop with 6 processes:

    python scripts/run_glacier.py RGI60-11.00719 --working-dir wd \\
        --output-dir out --n-workers 6

It writes, in the output directory:

- ``<rgi_id>_adapbs.nc``: all the AdaPBS runs and their weights
  (:py:func:`oggm_adapbs.dynamic.dynamic_equifinality_adapbs`);
- ``<rgi_id>_hydro.nc``: the plausible runs rerun with OGGM's hydrological
  output (:py:func:`oggm_adapbs.dynamic.dynamic_hydro_rerun`), unless
  ``--no-hydro``;
- ``<rgi_id>_runoff_metrics.nc``: the runoff metrics of these runs
  (:py:func:`oggm_adapbs.dynamic.runoff_metrics`);
- ``<rgi_id>_log.json``: settings, run times and the status of each step.
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings

BASE_URL = ('https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/'
            'L3-L5_files/2025.6/elev_bands/W5E5/per_glacier/')
TB_FILE = ('https://cluster.klima.uni-bremen.de/~oggm/ref_mb_params/oggm_v1.6/'
           'w5e5_rgi6_perglacier_temp_bias_v2025.6.2.csv')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Plausible runs of one glacier with AdaPBS, then their '
                    'runoff.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('rgi_id', help='RGI id of the glacier (RGI6)')
    p.add_argument('--working-dir', required=True,
                   help='OGGM working directory (one per run happening at '
                        'the same time)')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--n-workers', type=int, default=1,
                   help='processes running the particles in parallel')
    # Data
    p.add_argument('--base-url', default=BASE_URL,
                   help='preprocessed directories (level 3)')
    p.add_argument('--prepro-border', type=int, default=80)
    p.add_argument('--temp-bias-file', default=TB_FILE)
    # The model runs
    p.add_argument('--spinup-years', type=int, default=30,
                   help='historical run before the RGI date (years)')
    p.add_argument('--initial-state-years', type=int, default=20,
                   help='initial state period (0: none)')
    # Observation errors
    p.add_argument('--mb-err-factor', type=float, default=0.2,
                   help='factor on the geodetic MB error')
    p.add_argument('--volume-rel-err', type=float, default=0.2,
                   help='error of the mean thickness, relative, the same '
                        'for all glaciers')
    p.add_argument('--volume-err-factor', type=float, default=None,
                   help='instead of --volume-rel-err: the error of the mean '
                        'thickness is this factor times the relative '
                        'IceBoost error of the glacier')
    # Prior: wide, nearly uniform (logit-normal centred mid-range)
    p.add_argument('--prior-sd', type=float, default=1.4)
    p.add_argument('--temp-bias-range', type=float, default=10,
                   help='temp_bias bounds: informed value +- this (K)')
    p.add_argument('--glen-a-range', type=float, default=1,
                   help='log10 A factor bounds: +- this')
    p.add_argument('--prcp-fac-min', type=float, default=0.5)
    p.add_argument('--prcp-fac-max', type=float, default=5)
    p.add_argument('--offset-min', type=float, default=-3,
                   help='initial state offset, lower bound (K)')
    p.add_argument('--offset-max', type=float, default=2,
                   help='initial state offset, upper bound (K)')
    # AdaPBS
    p.add_argument('--n-particles', type=int, default=500)
    p.add_argument('--max-iterations', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    # Hydro rerun
    p.add_argument('--no-hydro', action='store_true',
                   help='skip the hydro rerun')
    p.add_argument('--weight-fraction', type=float, default=0.999,
                   help='rerun the particles holding this fraction of the '
                        'weight')
    p.add_argument('--ref-area-yr', type=int, default=2000,
                   help='year of the glacier area defining the catchment')
    return p.parse_args(argv)


def init_gdir(rgi_id, args):
    """Open the glacier directory from disk if it exists (initialising it
    from the URL again would overwrite it), else from the preprocessed
    directories."""
    from oggm import workflow
    from oggm.cfg import PATHS
    path = os.path.join(PATHS['working_dir'], 'per_glacier', rgi_id[:8],
                        rgi_id[:11], rgi_id)
    if os.path.isdir(path):
        return workflow.init_glacier_directories([rgi_id])[0]
    return workflow.init_glacier_directories(
        [rgi_id], from_prepro_level=3, prepro_border=args.prepro_border,
        prepro_base_url=args.base_url)[0]


def wide_prior(gdir, args):
    """The bounds and centres of the wide prior of the notebooks: OGGM's
    bounds for melt_f, prcp_fac between prcp_fac_min and prcp_fac_max (both
    in log10), the informed temp_bias +- temp_bias_range, A from 0.1 to 10
    times the default, the initial state offset between offset_min and
    offset_max. The centres are mid-range."""
    import numpy as np
    from oggm_adapbs.dynamic import dynamic_prior
    tb = dynamic_prior(gdir, temp_bias_file_path=args.temp_bias_file
                       )['centre']['temp_bias']
    s = gdir.settings
    bounds = {'log10_melt_f': tuple(np.log10([s['melt_f_min'],
                                              s['melt_f_max']])),
              'log10_prcp_fac': tuple(np.log10([args.prcp_fac_min,
                                                args.prcp_fac_max])),
              'temp_bias': (tb - args.temp_bias_range,
                            tb + args.temp_bias_range),
              'log10_glen_a_factor': (-args.glen_a_range, args.glen_a_range),
              'initial_state_temp_offset': (args.offset_min,
                                            args.offset_max)}
    centre = {p: (lo + hi) / 2 for p, (lo, hi) in bounds.items()}
    return centre, bounds


def main(argv=None):
    args = parse_args(argv)
    from oggm import cfg
    from oggm_adapbs.dynamic import (DYNAMIC_PARAMS,
                                     dynamic_equifinality_adapbs,
                                     select_particles, dynamic_hydro_rerun,
                                     runoff_metrics, save_netcdf)
    rgi_id = args.rgi_id
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, rgi_id)
    log = dict(rgi_id=rgi_id, settings=vars(args), status='started')

    def write_log():
        with open(f'{out}_log.json', 'w') as f:
            json.dump(log, f, indent=1, default=str)

    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = args.working_dir
    warnings.filterwarnings('ignore', message='Store data not found')
    try:
        t0 = time.time()
        gdir = init_gdir(rgi_id, args)
        centre, bounds = wide_prior(gdir, args)
        params = list(DYNAMIC_PARAMS)
        if args.initial_state_years == 0:
            params.remove('initial_state_temp_offset')
        ds = dynamic_equifinality_adapbs(
            gdir, spinup_years=args.spinup_years,
            initial_state_years=args.initial_state_years,
            mb_err_factor=args.mb_err_factor,
            volume_rel_err=(args.volume_rel_err
                            if args.volume_err_factor is None else None),
            volume_err_factor=(1. if args.volume_err_factor is None
                               else args.volume_err_factor),
            calibrate_params=params, prior_centre=centre,
            prior_bounds=bounds, prior_sd=args.prior_sd,
            temp_bias_file_path=args.temp_bias_file,
            n_particles=args.n_particles,
            max_iterations=args.max_iterations, seed=args.seed,
            n_workers=args.n_workers)
        ds.attrs['wall_time_s'] = time.time() - t0
        save_netcdf(ds, f'{out}_adapbs.nc')
        log.update(adapbs_runs=ds.sizes['particle'],
                   adapbs_iterations=ds.sizes['adapbs_iteration'],
                   adapbs_wall_time_s=ds.attrs['wall_time_s'],
                   log_evidence=ds.attrs['log_evidence'],
                   status='adapbs done')
        write_log()

        if not args.no_hydro:
            t0 = time.time()
            idx, w = select_particles(ds, weight_fraction=args.weight_fraction)
            dh = dynamic_hydro_rerun(gdir, ds, idx, w,
                                     ref_area_yr=args.ref_area_yr,
                                     n_workers=args.n_workers)
            dh.attrs['wall_time_s'] = time.time() - t0
            save_netcdf(dh, f'{out}_hydro.nc')
            save_netcdf(runoff_metrics(dh), f'{out}_runoff_metrics.nc')
            log.update(hydro_runs=len(idx),
                       hydro_failed=int((dh.status != 'ok').sum()),
                       hydro_wall_time_s=dh.attrs['wall_time_s'])
        log['status'] = 'ok'
    except Exception:
        log['status'] = 'error'
        log['error'] = traceback.format_exc()
        write_log()
        raise
    write_log()


if __name__ == '__main__':  # needed: the particles run in new processes
    sys.exit(main())
