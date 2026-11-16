import os

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from oggm import cfg
from oggm.exceptions import InvalidParamsError
from oggm_adapbs.dynamic import (DYNAMIC_PARAMS, dynamic_prior,
                                 mb_param_values,
                                 terminus_step_km2, area_error_km2,
                                 run_dynamic_particle,
                                 run_dynamic_particle_hydro,
                                 dynamic_equifinality_adapbs,
                                 select_particles, dynamic_hydro_rerun,
                                 runoff_metrics, save_netcdf)

# The HISTALP test climate ends in December 2002: the test runs end in 2000
RUN_KWARGS = dict(state_year=2000, spinup_years=20, initial_state_years=10,
                  ref_period=(1990, 2000))


def _params(gdir, **kwargs):
    p = {k: gdir.settings[k] for k in ['melt_f', 'prcp_fac', 'temp_bias']}
    p.update(log10_glen_a_factor=0., initial_state_temp_offset=0.)
    p.update(kwargs)
    return p


def test_select_particles():
    ds = xr.Dataset({'weight': ('particle', [0.5, 0.3, 0.15, 0.05, 0.])})
    idx, w = select_particles(ds, weight_fraction=0.8)
    assert list(idx) == [0, 1]
    assert_allclose(w, [0.5 / 0.8, 0.3 / 0.8])
    idx, w = select_particles(ds, weight_fraction=0.81)
    assert list(idx) == [0, 1, 2]
    idx, w = select_particles(ds)
    assert list(idx) == [0, 1, 2, 3]
    assert_allclose(w.sum(), 1)
    idx, w = select_particles(ds, n_resample=1000, seed=0)
    assert 4 not in idx
    assert_allclose(w.sum(), 1)
    assert_allclose(w[0], 0.5, atol=0.05)


def test_save_netcdf(tmp_path):
    ds = xr.Dataset({'weight': ('particle', [1e-30, 1 - 1e-30]),
                     'runoff': ('particle', [1.234567890123, 2.]),
                     'status': ('particle', ['ok', 'ok'])})
    f = tmp_path / 'test.nc'
    save_netcdf(ds, f)
    with xr.open_dataset(f) as d:
        assert d.weight.dtype == np.float64
        assert d.weight[0] == 1e-30
        assert d.runoff.encoding['dtype'] == np.float32
        assert_allclose(d.runoff, ds.runoff, rtol=1e-6)
        assert list(d.status.values) == ['ok', 'ok']


def test_runoff_metrics():
    # Two particles, 3 years, a runoff of 1 Mt per month (2 in Jul-Sep for
    # the second particle), half of it from melt on the glacier
    monthly = np.ones((2, 3, 12)) * 1e9
    monthly[1, :, 6:9] = 2e9
    dh = xr.Dataset(coords={'particle': [3, 7], 'time': [2000, 2001, 2002],
                            'month': np.arange(1, 13)})
    dh['melt_f'] = ('particle', [4., 6.])
    dh['weight'] = ('particle', [0.4, 0.6])
    dh['runoff_monthly'] = (('particle', 'time', 'month'), monthly)
    dh['runoff'] = dh['runoff_monthly'].sum('month')
    dh['melt_on_glacier'] = dh['runoff'] / 2
    dh.attrs['calibrate_params'] = ['melt_f']
    m = runoff_metrics(dh, period=(2000, 2002))
    assert_allclose(m['runoff'], [12, 15])
    assert_allclose(m['summer_fraction'], [3 / 12, 6 / 15])
    assert_allclose(m['peak_month'][1], 7)
    assert_allclose(m['glacier_melt_share'], 0.5)
    assert_allclose(m['weight'], [0.4, 0.6])
    with pytest.raises(InvalidParamsError):
        runoff_metrics(dh, period=(2000, 2010))


class TestDynamic:

    def test_dynamic_prior(self, hef_gdir):
        prior = dynamic_prior(hef_gdir)
        assert prior['params'] == list(DYNAMIC_PARAMS)
        assert_allclose(prior['x_centre'][3:], [0, 0])
        assert_allclose(prior['x_min'][3:], [-1, -2])
        assert_allclose(prior['x_max'][3:], [1, 2])
        assert prior['glen_a_ref'] == cfg.PARAMS['inversion_glen_a']
        # melt_f and prcp_fac in log10: the log10 of the linear prior
        lin = dynamic_prior(hef_gdir, ['melt_f', 'prcp_fac'])
        assert_allclose(prior['x_centre'][:2], np.log10(lin['x_centre']))
        assert_allclose(prior['x_min'][:2], np.log10(lin['x_min']))
        assert_allclose(prior['x_max'][:2], np.log10(lin['x_max']))
        assert 'melt_f' not in prior['centre']
        assert 'log10_melt_f' not in lin['centre']
        assert_allclose(mb_param_values(prior['centre'])['melt_f'],
                        lin['centre']['melt_f'])
        with pytest.raises(InvalidParamsError):
            dynamic_prior(hef_gdir, ['melt_f', 'log10_melt_f'])

        prior = dynamic_prior(hef_gdir,
                              ['initial_state_temp_offset', 'melt_f'],
                              prior_centre={'initial_state_temp_offset': -1,
                                            'melt_f': 6},
                              prior_bounds={
                                  'initial_state_temp_offset': (-4, 1)})
        assert_allclose(prior['x_centre'], [-1, 6])
        assert_allclose(prior['x_min'], [-4, hef_gdir.settings['melt_f_min']])
        assert prior['centre']['log10_glen_a_factor'] == 0

        with pytest.raises(InvalidParamsError):
            dynamic_prior(hef_gdir, ['glen_a'])
        with pytest.raises(InvalidParamsError):
            dynamic_prior(hef_gdir, prior_bounds={
                'initial_state_temp_offset': (1, 2)})

    def test_area_error(self, hef_gdir):
        fls = hef_gdir.read_store('model_flowlines')
        step = terminus_step_km2(fls)
        assert 0 < step < 0.2
        assert terminus_step_km2(fls, n_last=1) <= step
        assert_allclose(area_error_km2(8, 0), 0.4)
        assert_allclose(area_error_km2(8, 0.2), np.sqrt(0.4 ** 2 + 0.01))

    def test_run_dynamic_particle(self, hef_gdir):
        gdir = hef_gdir
        glen_a_ref = cfg.PARAMS['inversion_glen_a']
        out = {}
        for name, kw in [('ref', {}),
                         ('cold', dict(initial_state_temp_offset=-1)),
                         ('warm', dict(initial_state_temp_offset=1)),
                         ('soft', dict(log10_glen_a_factor=0.5))]:
            out[name] = run_dynamic_particle(gdir, _params(gdir, **kw),
                                             f'_test_{name}',
                                             glen_a_ref=glen_a_ref,
                                             **RUN_KWARGS)
            assert out[name]['status'] == 'ok'
            assert np.all(np.isfinite(out[name]['pred']))
        assert gdir.settings_filesuffix == ''
        assert not [f for f in os.listdir(gdir.dir) if '_test_' in f]

        r = out['ref']
        # Initial state period (10 years) + historical run (21 values)
        assert len(r['volume_km3']) == 31
        assert_allclose(r['volume_km3'][0], r['volume_inversion_km3'])
        assert_allclose(r['volume_km3'][10], r['volume_initial_state_km3'])
        # The MB target is the mass change per RGI area (1990-2000)
        dm = r['mass_kg'][-1] - r['mass_kg'][-11]
        assert_allclose(r['pred'][0], dm / gdir.rgi_area_m2 / 10)
        assert_allclose(r['pred'][2], r['volume_km3'][-1])
        assert_allclose(r['pred'][1], r['area_km2'][-1])
        # Mean thickness (m): volume / total area (>= area_min_h)
        assert 30 < r['pred'][3] < 200
        assert r['pred'][3] <= r['pred'][2] * 1e9 / (r['pred'][1] * 1e6)
        assert 0.3 < r['volume_inversion_km3'] < 1
        # Initial state: colder is bigger
        v = 'volume_initial_state_km3'
        assert (out['cold'][v] > r[v] >
                out['warm'][v])
        # Softer ice is thinner
        assert out['soft']['volume_inversion_km3'] < r['volume_inversion_km3']

        # No initial state period: the historical run starts from the inversion
        r0 = run_dynamic_particle(gdir, _params(gdir), '_test_noinit',
                                  glen_a_ref=glen_a_ref,
                                  **{**RUN_KWARGS, 'initial_state_years': 0})
        assert_allclose(r0['volume_initial_state_km3'],
                        r0['volume_inversion_km3'])
        assert len(r0['volume_km3']) == 21

        # Rejection before the dynamical runs
        rej = run_dynamic_particle(gdir, _params(gdir), '_test_rej',
                                   glen_a_ref=glen_a_ref,
                                   volume_bounds=(10, 20), **RUN_KWARGS)
        assert rej['status'] == 'rejected'
        assert np.all(np.isnan(rej['pred']))
        assert np.isnan(rej['volume_initial_state_km3'])
        assert_allclose(rej['volume_inversion_km3'],
                        r['volume_inversion_km3'])
        assert not [f for f in os.listdir(gdir.dir) if '_test_' in f]

        with pytest.raises(InvalidParamsError):
            run_dynamic_particle(gdir, _params(gdir), '',
                                 glen_a_ref=glen_a_ref, **RUN_KWARGS)

        # A glacier which disappears before state_year: no prediction
        dis = run_dynamic_particle(gdir, _params(gdir, temp_bias=8.),
                                   '_test_dis', glen_a_ref=glen_a_ref,
                                   **RUN_KWARGS)
        assert dis['status'] == 'disappeared'
        assert np.all(np.isnan(dis['pred']))
        assert dis['volume_km3'][-1] == 0
        assert not [f for f in os.listdir(gdir.dir) if '_test_' in f]

    def test_run_dynamic_particle_hydro(self, hef_gdir):
        gdir = hef_gdir
        glen_a_ref = cfg.PARAMS['inversion_glen_a']
        p = _params(gdir, initial_state_temp_offset=-0.5,
                    log10_glen_a_factor=0.3)
        r = run_dynamic_particle(gdir, p, '_test_mem', glen_a_ref=glen_a_ref,
                                 **RUN_KWARGS)
        h = run_dynamic_particle_hydro(gdir, p, '_test_hyd',
                                       glen_a_ref=glen_a_ref,
                                       ref_area_yr=1990, **RUN_KWARGS)
        assert h['status'] == 'ok'
        assert gdir.settings_filesuffix == ''
        assert not [f for f in os.listdir(gdir.dir) if '_test_' in f]
        # Same glacier as the run in memory (historical run only)
        assert len(h['volume_km3']) == 21
        assert_allclose(h['volume_km3'], r['volume_km3'][10:])
        assert_allclose(h['area_km2'], r['area_km2'][10:])
        # Hydro output: the last year is NaN, monthly sums are annual
        hy = h['hydro']
        assert np.isnan(hy['melt_on_glacier'][-1])
        assert np.all(hy['melt_on_glacier'][:-1] > 0)
        assert_allclose(hy['melt_on_glacier_monthly'][:-1].sum(axis=1),
                        hy['melt_on_glacier'][:-1])
        # Catchment: the glacier area of 1990 (index 10), larger where the
        # glacier is larger than in 1990 (OGGM's definition)
        area = hy['on_area'][:-1] + hy['off_area'][:-1]
        assert hy['off_area'][10] == 0
        assert np.all(area >= area[10] * (1 - 1e-9))

        with pytest.raises(InvalidParamsError):
            run_dynamic_particle_hydro(gdir, p, '_test_hyd',
                                       glen_a_ref=glen_a_ref,
                                       ref_area_yr=2010, **RUN_KWARGS)

    @pytest.mark.slow
    def test_dynamic_equifinality_adapbs(self, hef_gdir):
        gdir = hef_gdir
        kwargs = dict(state_year=2000, spinup_years=20, initial_state_years=10,
                      ref_mb_period='1990-01-01_2000-01-01',
                      ref_mb=-700, ref_mb_err=500, mb_err_factor=0.2,
                      ref_volume=0.6, ref_volume_err=0.2,
                      n_particles=30, max_iterations=2, seed=0)
        ds = dynamic_equifinality_adapbs(gdir, **kwargs)
        assert ds.attrs['calibrate_params'] == list(DYNAMIC_PARAMS)
        assert_allclose(ds.melt_f, 10 ** ds.log10_melt_f)
        assert_allclose(ds.posterior_prcp_fac,
                        10 ** ds.posterior_log10_prcp_fac)
        assert_allclose(ds.weight.sum(), 1)
        assert ds.sizes['time'] == 31
        assert ds.time[0] == 1970
        # Default: mean thickness, error 20%
        assert list(ds.obs.values) == ['dmdtda', 'area', 'thickness']
        h = 0.6 / gdir.rgi_area_km2 * 1e3
        assert_allclose(ds.ref_obs, [-700, gdir.rgi_area_km2, h])
        assert_allclose(ds.ref_obs_err[[0, 2]], [100, 0.2 * h])
        assert ds.ref_obs_err[1] > 0.05 * gdir.rgi_area_km2
        for k, p in enumerate(ds.attrs['calibrate_params']):
            assert ds[p].min() >= ds.attrs['prior_min'][k]
            assert ds[p].max() <= ds.attrs['prior_max'][k]
        assert_allclose(ds.glen_a, cfg.PARAMS['inversion_glen_a'] *
                        10 ** ds.log10_glen_a_factor)
        ok = ds.status == 'ok'
        assert ok.sum() > 20
        assert np.all(np.isfinite(ds.pred[ok]))
        assert not [f for f in os.listdir(gdir.dir) if 'adapbs' in f]

        # Hydro rerun of a few resampled particles
        idx, w = select_particles(ds, n_resample=4, seed=0)
        dh = dynamic_hydro_rerun(gdir, ds, idx, w, ref_area_yr=1990)
        assert list(dh.particle.values) == list(idx)
        assert_allclose(dh.weight.sum(), 1)
        assert np.all(dh.status == 'ok')
        assert_allclose(dh.volume_km3,
                        ds.volume_km3.sel(particle=idx).isel(time=slice(10,
                                                                        None)))
        assert_allclose(dh.melt_f, ds.melt_f.sel(particle=idx))
        m = runoff_metrics(dh, period=(1985, 2000))
        assert np.all(m.runoff > 0)
        assert np.all((m.glacier_melt_share > 0) &
                      (m.glacier_melt_share < 1))
        assert not [f for f in os.listdir(gdir.dir) if 'adapbs' in f]

        # Same with a pool of processes, fewer parameters, and the volume
        # with half the error of ref_volume_err
        ds2 = dynamic_equifinality_adapbs(
            gdir, calibrate_params=['melt_f', 'initial_state_temp_offset'],
            n_workers=2, volume_target='volume', volume_rel_err=None,
            **kwargs)
        assert list(ds2.obs.values) == ['dmdtda', 'area', 'volume']
        assert_allclose(ds2.ref_obs[2], 0.6)
        assert_allclose(ds2.ref_obs_err[2], 0.5 * 0.2)
        assert ds2.attrs['calibrate_params'] == ['melt_f',
                                                 'initial_state_temp_offset']
        assert_allclose(ds2.glen_a, cfg.PARAMS['inversion_glen_a'])
        assert (ds2.status == 'ok').sum() > 20
        assert not [f for f in os.listdir(gdir.dir) if 'adapbs' in f]

        with pytest.raises(InvalidParamsError):
            dynamic_equifinality_adapbs(gdir,
                                        **{**kwargs, 'initial_state_years': 0})
        with pytest.raises(InvalidParamsError):
            dynamic_equifinality_adapbs(gdir, volume_target='mass', **kwargs)
        with pytest.raises(InvalidParamsError):
            # The climate ends in 2002
            dynamic_equifinality_adapbs(gdir, **{**kwargs, 'state_year': 2004})
