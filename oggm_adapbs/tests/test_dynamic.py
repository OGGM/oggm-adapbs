import os

import numpy as np
import pytest
from numpy.testing import assert_allclose

from oggm import cfg
from oggm.exceptions import InvalidParamsError
from oggm_adapbs.dynamic import (DYNAMIC_PARAMS, dynamic_prior,
                                 terminus_step_km2, area_error_km2,
                                 run_dynamic_particle,
                                 dynamic_calibration_adapbs)

# The HISTALP test climate ends in December 2002: the test runs end in 2000
RUN_KWARGS = dict(state_year=2000, spinup_years=20, growth_years=10,
                  ref_period=(1990, 2000))


def _params(gdir, **kwargs):
    p = {k: gdir.settings[k] for k in ['melt_f', 'prcp_fac', 'temp_bias']}
    p.update(log10_glen_a_factor=0., growth_temp_offset=0.)
    p.update(kwargs)
    return p


class TestDynamic:

    def test_dynamic_prior(self, hef_gdir):
        prior = dynamic_prior(hef_gdir)
        assert prior['params'] == list(DYNAMIC_PARAMS)
        assert_allclose(prior['x_centre'][3:], [0, 0])
        assert_allclose(prior['x_min'][3:], [-1, -3])
        assert_allclose(prior['x_max'][3:], [1, 3])
        assert prior['glen_a_ref'] == cfg.PARAMS['inversion_glen_a']

        prior = dynamic_prior(hef_gdir, ['growth_temp_offset', 'melt_f'],
                              prior_centre={'growth_temp_offset': -1,
                                            'melt_f': 6},
                              prior_bounds={'growth_temp_offset': (-4, 1)})
        assert_allclose(prior['x_centre'], [-1, 6])
        assert_allclose(prior['x_min'], [-4, hef_gdir.settings['melt_f_min']])
        assert prior['centre']['log10_glen_a_factor'] == 0

        with pytest.raises(InvalidParamsError):
            dynamic_prior(hef_gdir, ['glen_a'])
        with pytest.raises(InvalidParamsError):
            dynamic_prior(hef_gdir, prior_bounds={'growth_temp_offset': (1, 2)})

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
                         ('cold', dict(growth_temp_offset=-1)),
                         ('warm', dict(growth_temp_offset=1)),
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
        # Growth period (10 years) + historical run (21 values)
        assert len(r['volume_km3']) == 31
        assert_allclose(r['volume_km3'][0], r['volume_inversion_km3'])
        assert_allclose(r['volume_km3'][10], r['volume_growth_km3'])
        # The MB target is the mass change per RGI area (1990-2000)
        dm = r['mass_kg'][-1] - r['mass_kg'][-11]
        assert_allclose(r['pred'][0], dm / gdir.rgi_area_m2 / 10)
        assert_allclose(r['pred'][2], r['volume_km3'][-1])
        assert_allclose(r['pred'][1], r['area_km2'][-1])
        # Mean thickness (m): volume / total area (>= area_min_h)
        assert 30 < r['pred'][3] < 200
        assert r['pred'][3] <= r['pred'][2] * 1e9 / (r['pred'][1] * 1e6)
        assert 0.3 < r['volume_inversion_km3'] < 1
        # Growth: colder is bigger
        assert (out['cold']['volume_growth_km3'] > r['volume_growth_km3'] >
                out['warm']['volume_growth_km3'])
        # Softer ice is thinner
        assert out['soft']['volume_inversion_km3'] < r['volume_inversion_km3']

        # No growth period: the historical run starts from the inversion
        r0 = run_dynamic_particle(gdir, _params(gdir), '_test_nogrowth',
                                  glen_a_ref=glen_a_ref,
                                  **{**RUN_KWARGS, 'growth_years': 0})
        assert_allclose(r0['volume_growth_km3'], r0['volume_inversion_km3'])
        assert len(r0['volume_km3']) == 21

        # Rejection before the dynamical runs
        rej = run_dynamic_particle(gdir, _params(gdir), '_test_rej',
                                   glen_a_ref=glen_a_ref,
                                   volume_bounds=(10, 20), **RUN_KWARGS)
        assert rej['status'] == 'rejected'
        assert np.all(np.isnan(rej['pred']))
        assert np.isnan(rej['volume_growth_km3'])
        assert_allclose(rej['volume_inversion_km3'],
                        r['volume_inversion_km3'])
        assert not [f for f in os.listdir(gdir.dir) if '_test_' in f]

        with pytest.raises(InvalidParamsError):
            run_dynamic_particle(gdir, _params(gdir), '',
                                 glen_a_ref=glen_a_ref, **RUN_KWARGS)

    @pytest.mark.slow
    def test_dynamic_calibration_adapbs(self, hef_gdir):
        gdir = hef_gdir
        kwargs = dict(state_year=2000, spinup_years=20, growth_years=10,
                      ref_mb_period='1990-01-01_2000-01-01',
                      ref_mb=-700, ref_mb_err=500, mb_err_factor=0.2,
                      ref_volume=0.6, ref_volume_err=0.2,
                      n_particles=30, max_iterations=2, seed=0)
        ds = dynamic_calibration_adapbs(gdir, **kwargs)
        assert ds.attrs['calibrate_params'] == list(DYNAMIC_PARAMS)
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

        # Same with a pool of processes, fewer parameters, and the volume
        # with the error of ref_volume_err
        ds2 = dynamic_calibration_adapbs(
            gdir, calibrate_params=['melt_f', 'growth_temp_offset'],
            n_workers=2, volume_target='volume', volume_rel_err=None,
            **kwargs)
        assert list(ds2.obs.values) == ['dmdtda', 'area', 'volume']
        assert_allclose(ds2.ref_obs[2], 0.6)
        assert_allclose(ds2.ref_obs_err[2], 0.2)
        assert ds2.attrs['calibrate_params'] == ['melt_f',
                                                 'growth_temp_offset']
        assert_allclose(ds2.glen_a, cfg.PARAMS['inversion_glen_a'])
        assert (ds2.status == 'ok').sum() > 20
        assert not [f for f in os.listdir(gdir.dir) if 'adapbs' in f]

        with pytest.raises(InvalidParamsError):
            dynamic_calibration_adapbs(gdir, **{**kwargs, 'growth_years': 0})
        with pytest.raises(InvalidParamsError):
            dynamic_calibration_adapbs(gdir, volume_target='mass', **kwargs)
        with pytest.raises(InvalidParamsError):
            # The climate ends in 2002
            dynamic_calibration_adapbs(gdir, **{**kwargs, 'state_year': 2004})
