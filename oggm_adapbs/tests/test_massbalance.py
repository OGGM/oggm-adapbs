import numpy as np
import pytest
from numpy.testing import assert_allclose

from oggm import cfg
from oggm.utils import date_to_floatyear
from oggm.exceptions import InvalidParamsError
from oggm_adapbs.adapbs import weighted_quantile
from oggm_adapbs.massbalance import (mb_prior, static_mb_model,
                                     seasonal_specific_mb,
                                     mb_calibration_from_geodetic_mb_adapbs)


class TestMBCalibration:

    def test_mb_prior(self, hef_gdir):
        gdir = hef_gdir
        prior = mb_prior(gdir)
        assert prior['params'] == ['melt_f', 'prcp_fac', 'temp_bias']
        assert_allclose(prior['x_centre'], [cfg.PARAMS['melt_f'],
                                            cfg.PARAMS['prcp_fac'], 0])
        assert_allclose(prior['x_min'], [gdir.settings['melt_f_min'],
                                         cfg.PARAMS['prcp_fac'] / 2, -3])
        assert_allclose(prior['x_max'], [gdir.settings['melt_f_max'],
                                         cfg.PARAMS['prcp_fac'] * 2, 3])

        prior = mb_prior(gdir, ['temp_bias', 'melt_f'],
                         prior_centre={'melt_f': 6, 'prcp_fac': 3},
                         prior_bounds={'temp_bias': (-1, 2)})
        assert prior['params'] == ['temp_bias', 'melt_f']
        assert_allclose(prior['x_centre'], [0, 6])
        assert_allclose(prior['x_min'][0], -1)
        assert_allclose(prior['x_max'][0], 2)
        assert prior['centre']['prcp_fac'] == 3

        with pytest.raises(InvalidParamsError):
            mb_prior(gdir, ['bias'])
        with pytest.raises(InvalidParamsError):
            mb_prior(gdir, prior_bounds={'melt_f': (6, 10)})

    def test_seasonal_specific_mb(self, hef_gdir):
        gdir = hef_gdir
        mb_mod, fls = static_mb_model(gdir, mb_prior(gdir)['centre'])
        years = np.arange(1990, 2003)

        # Calendar years: same as OGGM's annual specific MB
        df = seasonal_specific_mb(mb_mod, fls, years, start_month=1,
                                  summer_start_month=5)
        ref = mb_mod.get_specific_mb(fls=fls, year=years)
        assert_allclose(df['annual'], ref, rtol=1e-6)
        assert_allclose(df['annual'], df['winter'] + df['summer'])

        # Hydrological years: October to September
        df = seasonal_specific_mb(mb_mod, fls, years)
        monthly = mb_mod.get_specific_mb(
            fls=fls, year=date_to_floatyear([1999] * 3 + [2000] * 9,
                                    [10, 11, 12] + list(range(1, 10))),
            time_resolution='monthly')
        assert_allclose(df.loc[2000, 'winter'], monthly[:7].sum())
        assert_allclose(df.loc[2000, 'summer'], monthly[7:].sum())
        assert np.all(df['winter'] > 0)
        assert np.all(df['summer'] < 0)

        # Southern hemisphere seasons
        df = seasonal_specific_mb(mb_mod, fls, years[1:], start_month=4,
                                  summer_start_month=10)
        monthly = mb_mod.get_specific_mb(
            fls=fls, year=date_to_floatyear([1999] * 9 + [2000] * 3,
                                    list(range(4, 13)) + [1, 2, 3]),
            time_resolution='monthly')
        assert_allclose(df.loc[2000, 'winter'], monthly[:6].sum())
        assert_allclose(df.loc[2000, 'summer'], monthly[6:].sum())

    @pytest.mark.slow
    def test_mb_calibration_adapbs(self, hef_gdir):
        from oggm.core import massbalance

        gdir = hef_gdir
        mbdf = gdir.get_ref_mb_data()['ANNUAL_BALANCE']
        ref_mb = mbdf.mean()
        period = f'{mbdf.index[0]}-01-01_{mbdf.index[-1] + 1}-01-01'

        # Deterministic: melt_f only, prcp_fac and temp_bias=0 from cfg
        det = massbalance.mb_calibration_from_scalar_mb(
            gdir, ref_mb=ref_mb, ref_mb_period=period, write_to_gdir=False,
            overwrite_observations=True)

        # One parameter, small error: the posterior is around det
        ds = mb_calibration_from_geodetic_mb_adapbs(
            gdir, ref_mb=ref_mb, ref_mb_err=20, ref_mb_period=period,
            calibrate_params=['melt_f'], n_particles=200, seed=0)
        assert ds.attrs['calibrate_params'] == ['melt_f']
        assert_allclose(ds.attrs['prior_centre'], [5])
        w = ds.weight.values
        med = weighted_quantile(ds.melt_f.values, w, 0.5)
        assert_allclose(med, det['melt_f'], atol=0.3)
        assert_allclose(np.average(ds.mb[:, 0], weights=w), ref_mb, atol=20)
        assert np.median(ds.posterior_melt_f) == pytest.approx(med, abs=0.3)

        # Three parameters: runs, weights sum to 1, within the bounds
        ds = mb_calibration_from_geodetic_mb_adapbs(
            gdir, ref_mb=ref_mb, ref_mb_err=100, ref_mb_period=period,
            n_particles=100, max_iterations=3, seed=0)
        assert_allclose(ds.weight.sum(), 1)
        for k, p in enumerate(ds.attrs['calibrate_params']):
            assert ds[p].min() >= ds.attrs['prior_min'][k]
            assert ds[p].max() <= ds.attrs['prior_max'][k]
        assert_allclose(ds.attrs['prior_centre'][1], 2.5)
        assert len(ds.adapbs_iteration) >= 1
