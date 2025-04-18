# Python imports
import time
from typing import Optional, TYPE_CHECKING

# Third-Party Imports
import numpy
import scipy.stats

# PyCSEP imports
from csep.core.exceptions import CSEPEvaluationException
from csep.core.catalogs import CSEPCatalog
from csep.models import (
    CatalogNumberTestResult,
    CatalogSpatialTestResult,
    CatalogMagnitudeTestResult,
    CatalogPseudolikelihoodTestResult,
    CalibrationTestResult
)
from csep.utils.calc import _compute_likelihood
from csep.utils.stats import get_quantiles, cumulative_square_diff, MLL_score

if TYPE_CHECKING:
    from csep.core.forecasts import CatalogForecast


def number_test(forecast, observed_catalog, verbose=True):
    """ Performs the number test on a catalog-based forecast.

    The number test builds an empirical distribution of the event counts for each data. By default, this
    function does not perform any filtering on the catalogs in the forecast or observation. These should be handled
    outside of the function.

    Args:
        forecast (:class:`csep.core.forecasts.CatalogForecast`): forecast to evaluate
        observed_catalog (:class:`csep.core.catalogs.AbstractBaseCatalog`): evaluation data

    Returns:
        evaluation result (:class:`csep.models.EvaluationResult`): evaluation result
    """
    event_counts = []
    t0 = time.time()
    for i, catalog in enumerate(forecast):
        # output status
        if verbose:
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)
        event_counts.append(catalog.event_count)
    obs_count = observed_catalog.event_count
    delta_1, delta_2 = get_quantiles(event_counts, obs_count)
    # prepare result
    result = CatalogNumberTestResult(test_distribution=event_counts,
                                     name='Catalog N-Test',
                                     observed_statistic=obs_count,
                                     quantile=(delta_1, delta_2),
                                     status='normal',
                                     obs_catalog_repr=str(observed_catalog),
                                     sim_name=forecast.name,
                                     min_mw=forecast.min_magnitude,
                                     obs_name=observed_catalog.name)
    return result


def spatial_test(forecast, observed_catalog, verbose=True):
    """ Performs spatial test for catalog-based forecasts.



        Args:
            forecast: CatalogForecast
            observed_catalog: CSEPCatalog filtered to be consistent with the forecast

        Returns:
            CatalogSpatialTestResult
    """

    if forecast.region is None:
        raise CSEPEvaluationException("Forecast must have region member to perform spatial test.")

    # get observed likelihood
    if observed_catalog.event_count == 0:
        print(f'Spatial test not-invalid because no events in observed catalog.')

    test_distribution = []

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        forecast.get_expected_rates(verbose=verbose)

    expected_cond_count = forecast.expected_rates.sum()
    forecast_mean_spatial_rates = forecast.expected_rates.spatial_counts()

    # summing over spatial counts ensures that the correct number of events are used; even through the catalogs should
    # be filtered before calling this function
    gridded_obs = observed_catalog.spatial_counts()
    n_obs = numpy.sum(gridded_obs)

    # iterate through catalogs in forecast and compute likelihood
    t0 = time.time()
    for i, catalog in enumerate(forecast):
        gridded_cat = catalog.spatial_counts()
        _, lh_norm = _compute_likelihood(gridded_cat, forecast_mean_spatial_rates, expected_cond_count, n_obs)
        test_distribution.append(lh_norm)
        # output status
        if verbose:
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)

    _, obs_lh_norm = _compute_likelihood(gridded_obs, forecast_mean_spatial_rates, expected_cond_count, n_obs)
    # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
    message = "normal"
    if obs_lh_norm == -numpy.inf:
        idx_good_sim = forecast_mean_spatial_rates != 0
        new_gridded_obs = gridded_obs[idx_good_sim]
        new_n_obs = numpy.sum(new_gridded_obs)
        print(f"Found -inf as the observed likelihood score. "
              f"Assuming event(s) occurred in undersampled region of forecast.\n"
              f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")

        new_ard = forecast_mean_spatial_rates[idx_good_sim]
        _, obs_lh_norm = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count, n_obs)
        message = "undersampled"

    # check for nans here and remove from spatial distribution
    test_distribution_spatial_1d = numpy.array(test_distribution)
    if numpy.isnan(numpy.sum(test_distribution_spatial_1d)):
        test_distribution_spatial_1d = test_distribution_spatial_1d[~numpy.isnan(test_distribution_spatial_1d)]

    if n_obs == 0 or numpy.isnan(obs_lh_norm):
        message = "not-valid"
        delta_1, delta_2 = -1, -1
    else:
        delta_1, delta_2 = get_quantiles(test_distribution_spatial_1d, obs_lh_norm)

    result = CatalogSpatialTestResult(test_distribution=test_distribution_spatial_1d,
                                      name='S-Test',
                                      observed_statistic=obs_lh_norm,
                                      quantile=(delta_1, delta_2),
                                      status=message,
                                      min_mw=forecast.min_magnitude,
                                      obs_catalog_repr=str(observed_catalog),
                                      sim_name=forecast.name,
                                      obs_name=observed_catalog.name)

    return result


def magnitude_test(forecast, observed_catalog, verbose=True):
    """ Performs magnitude test for catalog-based forecasts """
    test_distribution = []

    if forecast.region.magnitudes is None:
        raise CSEPEvaluationException("Forecast must have region.magnitudes member to perform magnitude test.")

    # short-circuit if zero events
    if observed_catalog.event_count == 0:
        print("Cannot perform magnitude test when observed event count is zero.")
        # prepare result
        result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                                            name='M-Test',
                                            observed_statistic=None,
                                            quantile=(None, None),
                                            status='not-valid',
                                            min_mw=forecast.min_magnitude,
                                            obs_catalog_repr=str(observed_catalog),
                                            obs_name=observed_catalog.name,
                                            sim_name=forecast.name)

        return result

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        forecast.get_expected_rates(verbose=verbose)

    # returns the average events in the magnitude bins
    union_histogram = forecast.expected_rates.magnitude_counts()
    n_union_events = numpy.sum(union_histogram)
    obs_histogram = observed_catalog.magnitude_counts()
    n_obs = numpy.sum(obs_histogram)
    union_scale = n_obs / n_union_events
    scaled_union_histogram = union_histogram * union_scale

    # compute the test statistic for each catalog
    t0 = time.time()
    for i, catalog in enumerate(forecast):
        mag_counts = catalog.magnitude_counts()
        n_events = numpy.sum(mag_counts)
        if n_events == 0:
            # print("Skipping to next because catalog contained zero events.")
            continue
        scale = n_obs / n_events
        catalog_histogram = mag_counts * scale
        # compute magnitude test statistic for the catalog
        test_distribution.append(
            cumulative_square_diff(numpy.log10(catalog_histogram + 1), numpy.log10(scaled_union_histogram + 1))
        )
        # output status
        if verbose:
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)

    # compute observed statistic
    obs_d_statistic = cumulative_square_diff(numpy.log10(obs_histogram + 1), numpy.log10(scaled_union_histogram + 1))

    # score evaluation
    delta_1, delta_2 = get_quantiles(test_distribution, obs_d_statistic)

    # prepare result
    result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                              name='M-Test',
                              observed_statistic=obs_d_statistic,
                              quantile=(delta_1, delta_2),
                              status='normal',
                              min_mw=forecast.min_magnitude,
                              obs_catalog_repr=str(observed_catalog),
                              obs_name=observed_catalog.name,
                              sim_name=forecast.name)

    return result


def pseudolikelihood_test(forecast, observed_catalog, verbose=True):
    """ Performs the spatial pseudolikelihood test for catalog forecasts.

    Performs the spatial pseudolikelihood test as described by Savran et al., 2020. The tests uses a pseudolikelihood
    statistic computed from the expected rates in spatial cells. A pseudolikelihood test based on space-magnitude bins
    is in a development mode and does not exist currently.

    Args:
        forecast: :class:`csep.core.forecasts.CatalogForecast`
        observed_catalog: :class:`csep.core.catalogs.AbstractBaseCatalog`
    """

    if forecast.region is None:
        raise CSEPEvaluationException("Forecast must have region member to perform spatial test.")

    # get observed likelihood
    if observed_catalog.event_count == 0:
        print(f'Skipping pseudolikelihood test because no events in observed catalog.')
        return None

    test_distribution = []

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        _ = forecast.get_expected_rates(verbose=verbose)

    expected_cond_count = forecast.expected_rates.sum()
    forecast_mean_spatial_rates = forecast.expected_rates.spatial_counts()

    # summing over spatial counts ensures that the correct number of events are used; even through the catalogs should
    # be filtered before calling this function
    gridded_obs = observed_catalog.spatial_counts()
    n_obs = numpy.sum(gridded_obs)

    t0 = time.time()
    for i, catalog in enumerate(forecast):
        gridded_cat = catalog.spatial_counts()
        plh, _ = _compute_likelihood(gridded_cat, forecast_mean_spatial_rates, expected_cond_count, n_obs)
        test_distribution.append(plh)
        # output status
        if verbose:
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)

    obs_plh, _ = _compute_likelihood(gridded_obs, forecast_mean_spatial_rates, expected_cond_count, n_obs)
    # if obs_lh is -numpy.inf, recompute but only for indexes where obs and simulated are non-zero
    message = "normal"
    if obs_plh == -numpy.inf:
        idx_good_sim = forecast_mean_spatial_rates != 0
        new_gridded_obs = gridded_obs[idx_good_sim]
        new_n_obs = numpy.sum(new_gridded_obs)
        print(f"Found -inf as the observed likelihood score. "
              f"Assuming event(s) occurred in undersampled region of forecast.\n"
              f"Recomputing with {new_n_obs} events after removing {n_obs - new_n_obs} events.")
        if new_n_obs == 0:
            print(
                f'Skipping pseudo-likelihood based tests for because no events in observed catalog '
                f'after correcting for under-sampling in forecast.'
            )
            return None

        new_ard = forecast_mean_spatial_rates[idx_good_sim]
        # we need to use the old n_obs here, because if we normalize the ard to a different value the observed
        # statistic will not be computed correctly.
        obs_plh, _ = _compute_likelihood(new_gridded_obs, new_ard, expected_cond_count, n_obs)
        message = "undersampled"

    # check for nans here
    test_distribution_1d = numpy.array(test_distribution)
    if numpy.isnan(numpy.sum(test_distribution_1d)):
        test_distribution_1d = test_distribution_1d[~numpy.isnan(test_distribution_1d)]

    if n_obs == 0 or numpy.isnan(obs_plh):
        message = "not-valid"
        delta_1, delta_2 = -1, -1
    else:
        delta_1, delta_2 = get_quantiles(test_distribution_1d, obs_plh)

    # prepare evaluation result
    result = CatalogPseudolikelihoodTestResult(
        test_distribution=test_distribution_1d,
        name='PL-Test',
        observed_statistic=obs_plh,
        quantile=(delta_1, delta_2),
        status=message,
        min_mw=forecast.min_magnitude,
        obs_catalog_repr=str(observed_catalog),
        sim_name=forecast.name,
        obs_name=observed_catalog.name
    )

    return result


def calibration_test(evaluation_results, delta_1=False):
    """ Perform the calibration test by computing a Kilmogorov-Smirnov test of the observed quantiles against a uniform
    distribution.

        Args:
            evaluation_results: iterable of evaluation result objects
            delta_1 (bool): use delta_1 for quantiles. default false -> use delta_2 quantile score for calibration test
    """

    # this is using "delta_2" which is the cdf value less-equal
    idx = 0 if delta_1 else 1
    quantiles = []
    for result in evaluation_results:
        if result.status == 'not-valid':
            print(f'evaluation not valid for {result.name}. skipping in calibration test.')
        else:
            quantiles.append(result.quantile[idx])

    ks, p_value = scipy.stats.kstest(quantiles, 'uniform')

    result = CalibrationTestResult(
        test_distribution = quantiles,
        name=f'{evaluation_results[0].name} Calibration Test',
        observed_statistic=ks,
        quantile=p_value,
        status='normal',
        min_mw = evaluation_results[0].min_mw,
        obs_catalog_repr=evaluation_results[0].obs_catalog_repr,
        sim_name=evaluation_results[0].sim_name,
        obs_name=evaluation_results[0].obs_name
    )

    return result


def resampled_magnitude_test(forecast: "CatalogForecast",
                             observed_catalog: CSEPCatalog,
                             verbose: bool = False,
                             seed: Optional[int] = None) -> CatalogMagnitudeTestResult:
    """
    Performs the resampled magnitude test for catalog-based forecasts (Serafini et al., 2024),
    which corrects the bias from the original M-test implementation to the total N of events.
    Calculates the (pseudo log-likelihood) test statistic distribution from the forecast's
    synthetic catalogs Λ_j as:

        D_j = Σ_k [log(Λ_u(k) * N / N_u + 1) - log(Λ̃_j(k) + 1)] ^ 2

    where k are the magnitude bins, Λ_u the union of all synthetic catalogs, N_u the total
    number of events in Λ_u, and Λ̃_j the resampled catalog containing exactly N events.

    The pseudo log-likelihood statistic from the observations is calculated as:

        D_o = Σ_k [log(Λ_U(k) * N / N_u + 1) - log(Ω(k) + 1)]^2

    where Ω is the observed catalog.

    Args:
        forecast (CatalogForecast): The forecast to be evaluated
        observed_catalog (CSEPCatalog): The observation/testing catalog.
        verbose (bool): Flag to display debug messages
        seed (int): Random number generator seed

    Returns:
        A CatalogMagnitudeTestResult object containing the statistic distribution and the
        observed statistic.
    """

    # set seed
    if seed:
        numpy.random.seed(seed)
    """  """
    test_distribution = []

    if forecast.region.magnitudes is None:
        raise CSEPEvaluationException(
            "Forecast must have region.magnitudes member to perform magnitude test.")

    # short-circuit if zero events
    if observed_catalog.event_count == 0:
        print("Cannot perform magnitude test when observed event count is zero.")
        # prepare result
        result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                                            name='M-Test',
                                            observed_statistic=None,
                                            quantile=(None, None),
                                            status='not-valid',
                                            min_mw=forecast.min_magnitude,
                                            obs_catalog_repr=str(observed_catalog),
                                            obs_name=observed_catalog.name,
                                            sim_name=forecast.name)

        return result

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        forecast.get_expected_rates(verbose=verbose)

    # THIS IS NEW - returns the average events in the magnitude bins
    union_histogram = numpy.zeros(len(forecast.magnitudes))
    for j, cat in enumerate(forecast):
        union_histogram += cat.magnitude_counts()

    mag_half_bin = numpy.diff(observed_catalog.region.magnitudes)[0] / 2.
    # end new
    n_union_events = numpy.sum(union_histogram)
    obs_histogram = observed_catalog.magnitude_counts()
    n_obs = numpy.sum(obs_histogram)
    union_scale = n_obs / n_union_events
    scaled_union_histogram = union_histogram * union_scale

    # this is new - prob to be used for resampling
    probs = union_histogram / n_union_events
    # end new

    # compute the test statistic for each catalog
    t0 = time.time()
    for i, catalog in enumerate(forecast):
        # THIS IS NEW - sampled from the union forecast histogram
        mag_values = numpy.random.choice(forecast.magnitudes + mag_half_bin, p=probs,
                                         size=int(n_obs))
        extended_mag_max = max(forecast.magnitudes) + 10
        mag_counts, tmp = numpy.histogram(mag_values, bins=numpy.append(forecast.magnitudes,
                                                                        extended_mag_max))
        # end new
        n_events = numpy.sum(mag_counts)
        if n_events == 0:
            # print("Skipping to next because catalog contained zero events.")
            continue
        scale = n_obs / n_events
        catalog_histogram = mag_counts * scale
        # compute magnitude test statistic for the catalog
        test_distribution.append(
            cumulative_square_diff(numpy.log10(catalog_histogram + 1),
                                   numpy.log10(scaled_union_histogram + 1))
        )
        # output status
        if verbose:
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)

    # compute observed statistic
    obs_d_statistic = cumulative_square_diff(numpy.log10(obs_histogram + 1),
                                             numpy.log10(scaled_union_histogram + 1))

    # score evaluation
    delta_1, delta_2 = get_quantiles(test_distribution, obs_d_statistic)

    # prepare result
    result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                                        name='M-Test',
                                        observed_statistic=obs_d_statistic,
                                        quantile=(delta_1, delta_2),
                                        status='normal',
                                        min_mw=forecast.min_magnitude,
                                        obs_catalog_repr=str(observed_catalog),
                                        obs_name=observed_catalog.name,
                                        sim_name=forecast.name)

    return result


def MLL_magnitude_test(forecast: "CatalogForecast",
                       observed_catalog: CSEPCatalog,
                       full_calculation: bool = False,
                       verbose: bool = False,
                       seed: Optional[int] = None) -> CatalogMagnitudeTestResult:
    """
    Implements the modified Multinomial log-likelihood ratio (MLL) magnitude test (Serafini et
    al., 2024). Calculates the test statistic distribution as:

        D̃_j = -2 * log( L(Λ_u + N_u / N_j + Λ̃_j + 1) /
                       [L(Λ_u + N_u / N_j) * L(Λ̃_j + 1)]
                       )

    where L is the multinomial likelihood function, Λ_u the union of all the forecasts'
    synthetic catalogs, N_u the total number of events in Λ_u, Λ̃_j the resampled catalog
    containing exactly N observed events. The observed statistic is defined as:

        D_o = -2 * log( L(Λ_u + N_u / N + Ω + 1) /
                       [L(Λ_u + N_u / N) * L(Ω + 1)]
                       )

    where Ω is the observed catalog.

    Args:
        forecast (CatalogForecast): The forecast to be evaluated
        observed_catalog (CSEPCatalog): The observation/testing catalog.
        full_calculation (bool): Whether to sample from the entire stochastic catalogs or from
            its already processed magnitude histogram.
        verbose (bool): Flag to display debug messages
        seed (int): Random number generator seed

    Returns:
        A CatalogMagnitudeTestResult object containing the statistic distribution and the
        observed statistic.
    """

    # set seed
    if seed:
        numpy.random.seed(seed)

    test_distribution = []

    if forecast.region.magnitudes is None:
        raise CSEPEvaluationException(
            "Forecast must have region.magnitudes member to perform magnitude test.")

    # short-circuit if zero events
    if observed_catalog.event_count == 0:
        print("Cannot perform magnitude test when observed event count is zero.")
        # prepare result
        result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                                            name='M-Test',
                                            observed_statistic=None,
                                            quantile=(None, None),
                                            status='not-valid',
                                            min_mw=forecast.min_magnitude,
                                            obs_catalog_repr=str(observed_catalog),
                                            obs_name=observed_catalog.name,
                                            sim_name=forecast.name)

        return result

    # compute expected rates for forecast if needed
    if forecast.expected_rates is None:
        forecast.get_expected_rates(verbose=verbose)

    # calculate histograms of union forecast and total number of events
    Lambda_u_histogram = numpy.zeros(len(forecast.magnitudes))

    if full_calculation:
        Lambda_u = []
    else:
        mag_half_bin = numpy.diff(observed_catalog.region.magnitudes)[0] / 2.

    for j, cat in enumerate(forecast):
        if full_calculation:
            Lambda_u = numpy.append(Lambda_u, cat.get_magnitudes())
        Lambda_u_histogram += cat.magnitude_counts()

    # # calculate histograms of observations and observed number of events
    Omega_histogram = observed_catalog.magnitude_counts()
    n_obs = numpy.sum(Omega_histogram)

    # compute observed statistic
    obs_d_statistic = MLL_score(union_catalog_counts=Lambda_u_histogram,
                                catalog_counts=Omega_histogram)

    probs = Lambda_u_histogram / numpy.sum(Lambda_u_histogram)

    # compute the test statistic for each catalog
    t0 = time.time()
    for i, catalog in enumerate(forecast):
        # this is new - sampled from the union forecast histogram
        if full_calculation:
            mag_values = numpy.random.choice(Lambda_u, size=int(n_obs))
        else:
            mag_values = numpy.random.choice(forecast.magnitudes + mag_half_bin, p=probs,
                                             size=int(n_obs))
        extended_mag_max = max(forecast.magnitudes) + 10
        Lambda_j_histogram, tmp = numpy.histogram(mag_values,
                                                  bins=numpy.append(forecast.magnitudes,
                                                                    extended_mag_max))

        # compute magnitude test statistic for the catalog
        test_distribution.append(
            MLL_score(union_catalog_counts=Lambda_u_histogram,
                      catalog_counts=Lambda_j_histogram)
        )
        # output status
        if verbose:
            tens_exp = numpy.floor(numpy.log10(i + 1))
            if (i + 1) % 10 ** tens_exp == 0:
                t1 = time.time()
                print(f'Processed {i + 1} catalogs in {t1 - t0} seconds', flush=True)

    # score evaluation
    delta_1, delta_2 = get_quantiles(test_distribution, obs_d_statistic)

    # prepare result
    result = CatalogMagnitudeTestResult(test_distribution=test_distribution,
                                        name='M-Test',
                                        observed_statistic=obs_d_statistic,
                                        quantile=(delta_1, delta_2),
                                        status='normal',
                                        min_mw=forecast.min_magnitude,
                                        obs_catalog_repr=str(observed_catalog),
                                        obs_name=observed_catalog.name,
                                        sim_name=forecast.name)

    return result