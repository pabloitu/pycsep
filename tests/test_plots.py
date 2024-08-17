import json
import os.path
import random
import string
import unittest
from unittest import mock

import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy

import csep
from csep import EvaluationResult, CatalogNumberTestResult, CatalogSpatialTestResult, \
    CatalogMagnitudeTestResult, CatalogPseudolikelihoodTestResult
from csep.core import catalogs
from csep.utils import plots
from csep.utils.plots import (
    plot_cumulative_events_versus_time,
    plot_magnitude_vs_time,
    plot_dist_test, plot_histogram, plot_number_test, plot_magnitude_test, plot_spatial_test,
    plot_likelihood_test
)


class TestPoissonPlots(unittest.TestCase):

    def test_SingleNTestPlot(self):

        expected_val = numpy.random.randint(0, 20)
        observed_val = numpy.random.randint(0, 20)
        Ntest_result = mock.Mock()
        Ntest_result.name = "Mock NTest"
        Ntest_result.sim_name = "Mock SimName"
        Ntest_result.test_distribution = ["poisson", expected_val]
        Ntest_result.observed_statistic = observed_val
        matplotlib.pyplot.close()
        ax = plots.plot_consistency_test(Ntest_result)

        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in [Ntest_result]],
        )
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Ntest_result.name)

    def test_MultiNTestPlot(self, show=False):

        n_plots = numpy.random.randint(1, 20)
        Ntests = []
        for n in range(n_plots):
            Ntest_result = mock.Mock()
            Ntest_result.name = "Mock NTest"
            Ntest_result.sim_name = "".join(
                random.choice(string.ascii_letters) for _ in range(8)
            )
            Ntest_result.test_distribution = ["poisson", numpy.random.randint(0, 20)]
            Ntest_result.observed_statistic = numpy.random.randint(0, 20)
            Ntests.append(Ntest_result)
        matplotlib.pyplot.close()
        ax = plots.plot_consistency_test(Ntests)
        Ntests.reverse()

        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Ntests],
        )
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Ntests[0].name)
        if show:
            matplotlib.pyplot.show()

    def test_MultiSTestPlot(self):

        s_plots = numpy.random.randint(1, 20)
        Stests = []
        for n in range(s_plots):
            Stest_result = mock.Mock()  # Mock class with random attributes
            Stest_result.name = "Mock STest"
            Stest_result.sim_name = "".join(
                random.choice(string.ascii_letters) for _ in range(8)
            )
            Stest_result.test_distribution = numpy.random.uniform(
                -1000, 0, numpy.random.randint(3, 500)
            ).tolist()
            Stest_result.observed_statistic = numpy.random.uniform(
                -1000, 0
            )  # random observed statistic
            if numpy.random.random() < 0.02:  # sim possible infinite values
                Stest_result.observed_statistic = -numpy.inf
            Stests.append(Stest_result)
        matplotlib.pyplot.close()
        plots.plot_consistency_test(Stests)
        Stests.reverse()
        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Stests],
        )
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Stests[0].name)

    def test_MultiTTestPlot(self):

        for i in range(10):
            t_plots = numpy.random.randint(2, 20)
            t_tests = []

            def rand(limit=10, offset=0):
                return limit * (numpy.random.random() - offset)

            for n in range(t_plots):
                t_result = mock.Mock()  # Mock class with random attributes
                t_result.name = "CSEP1 Comparison Test"
                t_result.sim_name = (
                    "".join(random.choice(string.ascii_letters) for _ in range(8)),
                    "ref",
                )
                t_result.observed_statistic = rand(offset=0.5)
                t_result.test_distribution = [
                    t_result.observed_statistic - rand(5),
                    t_result.observed_statistic + rand(5),
                ]

                if numpy.random.random() < 0.05:  # sim possible infinite values
                    t_result.observed_statistic = -numpy.inf
                t_tests.append(t_result)
            matplotlib.pyplot.close()
            plots.plot_comparison_test(t_tests)
            t_tests.reverse()
            self.assertEqual(
                [i.get_text() for i in matplotlib.pyplot.gca().get_xticklabels()],
                [i.sim_name[0] for i in t_tests[::-1]],
            )
            self.assertEqual(matplotlib.pyplot.gca().get_title(), t_tests[0].name)


class TestTimeSeriesPlots(unittest.TestCase):

    def setUp(self):
        # This method is called before each test.
        # Load the stochastic event sets and observation here.

        cat_file = os.path.join(
            os.path.dirname(__file__),
            "artifacts",
            "example_csep2_forecasts",
            "Catalog",
            "catalog.json",
        )
        forecast_file = os.path.join(
            os.path.dirname(__file__),
            "artifacts",
            "example_csep2_forecasts",
            "Forecasts",
            "ucerf3-landers_short.csv",
        )

        self.stochastic_event_sets = csep.load_catalog_forecast(forecast_file)
        self.observation = catalogs.CSEPCatalog.load_json(cat_file)
        self.save_dir = os.path.join(os.path.dirname(__file__), "artifacts", "plots")

    def savefig(self, ax, name):
        ax.figure.savefig(os.path.join(self.save_dir, name))

    def test_plot_cumulative_events(self):
        # Call the plotting function
        ax = plot_cumulative_events_versus_time(
            catalog_forecast=self.stochastic_event_sets,
            observation=self.observation,
            figsize=(10, 6),
            xlabel="Days since Mainshock",
            ylabel="Cumulative Event Count",
            title="Cumulative Event Counts",
            legend_loc="best",
            show=False,  # Set to False to prevent displaying the plot during tests
        )

        # Save the plot for manual review
        self.savefig(ax, "cumulative_events_test_plot.png")

        # Assertions to check if the plot has been correctly set up
        self.assertEqual(ax.get_title(), "Cumulative Event Counts")
        self.assertEqual(ax.get_xlabel(), "Days since Mainshock")
        self.assertEqual(ax.get_ylabel(), "Cumulative Event Count")

    def test_plot_magnitude_vs_time(self):
        # Call the plotting function
        ax = plot_magnitude_vs_time(catalog=self.observation, grid=True, show=False)

        # Save the plot for manual review
        self.savefig(ax, "plot_magnitude_vs_time.png")

        # Assertions to check if the plot has been correctly set up
        self.assertEqual(ax.get_title(), "Magnitude vs. Time")
        self.assertEqual(ax.get_xlabel(), "Datetime")
        self.assertEqual(ax.get_ylabel(), "$M$")

    def tearDown(self):
        # This method is called after each test.
        # Clean up any resources if necessary.
        pass


class TestPlotHistogram(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        self.simulated = numpy.random.normal(0, 1, 1000)
        self.observation_array = numpy.random.normal(0, 1, 100)
        self.observation_scalar = 0.5

        # Real data for testing
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "artifacts",
                "example_csep2_forecasts",
                "Results",
                "catalog_n_test.json",
            )
        ) as json_file:
            result = EvaluationResult.from_dict(json.load(json_file))
        self.real_result = result
        self.save_dir = os.path.join(os.path.dirname(__file__), "artifacts", "plots")

    def savefig(self, ax, name):
        ax.figure.savefig(os.path.join(self.save_dir, name))

    def test_plot_dist_test_with_array_observation(self):
        ax = plot_dist_test(
            simulated=self.simulated,
            observation=self.observation_array,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            alpha=0.5,
            show=False,
        )

        # Check if labels and title are set correctly
        self.assertEqual(ax.get_xlabel(), "Test X Label")
        self.assertEqual(ax.get_ylabel(), "Test Y Label")
        self.assertEqual(ax.get_title(), "Test Title")

        self.savefig(ax, "plot_dist_test_obs_array.png")

        # Check if the histogram was created with the correct alpha
        bars = ax.patches
        self.assertTrue(
            all(bar.get_alpha() == 0.5 for bar in bars),
            "Alpha transparency not set correctly for bars",
        )

    def test_plot_dist_test_with_scalar_observation(self):
        ax = plot_dist_test(
            simulated=self.simulated,
            observation=self.observation_scalar,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            show=False,
        )
        self.savefig(ax, "plot_dist_test_obs_scalar.png")

        # Check if a vertical line was drawn for the scalar observation
        lines = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
        self.assertEqual(len(lines), 1)  # Expect one vertical line
        self.assertEqual(lines[0].get_xdata()[0], self.observation_scalar)

    def test_plot_dist_test_with_percentile_shading(self):
        ax = plot_dist_test(
            simulated=self.simulated,
            observation=self.observation_scalar,
            percentile=95,
            show=False,
        )
        expected_red = (1.0, 0.0, 0.0)
        tolerance = 0.01  # Tolerance for comparing color values
        # Check if patches were colored red for the percentile shading
        red_patches = []
        for patch in ax.patches:
            facecolor = patch.get_facecolor()[:3]  # Get RGB, ignore alpha
            if all(abs(facecolor[i] - expected_red[i]) < tolerance for i in range(3)):
                red_patches.append(patch)
        self.savefig(ax, "plot_dist_test_percentile_shading.png")
        self.assertGreater(
            len(red_patches),
            0,
            "Expected some patches to be colored red for percentile shading",
        )

    def test_plot_dist_test_with_annotation(self):
        annotation_text = "Test Annotation"
        ax = plot_dist_test(
            simulated=self.simulated,
            observation=self.observation_array,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            annotate=True,
            annotation_text=annotation_text,
            show=False,
        )
        self.savefig(ax, "plot_dist_test_w_annotation.png")

        # Check if annotation was added to the plot
        annotations = ax.texts
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].get_text(), annotation_text)

    def test_plot_dist_test_xlim_nan(self):
        xlim = (-3, 3)
        ax = plot_dist_test(
            simulated=self.simulated,
            observation=self.observation_array,
            percentile=95,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            xlim=xlim,
            show=False,
        )
        self.savefig(ax, "plot_dist_test_xlims.png")
        self.assertEqual(ax.get_xlim(), xlim)

    def test_plot_dist_test_autoxlim_nan(self):
        ax = plot_dist_test(
            simulated=self.simulated,
            observation=-numpy.nan,
            percentile=95,
            xlabel="Test X Label",
            ylabel="Test Y Label",
            title="Test Title",
            show=False,
        )
        self.savefig(ax, "plot_dist_test_xlims_obs_nan.png")

    def test_plot_dist_test_real_dist(self):
        simulated = self.real_result.test_distribution
        observed = self.real_result.observed_statistic
        ax = plot_dist_test(
            simulated=simulated,
            observation=observed,
            percentile=95,
            show=False,
        )
        self.savefig(ax, "plot_dist_test_catalog_n_test.png")

    def tearDown(self):
        plt.close("all")


class TestPlotHistogram_old(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        n_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_n_test.json")
        s_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_s_test.json")
        m_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_m_test.json")
        l_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_l_test.json")

        with open(n_test, 'r') as fp:
            self.n_test = EvaluationResult.from_dict(json.load(fp))
        with open(s_test, 'r') as fp:
            self.s_test = EvaluationResult.from_dict(json.load(fp))
        with open(m_test, 'r') as fp:
            self.m_test = EvaluationResult.from_dict(json.load(fp))
        with open(l_test, 'r') as fp:
            self.l_test = EvaluationResult.from_dict(json.load(fp))

        self.save_dir = os.path.join(os.path.dirname(__file__), "artifacts", "plots")

    def savefig(self, ax, name):
        ax.figure.savefig(os.path.join(self.save_dir, name))

    def test_plot_n_test(self):
        ax = plot_number_test(
            self.n_test,
            show=False,
        )
        self.savefig(ax, "plot_n_test_old.png")

    def test_plot_m_test(self):
        ax = plot_magnitude_test(
            self.m_test,
            show=False,
        )
        self.savefig(ax, "plot_m_test_old.png")

    def test_plot_s_test(self):
        ax = plot_spatial_test(
            self.s_test,
            show=False,
        )
        self.savefig(ax, "plot_s_test_old.png")

    def test_plot_l_test(self):
        ax = plot_likelihood_test(
            self.l_test,
            show=False,
        )
        self.savefig(ax, "plot_l_test_old.png")

    def tearDown(self):
        plt.close("all")


class TestPlotDistributionTest(unittest.TestCase):

    def setUp(self):
        # Example data for testing
        n_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_n_test.json")
        s_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_s_test.json")
        m_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_m_test.json")
        l_test = os.path.join(os.path.dirname(__file__), "artifacts",
                              "example_csep2_forecasts", "Results" ,
                              "catalog_l_test.json")

        with open(n_test, 'r') as fp:
            self.n_test = CatalogNumberTestResult.from_dict(json.load(fp))
        with open(s_test, 'r') as fp:
            self.s_test = CatalogSpatialTestResult.from_dict(json.load(fp))
        with open(m_test, 'r') as fp:
            self.m_test = CatalogMagnitudeTestResult.from_dict(json.load(fp))
        with open(l_test, 'r') as fp:
            self.l_test = CatalogPseudolikelihoodTestResult.from_dict(json.load(fp))

        self.save_dir = os.path.join(os.path.dirname(__file__), "artifacts", "plots")

    def savefig(self, ax, name):
        ax.figure.savefig(os.path.join(self.save_dir, name))

    def test_plot_n_test(self):
        ax = plot_dist_test(
            self.n_test,
            show=False,
        )
        self.savefig(ax, "plot_n_test.png")

    def test_plot_m_test(self):
        ax = plot_dist_test(
            self.m_test,
            show=False,
        )
        self.savefig(ax, "plot_m_test.png")

    def test_plot_s_test(self):
        ax = plot_dist_test(
            self.s_test,
            show=False,
        )
        self.savefig(ax, "plot_s_test.png")

    def test_plot_l_test(self):
        ax = plot_dist_test(
            self.l_test,
            show=False,
        )
        self.savefig(ax, "plot_l_test.png")

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
