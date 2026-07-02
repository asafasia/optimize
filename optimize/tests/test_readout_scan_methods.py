from types import SimpleNamespace

import pytest

from optimize.readout.utils.readout_scan_methods import (
    ReadoutGoldenSectionScan,
    ReadoutGradientAscentScan,
    ReadoutSweepScan,
    ReadoutZoomInScan,
    scan_method_for,
)
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod


class FakeOptimizer:
    def __init__(self, amplitudes, scores=None, **settings):
        defaults = {
            "amplitudes": amplitudes,
            "method": ReadoutScanMethod.SWEEP,
            "zoom_in_iterations": 2,
            "zoom_in_shrink_factor": 0.5,
            "gradient_max_iterations": 5,
            "gradient_initial_step": None,
            "gradient_min_step": 0.001,
            "gradient_fidelity_tolerance": 0.0,
            "golden_section_max_iterations": 5,
            "golden_section_interval_tolerance": 0.001,
        }
        defaults.update(settings)
        self.settings = SimpleNamespace(**defaults)
        self.scores = scores or {}
        self.measured_amplitudes = []
        self.results = {}
        self.progress = []

    def _measure_amplitude(self, amplitude):
        amplitude = float(amplitude)
        if amplitude not in self.results:
            score = float(self.scores.get(amplitude, 1.0 - abs(amplitude - 0.5)))
            self.results[amplitude] = {"score": score}
            self.measured_amplitudes.append(amplitude)
        return self.results[amplitude]["score"]

    def _score_result(self, result):
        return result["score"]

    def _show_progress(self, index, total, amplitude):
        self.progress.append((index, total, float(amplitude)))


def test_scan_method_for_returns_requested_strategy():
    expected = {
        ReadoutScanMethod.SWEEP: ReadoutSweepScan,
        ReadoutScanMethod.ZOOM_IN: ReadoutZoomInScan,
        ReadoutScanMethod.GRADIENT: ReadoutGradientAscentScan,
        ReadoutScanMethod.GOLDEN_SECTION: ReadoutGoldenSectionScan,
    }

    for method, strategy_type in expected.items():
        optimizer = FakeOptimizer([0.1, 0.2], method=method)
        assert isinstance(scan_method_for(optimizer), strategy_type)


def test_sweep_measures_each_configured_amplitude_in_order():
    optimizer = FakeOptimizer([0.3, 0.1, 0.2])

    ReadoutSweepScan(optimizer).run()

    assert optimizer.measured_amplitudes == [0.3, 0.1, 0.2]


def test_zoom_in_adds_points_around_best_amplitude():
    optimizer = FakeOptimizer(
        [0.0, 0.5, 1.0],
        scores={0.0: 0.2, 0.5: 0.9, 1.0: 0.3},
        zoom_in_iterations=2,
        zoom_in_shrink_factor=0.5,
    )

    ReadoutZoomInScan(optimizer).run()

    assert optimizer.measured_amplitudes == [0.0, 0.5, 1.0, 0.25, 0.75]


@pytest.mark.parametrize(
    ("amplitudes", "settings", "message"),
    [
        ([], {}, "at least one amplitude"),
        ([-0.1, 0.1], {}, "greater than or equal to 0"),
        ([0.1, 0.2], {"zoom_in_iterations": 0}, "at least 1"),
        ([0.1, 0.2], {"zoom_in_shrink_factor": 1.0}, "between 0 and 1"),
    ],
)
def test_zoom_in_validates_settings(amplitudes, settings, message):
    optimizer = FakeOptimizer(amplitudes, **settings)

    with pytest.raises(ValueError, match=message):
        ReadoutZoomInScan(optimizer).run()


def test_gradient_respects_measurement_budget():
    optimizer = FakeOptimizer(
        [0.0, 0.25, 0.5, 0.75],
        gradient_max_iterations=10,
        gradient_initial_step=0.25,
    )

    ReadoutGradientAscentScan(optimizer).run()

    assert len(optimizer.measured_amplitudes) <= 4
    assert len(optimizer.measured_amplitudes) == len(set(optimizer.measured_amplitudes))


def test_golden_section_respects_bounds_and_measurement_budget():
    optimizer = FakeOptimizer(
        [0.1, 0.2, 0.3, 0.4, 0.5],
        golden_section_max_iterations=8,
    )

    ReadoutGoldenSectionScan(optimizer).run()

    assert len(optimizer.measured_amplitudes) <= 5
    assert min(optimizer.measured_amplitudes) >= 0.1
    assert max(optimizer.measured_amplitudes) <= 0.5
