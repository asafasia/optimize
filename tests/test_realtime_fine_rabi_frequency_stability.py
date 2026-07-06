from __future__ import annotations

import numpy as np
import pytest

from amplitude_stability import realtime_fine_rabi_frequency_stability as freq


def test_validate_repetitions_accepts_evenly_spaced_full_sweep():
    freq.validate_repetitions(np.arange(0, 10, 1))


def test_default_repetitions_use_400_points():
    args = freq.parse_args([])

    assert args.repetitions == [0, 400, 1]


def test_validate_repetitions_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least four"):
        freq.validate_repetitions(np.asarray([0, 1, 2]))


def test_fft_metrics_finds_known_frequency():
    repetitions = np.arange(0, 64, 1)
    cycles_per_rep = 4 / 64
    trace = 0.5 + 0.2 * np.cos(2 * np.pi * cycles_per_rep * repetitions)
    values = np.vstack([trace, trace])

    rows = freq.fft_metrics(values, repetitions)

    assert rows[0]["peak_frequency_cycles_per_repetition"] == pytest.approx(
        cycles_per_rep
    )
    assert rows[1]["peak_frequency_cycles_per_repetition"] == pytest.approx(
        cycles_per_rep
    )


def test_fit_single_trace_refines_frequency_between_fft_bins():
    repetitions = np.arange(0, 400, 1)
    cycles_per_rep = 0.037
    trace = 0.4 + 0.25 * np.cos(2 * np.pi * cycles_per_rep * repetitions + 0.3)

    fit = freq.fit_single_trace(
        trace=trace,
        repetitions=repetitions,
        initial_frequency=cycles_per_rep,
    )

    assert fit["fit_frequency_cycles_per_repetition"] == pytest.approx(
        cycles_per_rep,
        abs=0.001,
    )
    assert fit["fit_contrast"] == pytest.approx(0.25, abs=0.01)


def test_min_max_frequency_run_indices_selects_extremes():
    rows = [
        {"peak_frequency_cycles_per_repetition": 0.04},
        {"peak_frequency_cycles_per_repetition": 0.02},
        {"peak_frequency_cycles_per_repetition": 0.07},
    ]

    assert freq.min_max_frequency_run_indices(rows) == (1, 2)


def test_reshape_result_accepts_transposed_layout():
    repetitions = np.arange(0, 4, 1)
    raw = np.asarray(
        [
            [1 + 0j, 5 + 0j],
            [2 + 0j, 6 + 0j],
            [3 + 0j, 7 + 0j],
            [4 + 0j, 8 + 0j],
        ]
    )

    values = freq.reshape_result(raw, runs=2, repetitions=repetitions)

    np.testing.assert_array_equal(
        values,
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
    )
