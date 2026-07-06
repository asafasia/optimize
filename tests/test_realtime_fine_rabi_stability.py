from __future__ import annotations

import numpy as np
import pytest

from amplitude_stability import realtime_fine_rabi_stability as realtime


def test_validate_odd_repetitions_accepts_superposition_points():
    realtime.validate_odd_repetitions(np.asarray([1, 3, 5, 7]))


def test_validate_odd_repetitions_rejects_even_points():
    with pytest.raises(ValueError, match="only odd values"):
        realtime.validate_odd_repetitions(np.asarray([1, 2, 3]))


def test_reshape_result_returns_run_by_repetition_matrix():
    repetitions = np.asarray([1, 3, 5])
    raw = np.asarray([1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j, 5 + 0j, 6 + 0j])

    values = realtime.reshape_result(raw, runs=2, repetitions=repetitions)

    np.testing.assert_array_equal(
        values,
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )


def test_reshape_result_accepts_transposed_laboneq_shape():
    repetitions = np.asarray([1, 3, 5])
    raw = np.asarray(
        [
            [1 + 0j, 4 + 0j],
            [2 + 0j, 5 + 0j],
            [3 + 0j, 6 + 0j],
        ]
    )

    values = realtime.reshape_result(raw, runs=2, repetitions=repetitions)

    np.testing.assert_array_equal(
        values,
        np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )
