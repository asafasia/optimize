from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from laboneq.analysis import calculate_integration_kernels_thresholds
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES

from optimize.readout.utils.static_kernels import (
    constant_if_kernel_samples,
    create_static_readout_kernel_for_qubit,
    synthetic_traces_for_kernel,
)


def test_synthetic_traces_recalculate_requested_kernel() -> None:
    expected = constant_if_kernel_samples(
        intermediate_frequency=125e6,
        sampling_rate=2e9,
        num_samples=32,
        phase=0.2,
    )

    kernels, thresholds = calculate_integration_kernels_thresholds(
        synthetic_traces_for_kernel(expected)
    )

    assert len(kernels) == 1
    assert np.allclose(kernels[0].samples, expected)
    assert thresholds == pytest.approx([0.0])


def test_create_static_readout_kernel_for_qubit_writes_qratena_files(tmp_path) -> None:
    profile = _profile(readout_frequency=5.525e9, readout_lo_frequency=5.4e9)

    result = create_static_readout_kernel_for_qubit(
        profile,
        "q1",
        sampling_rate=2e9,
        num_samples=16,
        output_dir=tmp_path,
    )

    assert result.intermediate_frequency == pytest.approx(125e6)
    assert result.traces_path == tmp_path / "traces_q1.npy"
    assert result.kernels_path == tmp_path / "kernels_q1.pkl"
    assert result.traces_path.exists()
    assert result.kernels_path.exists()

    expected = constant_if_kernel_samples(
        intermediate_frequency=125e6,
        sampling_rate=2e9,
        num_samples=16,
    )
    traces = np.load(result.traces_path)
    calculated_kernels, _thresholds = calculate_integration_kernels_thresholds(traces)
    assert np.allclose(calculated_kernels[0].samples, expected)

    with result.kernels_path.open("rb") as f:
        pickled_kernels = pickle.load(f)
    assert np.allclose(pickled_kernels[0].samples, expected)


def test_create_static_readout_kernel_refuses_to_overwrite(tmp_path) -> None:
    profile = _profile()
    create_static_readout_kernel_for_qubit(profile, "q1", output_dir=tmp_path, num_samples=4)

    with pytest.raises(FileExistsError):
        create_static_readout_kernel_for_qubit(
            profile,
            "q1",
            output_dir=tmp_path,
            num_samples=4,
            overwrite=False,
        )


def _profile(
    *,
    readout_frequency: float = 5.5e9,
    readout_lo_frequency: float = 5.4e9,
    readout_duration: float = 500e-9,
) -> SimpleNamespace:
    return SimpleNamespace(
        qubits={
            "q1": SimpleNamespace(
                readout_resonator_frequency=SimpleNamespace(value=readout_frequency),
                readout_lo_frequency=readout_lo_frequency,
                pulses={
                    SUPPORTED_PULSE_TYPES.readout: {
                        SUPPORTED_PULSE_SHAPES.const: SimpleNamespace(
                            readout_duration=readout_duration
                        )
                    }
                },
            )
        }
    )
