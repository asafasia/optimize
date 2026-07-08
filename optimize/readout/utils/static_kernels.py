from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from laboneq.analysis import calculate_integration_kernels_thresholds
from qratena.util import settings
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES
from qratena.util.kernels import pickle_kernels

DEFAULT_KERNEL_SAMPLING_RATE = 2e9


@dataclass(frozen=True, slots=True)
class StaticReadoutKernelResult:
    qubit_name: str
    intermediate_frequency: float
    readout_duration: float
    sampling_rate: float
    num_samples: int
    traces_path: Path
    kernels_path: Path


def create_static_readout_kernel_for_qubit(
    profile: Any,
    qubit_name: str,
    *,
    intermediate_frequency: float | None = None,
    readout_duration: float | None = None,
    sampling_rate: float = DEFAULT_KERNEL_SAMPLING_RATE,
    num_samples: int | None = None,
    phase: float = 0.0,
    amplitude: float = 1.0,
    output_dir: str | Path | None = None,
    overwrite: bool = True,
) -> StaticReadoutKernelResult:
    """Create constant IF-demodulating readout kernel files for one qubit.

    qratena readout code usually reads ``traces_<qubit>.npy`` and lets LabOneQ
    calculate integration kernels at compile time. Some paths can read
    ``kernels_<qubit>.pkl`` directly. This helper writes both files in the same
    directory used by the profile/kernel-traces workflow.
    """

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    if amplitude == 0:
        raise ValueError("amplitude must be non-zero.")

    qubit = profile.qubits[qubit_name]
    resolved_if = (
        float(intermediate_frequency)
        if intermediate_frequency is not None
        else _readout_intermediate_frequency(qubit)
    )
    resolved_duration = (
        float(readout_duration)
        if readout_duration is not None
        else _readout_duration(qubit)
    )
    if resolved_duration <= 0:
        raise ValueError("readout_duration must be positive.")

    resolved_num_samples = (
        int(num_samples)
        if num_samples is not None
        else int(round(resolved_duration * sampling_rate))
    )
    if resolved_num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    target_dir = Path(output_dir) if output_dir is not None else settings.KERNEL_TRACES_DIR_PATH
    target_dir.mkdir(parents=True, exist_ok=True)
    traces_path = target_dir / f"traces_{qubit_name}.npy"
    kernels_path = target_dir / f"kernels_{qubit_name}.pkl"
    if not overwrite and (traces_path.exists() or kernels_path.exists()):
        raise FileExistsError(
            f"Kernel files already exist for {qubit_name} in {target_dir}."
        )

    kernel_samples = constant_if_kernel_samples(
        intermediate_frequency=resolved_if,
        sampling_rate=sampling_rate,
        num_samples=resolved_num_samples,
        phase=phase,
        amplitude=amplitude,
    )

    traces = synthetic_traces_for_kernel(kernel_samples)
    np.save(traces_path, traces)

    calculated_kernels, _thresholds = calculate_integration_kernels_thresholds(traces)
    pickle_kernels(qubit_traces=calculated_kernels, export_path=kernels_path)

    return StaticReadoutKernelResult(
        qubit_name=qubit_name,
        intermediate_frequency=resolved_if,
        readout_duration=resolved_duration,
        sampling_rate=sampling_rate,
        num_samples=resolved_num_samples,
        traces_path=traces_path,
        kernels_path=kernels_path,
    )


def constant_if_kernel_samples(
    *,
    intermediate_frequency: float,
    sampling_rate: float,
    num_samples: int,
    phase: float = 0.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Return complex weights for constant demodulation at readout IF."""

    sample_times = np.arange(num_samples, dtype=float) / float(sampling_rate)
    return amplitude * np.exp(1j * phase - 2j * np.pi * intermediate_frequency * sample_times)


def synthetic_traces_for_kernel(kernel_samples: np.ndarray) -> np.ndarray:
    """Build two synthetic state traces whose LabOneQ kernel is ``kernel_samples``."""

    trace = np.asarray(kernel_samples, dtype=np.complex128)
    return np.asarray([0.5 * np.conj(trace), -0.5 * np.conj(trace)])


def _readout_intermediate_frequency(qubit: Any) -> float:
    readout_frequency = getattr(qubit.readout_resonator_frequency, "value", None)
    readout_lo_frequency = getattr(qubit, "readout_lo_frequency", None)
    if readout_frequency is None or readout_lo_frequency is None:
        raise ValueError(
            "Could not infer readout IF. Pass intermediate_frequency explicitly."
        )
    return float(readout_frequency) - float(readout_lo_frequency)


def _readout_duration(qubit: Any) -> float:
    readout_pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][SUPPORTED_PULSE_SHAPES.const]
    return float(readout_pulse.readout_duration)
