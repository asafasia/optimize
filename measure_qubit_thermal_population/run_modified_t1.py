"""Run modified T1 measurements for thermal-population checks."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from laboneq.serializers import from_json
from laboneq.simple import AcquisitionType, AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    SUPPORTED_PULSE_SHAPES,
    ExportationMethod,
    ResetType,
    UpdateParamsMethod,
)

from measure_qubit_thermal_population.modified_t1 import ModifiedT1Handler
from resources.load_profile import load_profile, load_task_manager

PROFILE_NAME = "main"
QUBITS = [
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "q8",
    "q9",
    "q10",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q20",
]
INITIAL_STATES = ["e", "g"]
DECAY_TIME_SWEEP_INTERVAL_LENGTH = 200e-6
NUM_SWEEP_POINTS = 101
SECONDS_TO_MICROSECONDS = 1e6


def decay_model(
    x_time_data_points: np.ndarray,
    amplitude: float,
    t1: float,
    offset: float,
) -> np.ndarray:
    return amplitude * np.exp(-x_time_data_points / t1) + offset


def fit_decay_from_data(
    x_time_data_points: Iterable[float],
    y_values: Iterable[float],
) -> dict[str, float | np.ndarray]:
    x_full = np.asarray(x_time_data_points, dtype=float)
    y_full = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x_full) & np.isfinite(y_full)
    x = x_full[valid]
    y = y_full[valid]

    if x.size < 3 or np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return empty_decay_fit(x_full)

    span = float(np.ptp(y))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    eps = max(span * 1e-9, 1e-15)
    offset_candidates = np.linspace(y_min - 0.5 * span, y_max + 0.5 * span, 300)

    best: dict[str, float | np.ndarray] | None = None
    for offset in offset_candidates:
        for sign in (1.0, -1.0):
            shifted = sign * (y - offset)
            fit_mask = shifted > eps
            if np.count_nonzero(fit_mask) < 3 or np.ptp(x[fit_mask]) <= 0:
                continue

            slope, intercept = np.polyfit(x[fit_mask], np.log(shifted[fit_mask]), 1)
            if slope >= 0:
                continue

            t1 = float(-1.0 / slope)
            amplitude = float(sign * np.exp(intercept))
            if not np.isfinite(t1) or t1 <= 0 or not np.isfinite(amplitude):
                continue

            fitted = decay_model(x, amplitude, t1, float(offset))
            rss = float(np.sum((y - fitted) ** 2))
            if best is None or rss < float(best["rss"]):
                fitted_full = np.full_like(x_full, np.nan, dtype=float)
                fitted_full[valid] = fitted
                best = {
                    "amplitude": amplitude,
                    "t1": t1,
                    "offset": float(offset),
                    "fitted": fitted_full,
                    "rss": rss,
                }

    return best if best is not None else empty_decay_fit(x_full)


def empty_decay_fit(x_time_data_points: np.ndarray) -> dict[str, float | np.ndarray]:
    return {
        "amplitude": float("nan"),
        "t1": float("nan"),
        "offset": float("nan"),
        "fitted": np.full_like(x_time_data_points, np.nan, dtype=float),
        "rss": float("inf"),
    }


def fit_decay_from_qubit_data(qubit_data: dict) -> dict[str, float | np.ndarray]:
    x_values = np.asarray(qubit_data["x_time_data_points"], dtype=float)
    y_values = np.asarray(qubit_data["y_abs_amplitudes"], dtype=float)

    stored_offset = first_finite_value(
        qubit_data,
        ("fit_offset", "fitted_offset", "offset", "B", "fit_b", "fitted_b"),
    )
    stored_amplitude = first_finite_value(
        qubit_data,
        ("fit_amplitude", "fitted_amplitude", "amplitude", "A", "fit_a", "fitted_a"),
    )
    stored_t1 = first_finite_value(qubit_data, ("fitted_t1", "fit_t1", "t1", "T1"))

    if stored_offset is not None and stored_amplitude is not None and stored_t1:
        return {
            "amplitude": stored_amplitude,
            "t1": stored_t1,
            "offset": stored_offset,
            "fitted": decay_model(x_values, stored_amplitude, stored_t1, stored_offset),
            "rss": float("nan"),
        }

    if stored_t1 and stored_t1 > 0:
        envelope = np.exp(-x_values / stored_t1)
        design = np.column_stack((envelope, np.ones_like(envelope)))
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if np.count_nonzero(valid) >= 2:
            amplitude, offset = np.linalg.lstsq(design[valid], y_values[valid], rcond=None)[0]
            fitted = decay_model(x_values, float(amplitude), stored_t1, float(offset))
            return {
                "amplitude": float(amplitude),
                "t1": stored_t1,
                "offset": float(offset),
                "fitted": fitted,
                "rss": float(np.sum((y_values[valid] - fitted[valid]) ** 2)),
            }

    return fit_decay_from_data(x_values, y_values)


def first_finite_value(data: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in data:
            continue
        value = np.asarray(data[key]).reshape(-1)
        if value.size != 1:
            continue
        try:
            scalar = float(value[0])
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            return scalar
    return None


def plot_modified_t1_grid(handlers: list[ModifiedT1Handler]) -> plt.Figure:
    qubit_names = list(handlers[0].qubit_names)
    num_columns = min(5, len(qubit_names))
    num_rows = int(np.ceil(len(qubit_names) / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(3.8 * num_columns, 2.8 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    colors = {"e": "tab:blue", "g": "tab:orange"}

    for axis, qubit_name in zip(axes.ravel(), qubit_names, strict=False):
        for handler in handlers:
            qubit_data = handler.data[qubit_name]
            x_values = np.asarray(qubit_data["x_time_data_points"], dtype=float)
            y_values = np.asarray(qubit_data["y_abs_amplitudes"], dtype=float)
            fit = fit_decay_from_qubit_data(qubit_data)
            color = colors.get(handler.initial_state)
            axis.plot(
                x_values * SECONDS_TO_MICROSECONDS,
                y_values,
                marker="o",
                linestyle="",
                markersize=3,
                color=color,
                alpha=0.75,
                label=f"{handler.initial_state} data",
            )
            axis.plot(
                x_values * SECONDS_TO_MICROSECONDS,
                np.asarray(fit["fitted"], dtype=float),
                color=color,
                linewidth=1.8,
                label=f"{handler.initial_state} fit",
            )

        axis.set_title(qubit_name)
        axis.grid(True, alpha=0.3)
        axis.set_ylim(0, 1)

    for axis in axes.ravel()[len(qubit_names) :]:
        axis.set_visible(False)

    for axis in axes[-1, :]:
        axis.set_xlabel("Decay time (us)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Abs amplitude")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncols=4)
    figure.suptitle("Modified T1: initial e/g comparison by qubit", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_fit_offsets(handlers: list[ModifiedT1Handler]) -> plt.Figure:
    qubit_names = list(handlers[0].qubit_names)
    x_positions = np.arange(len(qubit_names))
    bar_width = 0.8 / max(len(handlers), 1)
    figure, axis = plt.subplots(figsize=(max(11.0, 0.6 * len(qubit_names)), 5.8))

    for handler_index, handler in enumerate(handlers):
        offsets = [
            float(fit_decay_from_qubit_data(handler.data[qubit_name])["offset"])
            for qubit_name in qubit_names
        ]
        axis.bar(
            x_positions + (handler_index - (len(handlers) - 1) / 2) * bar_width,
            offsets,
            width=bar_width,
            label=f"initial {handler.initial_state}",
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(qubit_names, rotation=45, ha="right")
    axis.set_ylabel("B offset in A*exp(-t/T1)+B")
    axis.set_title("Modified T1 fitted offset B by qubit")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


profile = load_profile(PROFILE_NAME)
profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)
task_manager = load_task_manager()

settings = ExperimentSettings(
    acquisition_type=AcquisitionType.DISCRIMINATION,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.NONE,
    exportation_method=ExportationMethod.FULL,
    pulse_shape=SUPPORTED_PULSE_SHAPES.const,
    num_shots=2500,
    reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=5),
)

handlers = []

for initial_state in INITIAL_STATES:
    handler = ModifiedT1Handler(
        qubit_names=QUBITS,
        initial_state=initial_state,
        decay_time_sweep_interval_length=DECAY_TIME_SWEEP_INTERVAL_LENGTH,
        num_sweep_points=NUM_SWEEP_POINTS,
        settings=settings,
        configuration_params=profile,
    )

    compiled_experiment = handler.get_compiled_experiment()

    task_id = task_manager.submit_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name=PROFILE_NAME,
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=False,
    )
    task_result = task_manager.wait_for_result(task_id)

    handler.experiment_result = from_json(task_result.raw_data)
    handler.analysis_result = handler.analyze()
    handler.figs = handler.plot()
    handler.export_data(figs=handler.figs)
    handlers.append(handler)
    
    
#%%

plot_modified_t1_grid(handlers)
plot_fit_offsets(handlers)
plt.show()

# %%
