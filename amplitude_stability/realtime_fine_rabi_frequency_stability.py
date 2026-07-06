"""Track fine-Rabi frequency stability from repeated full sweeps in one experiment.

This intentionally uses a smaller pi-pulse amplitude than the calibrated value
and measures all repetition points. Each realtime run returns one full Rabi
oscillation trace; a per-run FFT estimates the dominant oscillation frequency
and amplitude.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from copy import copy
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated full fine-Rabi sweeps in one LabOneQ experiment and FFT each run."
        )
    )
    parser.add_argument("--qubit", default="q8", help="Qubit to measure, e.g. q8.")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=[0, 400, 1],
        help="Full fine-Rabi repetition range. Use step 1 for FFT.",
    )
    parser.add_argument(
        "--pi-amplitude",
        type=float,
        default=0.01,
        help="Pi-pulse amplitude override used to make oscillation drift visible.",
    )
    parser.add_argument(
        "--profile-name",
        default="main",
        help="Profile name used by task-manager submission.",
    )
    parser.add_argument(
        "--profile-branch",
        default="main",
        help="Profile branch to load before compiling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/amplitude_stability"),
        help="Directory for CSV and PNG outputs.",
    )
    parser.add_argument(
        "--emulate",
        action="store_true",
        help="Run through task-manager emulation instead of hardware.",
    )
    if argv is None and Path(sys.argv[0]).name == "ipykernel_launcher.py":
        argv = []
    return parser.parse_args(argv)


def validate_repetitions(repetitions: np.ndarray) -> None:
    if repetitions.size < 4:
        raise ValueError("--repetitions must produce at least four points for FFT.")
    steps = np.diff(repetitions)
    if not np.all(steps == steps[0]):
        raise ValueError("--repetitions must be evenly spaced for FFT.")


def reshape_result(raw: np.ndarray, runs: int, repetitions: np.ndarray) -> np.ndarray:
    raw_values = np.asarray(raw)
    expected_shape = (int(runs), int(repetitions.size))
    transposed_shape = (int(repetitions.size), int(runs))
    squeezed = np.abs(raw_values).squeeze()
    if squeezed.shape == expected_shape:
        return squeezed
    if squeezed.shape == transposed_shape:
        return squeezed.T

    values = squeezed.reshape(-1)
    expected_size = int(runs) * int(repetitions.size)
    if values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} values "
            f"({runs} runs x {repetitions.size} repetitions), got shape {raw_values.shape}."
        )
    return values.reshape(expected_shape)


def fft_spectrum(trace: np.ndarray, repetitions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spacing = float(repetitions[1] - repetitions[0])
    freqs = np.fft.rfftfreq(repetitions.size, d=spacing)
    centered = trace - np.mean(trace)
    spectrum = np.fft.rfft(centered)
    return freqs, np.abs(spectrum)


def fft_metrics(values: np.ndarray, repetitions: np.ndarray) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for run_index, trace in enumerate(values):
        freqs, magnitudes = fft_spectrum(trace, repetitions)
        if magnitudes.size <= 1:
            peak_index = 0
        else:
            peak_index = int(np.argmax(magnitudes[1:]) + 1)
        rows.append(
            {
                "run_index": int(run_index),
                "mean_signal": float(np.mean(trace)),
                "std_signal": float(np.std(trace)),
                "peak_frequency_cycles_per_repetition": float(freqs[peak_index]),
                "peak_magnitude": float(magnitudes[peak_index]),
                "dc_magnitude": float(magnitudes[0]),
            }
        )
    return rows


def fit_single_trace(
    trace: np.ndarray,
    repetitions: np.ndarray,
    initial_frequency: float,
) -> dict[str, float]:
    spacing = float(repetitions[1] - repetitions[0])
    resolution = 1.0 / (float(repetitions.size) * spacing)
    lower = max(resolution, initial_frequency - 2.0 * resolution)
    upper = min(0.5 / spacing, initial_frequency + 2.0 * resolution)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = resolution
        upper = 0.5 / spacing

    candidate_freqs = np.linspace(lower, upper, 81)
    best: dict[str, float] | None = None
    x = repetitions.astype(float)
    y = trace.astype(float)

    for frequency in candidate_freqs:
        angle = 2.0 * np.pi * frequency * x
        design = np.column_stack(
            [
                np.ones_like(x),
                np.cos(angle),
                np.sin(angle),
            ]
        )
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coeffs
        residual = y - fitted
        rss = float(np.sum(residual**2))
        if best is None or rss < best["fit_rss"]:
            offset, cos_coeff, sin_coeff = [float(value) for value in coeffs]
            contrast = float(np.hypot(cos_coeff, sin_coeff))
            phase = float(np.arctan2(-sin_coeff, cos_coeff))
            best = {
                "fit_frequency_cycles_per_repetition": float(frequency),
                "fit_contrast": contrast,
                "fit_offset": offset,
                "fit_phase_rad": phase,
                "fit_rss": rss,
                "fit_rmse": float(np.sqrt(rss / y.size)),
            }

    if best is None:
        raise RuntimeError("No fit candidates were evaluated.")
    return best


def fit_metrics(
    values: np.ndarray,
    repetitions: np.ndarray,
    fft_rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for run_index, trace in enumerate(values):
        fit = fit_single_trace(
            trace=trace,
            repetitions=repetitions,
            initial_frequency=float(
                fft_rows[run_index]["peak_frequency_cycles_per_repetition"]
            ),
        )
        rows.append({"run_index": int(run_index), **fit})
    return rows


def min_max_frequency_run_indices(
    fft_rows: list[dict[str, float | int]],
) -> tuple[int, int]:
    peak_freqs = np.asarray(
        [float(row["peak_frequency_cycles_per_repetition"]) for row in fft_rows]
    )
    return int(np.argmin(peak_freqs)), int(np.argmax(peak_freqs))


def save_trace_csv(
    path: Path,
    values: np.ndarray,
    repetitions: np.ndarray,
    qubit: str,
    timestamp: str,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "run_index", "qubit", "repetition", "value"],
        )
        writer.writeheader()
        for run_index, row in enumerate(values):
            for repetition, value in zip(repetitions, row):
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "run_index": int(run_index),
                        "qubit": qubit,
                        "repetition": int(repetition),
                        "value": float(value),
                    }
                )


def save_fft_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "mean_signal",
                "std_signal",
                "peak_frequency_cycles_per_repetition",
                "peak_magnitude",
                "dc_magnitude",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_fit_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "fit_frequency_cycles_per_repetition",
                "fit_contrast",
                "fit_offset",
                "fit_phase_rad",
                "fit_rss",
                "fit_rmse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    path: Path,
    values: np.ndarray,
    repetitions: np.ndarray,
    fft_rows: list[dict[str, float | int]],
    fit_rows: list[dict[str, float | int]],
    qubit: str,
) -> None:
    import matplotlib.pyplot as plt

    run_indices = np.arange(values.shape[0])
    peak_freqs = np.asarray(
        [float(row["peak_frequency_cycles_per_repetition"]) for row in fft_rows]
    )
    fit_freqs = np.asarray(
        [float(row["fit_frequency_cycles_per_repetition"]) for row in fit_rows]
    )

    min_freq_run, max_freq_run = min_max_frequency_run_indices(fft_rows)
    min_freqs, min_spectrum = fft_spectrum(values[min_freq_run], repetitions)
    max_freqs, max_spectrum = fft_spectrum(values[max_freq_run], repetitions)
    min_peak_freq = peak_freqs[min_freq_run]
    max_peak_freq = peak_freqs[max_freq_run]
    min_peak_index = int(np.argmin(np.abs(min_freqs - min_peak_freq)))
    max_peak_index = int(np.argmin(np.abs(max_freqs - max_peak_freq)))

    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.2, 1, 1, 1.3],
        width_ratios=[1, 0.035],
        hspace=0.1,
        wspace=0.1,
    )
    ax_heatmap = fig.add_subplot(grid[0, 0])
    ax_freq = fig.add_subplot(grid[1, 0], sharex=ax_heatmap)
    ax_fit = fig.add_subplot(grid[2, 0], sharex=ax_heatmap)
    ax_spectrum = fig.add_subplot(grid[3, 0])
    cbar_ax = fig.add_subplot(grid[0, 1])
    fig.add_subplot(grid[1, 1]).axis("off")
    fig.add_subplot(grid[2, 1]).axis("off")
    fig.add_subplot(grid[3, 1]).axis("off")

    mesh = ax_heatmap.pcolormesh(run_indices, repetitions, values.T, shading="auto")
    ax_heatmap.set_ylabel("Fine-Rabi repetition")
    ax_heatmap.set_title(f"Realtime full-sweep fine-Rabi frequency stability - {qubit}")
    fig.colorbar(mesh, cax=cbar_ax, label="Excitation / acquired signal")

    ax_freq.plot(run_indices, peak_freqs, marker="o", linewidth=1.2)
    ax_freq.set_ylabel("FFT peak freq\n[cycles/rep]")
    ax_freq.grid(True, alpha=0.25)

    ax_fit.plot(run_indices, fit_freqs, marker="s", linewidth=1.2)
    ax_fit.set_xlabel("Realtime run index")
    ax_fit.set_ylabel("Fit freq\n[cycles/rep]")
    ax_fit.grid(True, alpha=0.25)

    ax_spectrum.plot(
        min_freqs,
        min_spectrum,
        linewidth=1.2,
        label=f"min freq run {min_freq_run}: {min_peak_freq:.5g}",
    )
    ax_spectrum.plot(
        max_freqs,
        max_spectrum,
        linewidth=1.2,
        label=f"max freq run {max_freq_run}: {max_peak_freq:.5g}",
    )
    ax_spectrum.plot(
        min_freqs[min_peak_index],
        min_spectrum[min_peak_index],
        "o",
    )
    ax_spectrum.plot(
        max_freqs[max_peak_index],
        max_spectrum[max_peak_index],
        "s",
    )
    ax_spectrum.set_xlabel("Frequency [cycles/repetition]")
    ax_spectrum.set_ylabel("FFT magnitude")
    ax_spectrum.set_xlim(0.0, 0.2)
    ax_spectrum.grid(True, alpha=0.25)
    ax_spectrum.legend(fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    from laboneq.simple import AcquisitionType, AveragingMode, from_json
    from laboneq.dsl.parameter import SweepParameter

    from qratena.experiments.base_experiment import BaseExperiment
    from qratena.experiments.experiment_handler import ExperimentHandler
    from qratena.experiments.fine_rabi.fine_rabi_1d import (
        RotationType,
        SettingsFineRabi,
    )
    from qratena.system.components_params.profile import Profile
    from qratena.system.components_params.pulse_factory import PulseFactory
    from qratena.system.components_params.reset_settings import ResetSettings
    from qratena.util.enums import (
        SUPPORTED_PULSE_SHAPES,
        SUPPORTED_PULSE_TYPES,
        ExportationMethod,
        ResetType,
        UpdateParamsMethod,
    )

    from resources.load_profile import load_profile, load_task_manager

    repetitions = np.arange(*args.repetitions)
    validate_repetitions(repetitions)
    run_indices = np.arange(args.runs)

    class RealtimeFineRabiFrequencyStability(BaseExperiment):
        def __init__(
            self,
            qubit_names: list[str],
            repetitions: np.ndarray,
            run_indices: np.ndarray,
            settings: SettingsFineRabi,
            profile: Profile,
        ) -> None:
            super().__init__(
                experiment_name="realtime_fine_rabi_frequency_stability",
                qubit_names=qubit_names,
                settings=settings,
                configuration_params=profile,
            )
            self.repetitions = repetitions
            self.run_indices = run_indices
            self.settings = settings

        def define_experiment_sequence(self) -> None:
            run_sweep = SweepParameter(uid="run_sweep", values=self.run_indices)
            repetition_sweep = SweepParameter(
                uid="repetition_sweep",
                values=self.repetitions,
            )
            with self.acquire_loop_rt(
                uid="frequency_stability_shots",
                count=self.settings.num_shots,
                acquisition_type=self.settings.acquisition_type,
                averaging_mode=self.settings.averaging_mode,
            ):
                with self.sweep(uid="run_sweep", parameter=run_sweep):
                    with self.sweep(
                        uid="repetition_sweep",
                        parameter=repetition_sweep,
                        auto_chunking=self.settings.auto_chunking,
                    ):
                        self._reset_block()
                        self._excitation_block(repetition_sweep)
                        self.add_readout_primitive(
                            uid="readout_section",
                            play_after="qubit_excitation_section",
                        )

        def _reset_block(self) -> None:
            with self.section(uid="reset_section"):
                if self.settings.reset.reset_type == ResetType.ACTIVE:
                    self.add_active_reset(
                        uid="active_reset",
                        qubit_names=self.qubit_names,
                    )
                elif self.settings.reset.reset_type == ResetType.PASSIVE:
                    self.add_passive_reset(
                        uid="passive_reset",
                        qubit_names=self.qubit_names,
                        profile=self.configuration_params,
                    )

        def _excitation_block(self, repetition_sweep: SweepParameter) -> None:
            with self.section(
                uid="qubit_excitation_section",
                play_after="reset_section",
            ):
                with self.match(sweep_parameter=repetition_sweep):
                    for num in self.repetitions:
                        with self.case(num):
                            for _ in range(int(num)):
                                for qubit_name in self.qubit_names:
                                    self._pi_half_pulse(qubit_name)

        def _pi_half_pulse(self, qubit_name: str) -> None:
            pulse_params = self.configuration_params.get_pi_params(
                qubit=qubit_name,
                pulse_shape=self.settings.pulse_shape,
            )
            pulse = PulseFactory.create(SUPPORTED_PULSE_TYPES.pi, copy(pulse_params))
            pulse.section_play(
                section=self,
                signal=f"drive_{qubit_name}",
                uid=qubit_name,
                amplitude=0.5,
            )

    class RealtimeFineRabiFrequencyStabilityHandler(ExperimentHandler):
        def __init__(
            self,
            qubit_names: list[str],
            repetitions: np.ndarray,
            run_indices: np.ndarray,
            settings: SettingsFineRabi,
            profile: Profile,
        ) -> None:
            super().__init__(
                experiment_name="realtime_fine_rabi_frequency_stability",
                qubit_names=qubit_names,
                settings=settings,
                configuration_params=profile,
            )
            self.repetitions = repetitions
            self.run_indices = run_indices

        def define_experiment(self) -> RealtimeFineRabiFrequencyStability:
            experiment = RealtimeFineRabiFrequencyStability(
                qubit_names=self.qubit_names,
                repetitions=self.repetitions,
                run_indices=self.run_indices,
                settings=self.settings,
                profile=self.configuration_params,
            )
            experiment.define_experiment_sequence()
            return experiment

        def analyze(self) -> list:
            return []

        def plot(self) -> list:
            return []

        def update_system_params(self) -> None:
            return None

    profile = load_profile(args.profile_branch)
    pulse = profile.get_pi_params(args.qubit, pulse_shape=SUPPORTED_PULSE_SHAPES.const)
    pulse.pi_pulse_amplitude = args.pi_amplitude

    settings = SettingsFineRabi(
        do_emulation=True,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        exportation_method=ExportationMethod.NONE,
        update_params_method=UpdateParamsMethod.NONE,
        num_shots=args.shots,
        rotation_type=RotationType.PI_HALF,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        reset=ResetSettings(reset_type=ResetType.ACTIVE),
    )

    handler = RealtimeFineRabiFrequencyStabilityHandler(
        qubit_names=[args.qubit],
        repetitions=repetitions,
        run_indices=run_indices,
        settings=settings,
        profile=profile,
    )
    compiled_experiment = handler.get_compiled_experiment()

    task_manager = load_task_manager()
    task_id = task_manager.submit_compiled_experiment(
        experiment_name=handler.experiment_name,
        profile_name=args.profile_name,
        qubit_names=handler.qubit_names,
        compiled_experiment=compiled_experiment,
        do_emulation=args.emulate,
    )
    task_result = task_manager.wait_for_result(task_id)
    experiment_result = from_json(task_result.raw_data)
    values = reshape_result(
        experiment_result.get_data(f"handle_{args.qubit}"),
        args.runs,
        repetitions,
    )
    fft_rows = fft_metrics(values, repetitions)
    fit_rows = fit_metrics(values, repetitions, fft_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat(timespec="seconds")
    trace_csv_path = (
        args.output_dir / f"{args.qubit}_realtime_fine_rabi_frequency_traces_{stamp}.csv"
    )
    fft_csv_path = (
        args.output_dir / f"{args.qubit}_realtime_fine_rabi_frequency_fft_{stamp}.csv"
    )
    fit_csv_path = (
        args.output_dir / f"{args.qubit}_realtime_fine_rabi_frequency_fit_{stamp}.csv"
    )
    png_path = (
        args.output_dir / f"{args.qubit}_realtime_fine_rabi_frequency_{stamp}.png"
    )

    save_trace_csv(trace_csv_path, values, repetitions, args.qubit, timestamp)
    save_fft_csv(fft_csv_path, fft_rows)
    save_fit_csv(fit_csv_path, fit_rows)
    save_plot(png_path, values, repetitions, fft_rows, fit_rows, args.qubit)

    print(f"Saved trace CSV: {trace_csv_path}")
    print(f"Saved FFT CSV: {fft_csv_path}")
    print(f"Saved fit CSV: {fit_csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
