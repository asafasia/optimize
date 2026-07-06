"""Run repeated odd-point fine-Rabi stability inside one LabOneQ experiment.

This differs from ``fixed_amplitude_stability.py``: that script submits the
same compiled 1D fine-Rabi experiment many times from Python. This script puts
the repeat index inside the LabOneQ realtime experiment, so one submission
returns a 2D result: run index by fine-Rabi repetition.
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
            "Run repeated odd-point fine Rabi inside one LabOneQ realtime experiment."
        )
    )
    parser.add_argument("--qubit", default="q8", help="Qubit to measure, e.g. q8.")
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of repeated fine-Rabi traces inside the same LabOneQ experiment.",
    )
    parser.add_argument("--shots", type=int, default=500, help="Shots per point.")
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=[1, 100, 2],
        help="Fine-Rabi repetition range. Default gives 1, 3, 5, ...",
    )
    parser.add_argument(
        "--pi-amplitude",
        type=float,
        default=None,
        help="Optional pi-pulse amplitude override before compiling.",
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


def validate_odd_repetitions(repetitions: np.ndarray) -> None:
    if repetitions.size == 0:
        raise ValueError("--repetitions produced an empty array.")
    if np.any(repetitions % 2 == 0):
        raise ValueError(
            "--repetitions must contain only odd values for superposition points."
        )


def reshape_result(raw: np.ndarray, runs: int, repetitions: np.ndarray) -> np.ndarray:
    raw_values = np.abs(np.asarray(raw)).squeeze()
    expected_shape = (int(runs), int(repetitions.size))
    transposed_shape = (int(repetitions.size), int(runs))
    if raw_values.shape == expected_shape:
        return raw_values
    if raw_values.shape == transposed_shape:
        return raw_values.T

    values = raw_values.reshape(-1)
    expected_size = int(runs) * int(repetitions.size)
    if values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} acquired values "
            f"({runs} runs x {repetitions.size} repetitions), got shape {np.asarray(raw).shape}."
        )
    return values.reshape(expected_shape)


def save_csv(
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


def save_metrics_csv(path: Path, values: np.ndarray) -> None:
    std_values = np.std(values, axis=1)
    jump_scores = np.full(values.shape[0], np.nan, dtype=float)
    if values.shape[0] > 1:
        jump_scores[1:] = np.mean(np.abs(np.diff(values, axis=0)), axis=1)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "selected_point_std",
                "selected_point_variance",
                "jump_score",
            ],
        )
        writer.writeheader()
        for run_index, std_value, jump_score in zip(
            range(values.shape[0]), std_values, jump_scores
        ):
            writer.writerow(
                {
                    "run_index": int(run_index),
                    "selected_point_std": float(std_value),
                    "selected_point_variance": float(std_value**2),
                    "jump_score": float(jump_score),
                }
            )


def save_plot(path: Path, values: np.ndarray, repetitions: np.ndarray, qubit: str) -> None:
    import matplotlib.pyplot as plt

    run_indices = np.arange(values.shape[0])
    std_values = np.std(values, axis=1)
    jump_scores = np.full(values.shape[0], np.nan, dtype=float)
    if values.shape[0] > 1:
        jump_scores[1:] = np.mean(np.abs(np.diff(values, axis=0)), axis=1)

    fig = plt.figure(figsize=(9, 7))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.2, 1],
        width_ratios=[1, 0.035],
        hspace=0.08,
        wspace=0.1,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0], sharex=ax_top)
    cbar_ax = fig.add_subplot(grid[0, 1])
    spacer_ax = fig.add_subplot(grid[1, 1])
    spacer_ax.axis("off")

    mesh = ax_top.pcolormesh(run_indices, repetitions, values.T, shading="auto")
    ax_top.set_ylabel("Fine-Rabi repetition")
    ax_top.set_title(f"Realtime odd-point fine-Rabi stability - {qubit}")
    fig.colorbar(mesh, cax=cbar_ax, label="Acquired signal")

    ax_bottom.plot(
        run_indices,
        std_values,
        marker="o",
        linewidth=1.2,
        label="std across selected points",
    )
    ax_bottom.plot(
        run_indices,
        jump_scores,
        marker="s",
        linewidth=1.2,
        label="run-to-run jump score",
    )
    ax_bottom.set_xlabel("Realtime run index")
    ax_bottom.set_ylabel("|output| instability")
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend(fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    from laboneq.simple import AcquisitionType, AveragingMode, from_json
    from laboneq.dsl.parameter import SweepParameter

    from qratena.experiments.base_experiment import BaseExperiment
    from qratena.experiments.fine_rabi.fine_rabi_1d import (
        RotationType,
        SettingsFineRabi,
    )
    from qratena.experiments.experiment_handler import ExperimentHandler
    from qratena.system.components_params.pulse_factory import PulseFactory
    from qratena.system.components_params.profile import Profile
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
    validate_odd_repetitions(repetitions)
    run_indices = np.arange(args.runs)

    class RealtimeFineRabiStability(BaseExperiment):
        def __init__(
            self,
            qubit_names: list[str],
            repetitions: np.ndarray,
            run_indices: np.ndarray,
            settings: SettingsFineRabi,
            profile: Profile,
        ) -> None:
            super().__init__(
                experiment_name="realtime_fine_rabi_stability",
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
                uid="stability_shots",
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
            pulse = PulseFactory.create(
                SUPPORTED_PULSE_TYPES.pi,
                copy(pulse_params),
            )
            pulse.section_play(
                section=self,
                signal=f"drive_{qubit_name}",
                uid=qubit_name,
                amplitude=0.5,
            )

    class RealtimeFineRabiStabilityHandler(ExperimentHandler):
        def __init__(
            self,
            qubit_names: list[str],
            repetitions: np.ndarray,
            run_indices: np.ndarray,
            settings: SettingsFineRabi,
            profile: Profile,
        ) -> None:
            super().__init__(
                experiment_name="realtime_fine_rabi_stability",
                qubit_names=qubit_names,
                settings=settings,
                configuration_params=profile,
            )
            self.repetitions = repetitions
            self.run_indices = run_indices

        def define_experiment(self) -> RealtimeFineRabiStability:
            experiment = RealtimeFineRabiStability(
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
    
    pulse.pi_pulse_amplitude = 0.0322721
    
    if args.pi_amplitude is not None:
        pulse = profile.get_pi_params(
            args.qubit,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        )
        pulse.pi_pulse_amplitude = args.pi_amplitude

    settings = SettingsFineRabi(
        do_emulation=True,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
        exportation_method=ExportationMethod.NONE,
        update_params_method=UpdateParamsMethod.NONE,
        num_shots=args.shots,
        rotation_type=RotationType.PI_HALF,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        reset=ResetSettings(reset_type=ResetType.ACTIVE),
    )

    handler = RealtimeFineRabiStabilityHandler(
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().isoformat(timespec="seconds")
    csv_path = args.output_dir / f"{args.qubit}_realtime_fine_rabi_stability_{stamp}.csv"
    metrics_csv_path = (
        args.output_dir / f"{args.qubit}_realtime_fine_rabi_stability_metrics_{stamp}.csv"
    )
    png_path = args.output_dir / f"{args.qubit}_realtime_fine_rabi_stability_{stamp}.png"

    save_csv(csv_path, values, repetitions, args.qubit, timestamp)
    save_metrics_csv(metrics_csv_path, values)
    save_plot(png_path, values, repetitions, args.qubit)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved metrics CSV: {metrics_csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
