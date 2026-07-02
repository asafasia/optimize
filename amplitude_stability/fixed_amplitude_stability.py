"""Repeat fixed fine-Rabi amplitudes to check time stability.

This diagnostic tests whether the measured output jumps while the requested
amplitude is unchanged. It repeatedly runs the same compiled fine-Rabi
experiment for a small set of amplitude scale factors and saves the block-by-
block values to CSV and PNG files.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from laboneq.simple import AcquisitionType, AveragingMode, from_json

from qratena.experiments.fine_rabi.fine_rabi_2d import (
    FineRabi2DHandler,
    FineRabi2DSettings,
    RotationType,
    ScanParameter,
)
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    SUPPORTED_PULSE_SHAPES,
    ExportationMethod,
    ResetType,
    UpdateParamsMethod,
)

from resources.load_profile import load_profile, load_task_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat fixed fine-Rabi amplitude points and plot output vs block index."
    )
    parser.add_argument("--qubit", default="q4", help="Qubit to measure, e.g. q4.")
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.9, 1.0, 1.1],
        help="Amplitude scale factors to repeat.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=30,
        help="Number of repeated experiment submissions.",
    )
    parser.add_argument("--shots", type=int, default=500, help="Shots per amplitude point.")
    parser.add_argument(
        "--profile-name",
        default="main",
        help="Profile name used by the task manager submission.",
    )
    parser.add_argument(
        "--profile-branch",
        default="main",
        help="Profile branch to pull before compiling.",
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
        help="Run through task manager emulation instead of hardware.",
    )
    return parser.parse_args()


def extract_values(experiment_result, qubit: str, n_amplitudes: int) -> np.ndarray:
    """Return one real-valued output per requested amplitude."""
    raw = np.asarray(experiment_result.get_data(f"handle_{qubit}"))
    values = np.abs(raw).reshape(-1)
    if values.size != n_amplitudes:
        raise ValueError(
            f"Expected {n_amplitudes} values for {qubit}, got shape {raw.shape}."
        )
    return values


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = ["timestamp", "block", "qubit", "amplitude", "value"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: list[dict[str, float | int | str]], amplitudes: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for amplitude in amplitudes:
        xs = [r["block"] for r in rows if r["amplitude"] == float(amplitude)]
        ys = [r["value"] for r in rows if r["amplitude"] == float(amplitude)]
        ax.plot(xs, ys, marker="o", linewidth=1.2, label=f"amp={amplitude:g}")

    ax.set_xlabel("Repeated block index")
    ax.set_ylabel("|output|")
    ax.set_title("Fixed-amplitude stability check")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    amplitudes = np.asarray(args.amplitudes, dtype=float)

    profile = load_profile(args.profile_branch)
    task_manager = load_task_manager()

    settings = FineRabi2DSettings(
        do_emulation=args.emulate,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        exportation_method=ExportationMethod.NONE,
        update_params_method=UpdateParamsMethod.NONE,
        num_shots=args.shots,
        rotation_type=RotationType.PI,
        scan_parameter=ScanParameter.AMPLITUDE,
        pulse_shape=SUPPORTED_PULSE_SHAPES.drag,
        reset=ResetSettings(reset_type=ResetType.ACTIVE),
    )

    handler = FineRabi2DHandler(
        qubit_names=[args.qubit],
        repetitions=np.array([1]),
        scan_values=amplitudes,
        settings=settings,
        profile=profile,
    )
    compiled_experiment = handler.get_compiled_experiment()

    rows: list[dict[str, float | int | str]] = []
    for block in range(args.blocks):
        print(f"Running block {block + 1}/{args.blocks} ...")
        task_result = task_manager.wait(
            task_manager.run_compiled_experiment(
                experiment_name=handler.experiment_name,
                profile_name=args.profile_name,
                qubit_names=handler.qubit_names,
                compiled_experiment=compiled_experiment,
                do_emulation=args.emulate,
            )
        )

        experiment_result = from_json(task_result.raw_data)
        values = extract_values(experiment_result, args.qubit, len(amplitudes))
        timestamp = datetime.now().isoformat(timespec="seconds")

        for amplitude, value in zip(amplitudes, values):
            rows.append(
                {
                    "timestamp": timestamp,
                    "block": block,
                    "qubit": args.qubit,
                    "amplitude": float(amplitude),
                    "value": float(value),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"{args.qubit}_fixed_amplitude_stability_{stamp}.csv"
    png_path = args.output_dir / f"{args.qubit}_fixed_amplitude_stability_{stamp}.png"

    save_csv(csv_path, rows)
    save_plot(png_path, rows, amplitudes)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
