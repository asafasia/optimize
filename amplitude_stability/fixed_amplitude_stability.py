"""Monitor fine-Rabi stability by repeating 1D fine Rabi over time.

For a calibrated pi/2 pulse, the selected odd fine-Rabi points should stay
flat as the experiment is repeated. This script loops a 1D fine-Rabi
experiment for a requested duration, extracts the selected odd/even points
from every run, and plots each selected repetition as a function of time.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat 1D fine Rabi and plot selected odd/even points over time."
    )
    parser.add_argument("--qubit", default="q4", help="Qubit to measure, e.g. q4.")
    parser.add_argument(
        "--duration-min",
        type=float,
        default=10.0,
        help="How long to keep repeating the 1D fine-Rabi experiment.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of repeated runs.",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.0,
        help="Optional delay between completed runs.",
    )
    parser.add_argument("--shots", type=int, default=500, help="Shots per fine-Rabi point.")
    parser.add_argument(
        "--repetitions",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        default=[0, 100, 1],
        help="Fine-Rabi repetition range, passed to np.arange(start, stop, step).",
    )
    parser.add_argument(
        "--point-parity",
        choices=["odd", "even"],
        default="odd",
        help="Which repetition-number parity to track versus time.",
    )
    parser.add_argument(
        "--drop-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop the first and last selected points, matching the existing plotter style.",
    )
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
    if argv is None and Path(sys.argv[0]).name == "ipykernel_launcher.py":
        argv = []
    return parser.parse_args(argv)


def selected_repetition_mask(repetitions: np.ndarray, parity: str, drop_edges: bool) -> np.ndarray:
    mask = (repetitions % 2 == 1) if parity == "odd" else (repetitions % 2 == 0)

    if drop_edges:
        selected_indices = np.flatnonzero(mask)
        if selected_indices.size > 2:
            mask[selected_indices[0]] = False
            mask[selected_indices[-1]] = False

    return mask


def extract_trace(experiment_result, qubit: str, n_repetitions: int) -> np.ndarray:
    raw = np.asarray(experiment_result.get_data(f"handle_{qubit}"))
    values = np.abs(raw).reshape(-1)
    if values.size != n_repetitions:
        raise ValueError(
            f"Expected {n_repetitions} fine-Rabi points for {qubit}, got shape {raw.shape}."
        )
    return values


def save_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "timestamp",
        "run_index",
        "elapsed_s",
        "qubit",
        "repetition",
        "point_parity",
        "value",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_arrays(
    rows: list[dict[str, float | int | str]],
    selected_repetitions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_indices = sorted({int(r["run_index"]) for r in rows})
    elapsed_min = []
    values_by_run = []

    for run_index in run_indices:
        run_rows = [r for r in rows if int(r["run_index"]) == run_index]
        elapsed_min.append(float(run_rows[0]["elapsed_s"]) / 60)
        by_repetition = {int(r["repetition"]): float(r["value"]) for r in run_rows}
        values_by_run.append([by_repetition[int(rep)] for rep in selected_repetitions])

    return (
        np.asarray(run_indices, dtype=int),
        np.asarray(elapsed_min, dtype=float),
        np.asarray(values_by_run, dtype=float),
    )


def save_metrics_csv(
    path: Path,
    rows: list[dict[str, float | int | str]],
    selected_repetitions: np.ndarray,
) -> None:
    run_indices, elapsed_min, values = build_run_arrays(rows, selected_repetitions)
    if run_indices.size == 0:
        return

    std_values = np.std(values, axis=1)
    jump_scores = np.full(run_indices.shape, np.nan, dtype=float)
    if run_indices.size > 1:
        jump_scores[1:] = np.mean(np.abs(np.diff(values, axis=0)), axis=1)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_index",
                "elapsed_min",
                "selected_point_std",
                "selected_point_variance",
                "jump_score",
            ],
        )
        writer.writeheader()
        for run_index, elapsed, std_value, jump_score in zip(
            run_indices, elapsed_min, std_values, jump_scores
        ):
            writer.writerow(
                {
                    "run_index": int(run_index),
                    "elapsed_min": float(elapsed),
                    "selected_point_std": float(std_value),
                    "selected_point_variance": float(std_value**2),
                    "jump_score": float(jump_score),
                }
            )


def save_plot(
    path: Path,
    rows: list[dict[str, float | int | str]],
    selected_repetitions: np.ndarray,
    point_parity: str,
) -> None:
    _, elapsed_min, values = build_run_arrays(rows, selected_repetitions)
    std_values = np.std(values, axis=1)
    jump_scores = np.full(elapsed_min.shape, np.nan, dtype=float)
    if elapsed_min.size > 1:
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

    mesh = ax_top.pcolormesh(
        elapsed_min,
        selected_repetitions,
        values.T,
        shading="auto",
    )
    ax_top.set_ylabel("Fine-Rabi repetition")
    ax_top.set_title(f"1D fine-Rabi {point_parity} points stability")
    fig.colorbar(mesh, cax=cbar_ax, label="Excitation")

    ax_bottom.plot(
        elapsed_min,
        std_values,
        marker="o",
        linewidth=1.2,
        label="std across selected points",
    )
    ax_bottom.plot(
        elapsed_min,
        jump_scores,
        marker="s",
        linewidth=1.2,
        label="run-to-run jump score",
    )
    ax_bottom.set_xlabel("Elapsed time [min]")
    ax_bottom.set_ylabel("|output| instability")
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.legend(fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_outputs(
    csv_path: Path,
    metrics_csv_path: Path,
    png_path: Path,
    rows: list[dict[str, float | int | str]],
    selected_repetitions: np.ndarray,
    point_parity: str,
) -> None:
    save_csv(csv_path, rows)
    save_metrics_csv(metrics_csv_path, rows, selected_repetitions)
    save_plot(png_path, rows, selected_repetitions, point_parity)


def main() -> None:
    args = parse_args()

    global np, plt

    import matplotlib.pyplot as plt
    import numpy as np
    from laboneq.simple import AcquisitionType, AveragingMode, from_json

    import qratena.experiments.fine_rabi.fine_rabi_1d as fine_rabi_1d
    from qratena.experiments.fine_rabi.fine_rabi_1d import (
        FineRabi1DHandler,
        RotationType,
        SettingsFineRabi,
    )
    from qratena.system.components_params.reset_settings import ResetSettings
    from qratena.util.enums import (
        SUPPORTED_PULSE_SHAPES,
        ExportationMethod,
        ResetType,
        UpdateParamsMethod,
    )

    from resources.load_profile import load_profile, load_task_manager

    repetitions = np.arange(*args.repetitions)
    if repetitions.size == 0:
        raise ValueError("--repetitions produced an empty array.")

    mask = selected_repetition_mask(repetitions, args.point_parity, args.drop_edges)
    selected_repetitions = repetitions[mask]
    if selected_repetitions.size == 0:
        raise ValueError("No selected repetitions. Change --point-parity or --drop-edges.")

    profile = load_profile(args.profile_branch)
    task_manager = load_task_manager()

    # FineRabi1D currently reads this module-level variable inside define_experiment().
    fine_rabi_1d.profile = profile

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

    handler = FineRabi1DHandler(
        qubit_names=[args.qubit],
        repetitions=repetitions,
        settings=settings,
        profile=profile,
    )
    compiled_experiment = handler.get_compiled_experiment()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"{args.qubit}_fine_rabi_stability_{stamp}.csv"
    metrics_csv_path = args.output_dir / f"{args.qubit}_fine_rabi_stability_metrics_{stamp}.csv"
    png_path = args.output_dir / f"{args.qubit}_fine_rabi_stability_{stamp}.png"

    rows: list[dict[str, float | int | str]] = []
    start = time.monotonic()
    duration_s = args.duration_min * 60
    run_index = 0
    interrupted = False

    try:
        while True:
            elapsed_s = time.monotonic() - start
            if elapsed_s >= duration_s:
                break
            if args.max_runs is not None and run_index >= args.max_runs:
                break

            print(
                f"Running 1D fine Rabi {run_index + 1}; "
                f"elapsed {elapsed_s / 60:.2f}/{args.duration_min:.2f} min ..."
            )
            task_id = task_manager.submit_compiled_experiment(
                experiment_name=handler.experiment_name,
                profile_name=args.profile_name,
                qubit_names=handler.qubit_names,
                compiled_experiment=compiled_experiment,
                do_emulation=args.emulate,
            )
            task_result = task_manager.wait_for_result(task_id)

            experiment_result = from_json(task_result.raw_data)
            trace = extract_trace(experiment_result, args.qubit, repetitions.size)
            timestamp = datetime.now().isoformat(timespec="seconds")
            elapsed_s = time.monotonic() - start

            for repetition, value in zip(repetitions[mask], trace[mask]):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "run_index": run_index,
                        "elapsed_s": float(elapsed_s),
                        "qubit": args.qubit,
                        "repetition": int(repetition),
                        "point_parity": args.point_parity,
                        "value": float(value),
                    }
                )

            save_outputs(
                csv_path,
                metrics_csv_path,
                png_path,
                rows,
                selected_repetitions,
                args.point_parity,
            )

            run_index += 1
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping sweep early; saving completed runs.")
    finally:
        if rows:
            save_outputs(
                csv_path,
                metrics_csv_path,
                png_path,
                rows,
                selected_repetitions,
                args.point_parity,
            )

    if interrupted:
        print(f"Stopped after {run_index} completed runs.")
    else:
        print(f"Completed {run_index} runs.")
    if rows:
        print(f"Saved CSV: {csv_path}")
        print(f"Saved metrics CSV: {metrics_csv_path}")
        print(f"Saved plot: {png_path}")
    else:
        print("No completed runs; no CSV or plot was saved.")


if __name__ == "__main__":
    main()
