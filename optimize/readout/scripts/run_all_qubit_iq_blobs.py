from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[3]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType

from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
    ReadoutFidelityWorkflowSettings,
)
from resources.load_profile import load_profile, load_task_manager


PROFILE_NAME = "main"
OUTPUT_ROOT = Path("data/readout_iq_blobs_all_qubits")

DO_EMULATION = False
SHOW_HANDLER_OUTPUT = False
TASK_STATUS_POLL_INTERVAL = 10.0
RESET = ResetSettings(ResetType.ACTIVE, reset_num=5)



def main() -> None:
    args = parse_args()

    profile = load_profile()
    task_manager = load_task_manager()

    qubit_names = args.qubits or sorted(
        profile.qubits.keys(),
        key=qubit_sort_key,
    )
    
    qubit_names = qubit_names[:-3]  # --- IGNORE ---

    settings = ReadoutFidelityWorkflowSettings(
        profile_name=args.profile_name,
        do_emulation=args.do_emulation,
        run_resonator=False,
        run_kernels=False,
        run_iq_blobs=True,
        do_plotting=False,
        show_handler_output=args.show_handler_output,
        report_timing=True,
        task_status_poll_interval=args.task_status_poll_interval,
        reset=RESET,
    )

    workflow = ReadoutFidelityWorkflow(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
    )
    result = workflow.run()

    fidelity_rows = extract_fidelity_rows(qubit_names, result["iq_blobs"])
    run_dir = make_run_dir(args.output_root, qubit_names)
    save_fidelity_csv(run_dir / "readout_fidelities.csv", fidelity_rows)

    figure = plot_readout_fidelities(fidelity_rows)
    figure.savefig(run_dir / "readout_fidelities.png", dpi=200, bbox_inches="tight")

    print(f"Measured {count_measured(fidelity_rows)} of {len(qubit_names)} qubits")
    print(f"Saved summary to {run_dir}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single IQ blobs experiment for all selected qubits and plot "
            "their measured readout fidelities."
        )
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        help="Qubits to measure. Defaults to every qubit in the loaded profile.",
    )
    parser.add_argument("--profile-name", default=PROFILE_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where the run folder, plot, and CSV are saved.",
    )
    parser.add_argument("--do-emulation", action="store_true", default=DO_EMULATION)
    parser.add_argument(
        "--show-handler-output",
        action="store_true",
        default=SHOW_HANDLER_OUTPUT,
    )
    parser.add_argument(
        "--task-status-poll-interval",
        type=float,
        default=TASK_STATUS_POLL_INTERVAL,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the summary plot after saving it.",
    )
    args, _unknown_args = parser.parse_known_args()
    return args


def qubit_sort_key(qubit_name: str) -> tuple[str, int | str]:
    prefix = "".join(character for character in qubit_name if not character.isdigit())
    suffix = qubit_name[len(prefix):]
    if suffix.isdigit():
        return prefix, int(suffix)
    return prefix, qubit_name


def extract_fidelity_rows(
    qubit_names: list[str],
    iq_blob_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for qubit_name in qubit_names:
        qubit_result = iq_blob_results.get(qubit_name, {}) or {}
        fidelity = optional_float(qubit_result.get("readout_fidelity"))
        error = first_optional_float(
            qubit_result,
            [
                "readout_fidelity_std",
                "readout_fidelity_error",
                "readout_fidelity_err",
                "average_readout_fidelity_std",
                "averaged_readout_fidelity_std",
                "fidelity_std",
                "fidelity_error",
            ],
        )
        separation = first_optional_float(
            qubit_result,
            [
                "separation",
                "readout_separation",
                "iq_separation",
                "state_separation",
            ],
        )

        rows.append(
            {
                "qubit": qubit_name,
                "readout_fidelity": fidelity,
                "readout_fidelity_error": error,
                "separation": separation,
                "status": "measured" if fidelity is not None else "missing",
            }
        )

    return rows


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_optional_float(data: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = optional_float(data.get(key))
        if value is not None:
            return value
    return None


def make_run_dir(output_root: Path, qubit_names: list[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qubit_label = "all_qubits" if len(qubit_names) > 3 else "_".join(qubit_names)
    run_dir = output_root / f"{timestamp}_{qubit_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_fidelity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "qubit",
                "readout_fidelity",
                "readout_fidelity_error",
                "separation",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_readout_fidelities(rows: list[dict[str, Any]]) -> plt.Figure:
    measured_rows = [row for row in rows if row["readout_fidelity"] is not None]
    if not measured_rows:
        raise ValueError("No readout_fidelity values were found in the IQ blobs result.")

    labels = [row["qubit"] for row in measured_rows]
    fidelities = np.array(
        [float(row["readout_fidelity"]) for row in measured_rows],
        dtype=float,
    )
    errors = [
        0.0 if row["readout_fidelity_error"] is None else row["readout_fidelity_error"]
        for row in measured_rows
    ]

    colors = fidelity_colors(fidelities)
    fig, (bar_axis, hist_axis) = plt.subplots(
        1,
        2,
        figsize=(max(11.0, len(labels) * 0.42), 5.8),
        gridspec_kw={"width_ratios": [4.0, 1.2]},
    )

    x_values = np.arange(len(labels))
    bar_axis.bar(
        x_values,
        fidelities,
        yerr=errors if any(error > 0.0 for error in errors) else None,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        capsize=3,
    )
    bar_axis.axhline(float(np.mean(fidelities)), color="black", linewidth=1.2)
    bar_axis.axhline(0.90, color="#b23a48", linestyle="--", linewidth=1.0)
    bar_axis.axhline(0.95, color="#2f7d5c", linestyle="--", linewidth=1.0)
    bar_axis.set_title("Readout fidelity by qubit")
    bar_axis.set_ylabel("Readout fidelity")
    bar_axis.set_xticks(x_values)
    bar_axis.set_xticklabels(labels, rotation=60, ha="right")
    bar_axis.set_ylim(fidelity_axis_limits(fidelities))
    bar_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)

    hist_axis.hist(
        fidelities,
        bins=np.linspace(max(0.5, fidelities.min() - 0.02), 1.0, 12),
        orientation="horizontal",
        color="#6a8caf",
        edgecolor="black",
        alpha=0.85,
    )
    hist_axis.axhline(float(np.mean(fidelities)), color="black", linewidth=1.2)
    hist_axis.axhline(0.90, color="#b23a48", linestyle="--", linewidth=1.0)
    hist_axis.axhline(0.95, color="#2f7d5c", linestyle="--", linewidth=1.0)
    hist_axis.set_title("Histogram")
    hist_axis.set_xlabel("Qubits")
    hist_axis.set_ylim(bar_axis.get_ylim())
    hist_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)

    fig.suptitle(
        (
            f"Measured {len(measured_rows)} qubits | "
            f"mean={np.mean(fidelities):.4f}, "
            f"min={np.min(fidelities):.4f}, "
            f"max={np.max(fidelities):.4f}"
        ),
        y=1.02,
    )
    fig.tight_layout()
    return fig


def fidelity_colors(fidelities: np.ndarray) -> list[str]:
    colors = []
    for fidelity in fidelities:
        if fidelity >= 0.95:
            colors.append("#3f8f63")
        elif fidelity >= 0.90:
            colors.append("#d49a3a")
        else:
            colors.append("#bd4f5a")
    return colors


def fidelity_axis_limits(fidelities: np.ndarray) -> tuple[float, float]:
    lower = max(0.5, float(np.min(fidelities)) - 0.04)
    return lower, 1.0


def count_measured(rows: list[dict[str, Any]]) -> int:
    return sum(row["readout_fidelity"] is not None for row in rows)


if __name__ == "__main__":
    main()
