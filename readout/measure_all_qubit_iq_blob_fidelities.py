from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKBENCH_ROOT.parent
DEPENDENCY_ROOTS = (
    PROJECT_ROOT / "qratena",
    PROJECT_ROOT / "qigeon",
    PROJECT_ROOT / "qhipu-lab",
    PROJECT_ROOT / "q-b2c",
)
for path in (*DEPENDENCY_ROOTS, PROJECT_ROOT, WORKBENCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("QRATENA_DATA_DIR", str(WORKBENCH_ROOT / "outputs" / "qratena"))


PROFILE_NAME = "main"
OUTPUT_ROOT = Path("outputs/readout_iq_blob_fidelities")
NUM_BINS = 12
ACTIVE_RESET_NUM = 5
RESET_MODE = "active"
STATES = ["g", "e"]
DO_EMULATION = False
NUM_SHOTS = 10_000


def main() -> None:
    args = parse_args()

    import matplotlib.pyplot as plt
    from laboneq.core.types.enums.acquisition_type import AcquisitionType
    from laboneq.core.types.enums.averaging_mode import AveragingMode
    from laboneq.simple import from_json
    from qratena.experiments.base_experiment import ExperimentSettings
    from qratena.experiments.experiment_handler import ExperimentHandler
    from qratena.experiments.iq_blobs import EXPERIMENT_NAME, IQBlobsHandler
    from qratena.util.enums import (
        ExportationMethod,
        SUPPORTED_PULSE_SHAPES,
        UpdateParamsMethod,
    )
    from resources.load_profile import load_profile, load_task_manager

    validate_states(args.states)
    profile = load_profile(args.profile_name)
    qubit_names = args.qubits or sorted(profile.qubits.keys(), key=qubit_sort_key)

    settings = ExperimentSettings(
        num_shots=args.num_shots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLE_SHOT,
        exportation_method=ExportationMethod.NONE,
        update_params_method=UpdateParamsMethod.NONE,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        reset=build_reset_settings(args.reset_mode, args.active_reset_num),
        do_emulation=True,
        states=2,
    )
    class ProfileIQBlobsHandler(IQBlobsHandler):
        def __init__(
            self,
            qubit_names: list[str],
            settings: Any,
            profile: Any,
        ) -> None:
            ExperimentHandler.__init__(
                self,
                experiment_name=EXPERIMENT_NAME,
                qubit_names=qubit_names,
                settings=settings,
                configuration_params=profile,
            )
            self.num_shots = settings.num_shots
            self.readout_durations = None
            self.readout_amplitude_prefactors = None

    handler = ProfileIQBlobsHandler(
        qubit_names=qubit_names,
        settings=settings,
        profile=profile,
    )

    start = perf_counter()
    if args.do_emulation:
        handler.run()
    else:
        task_manager = load_task_manager()
        compiled_experiment = handler.get_compiled_experiment()
        task = task_manager.run_compiled_experiment(
            experiment_name=handler.experiment_name,
            profile_name=args.profile_name,
            qubit_names=handler.qubit_names,
            compiled_experiment=compiled_experiment,
            do_emulation=False,
        )
        task_result = task_manager.wait(task)
        handler.experiment_result = from_json(task_result.raw_data)
        handler.analysis_result = handler.analyze()
        handler.figs = []
        if args.export_iq_blobs:
            handler.figs = handler.plot()
            handler.export_data(figs=handler.figs)

    elapsed = perf_counter() - start
    rows = extract_fidelity_rows(qubit_names, handler.data)
    run_dir = make_run_dir(args.output_root, qubit_names)
    save_csv(run_dir / "readout_fidelities.csv", rows)
    save_json(
        run_dir / "summary.json",
        build_summary(
            rows=rows,
            profile_name=args.profile_name,
            states=args.states,
            reset_mode=args.reset_mode,
            active_reset_num=args.active_reset_num,
            do_emulation=args.do_emulation,
        ),
    )

    figure = plot_fidelity_summary(rows, num_bins=args.num_bins)
    figure.savefig(run_dir / "readout_fidelity_histogram.png", dpi=200, bbox_inches="tight")

    print(f"Measured {count_measured(rows)} of {len(qubit_names)} qubits")
    print(f"IQ blobs experiment finished in {elapsed:.1f}s")
    print(f"Saved CSV, summary JSON, and histogram figure to {run_dir}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure readout fidelity for selected qubits with the IQ blobs "
            "experiment and save a fidelity histogram."
        )
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        help="Qubits to measure. Defaults to every qubit in the selected profile.",
    )
    parser.add_argument("--profile-name", default=PROFILE_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where the timestamped run folder is saved.",
    )
    parser.add_argument("--states", nargs="+", default=STATES)
    parser.add_argument(
        "--reset-mode",
        choices=["active", "passive", "none"],
        default=RESET_MODE,
        help="Reset mode used before each IQ blobs shot.",
    )
    parser.add_argument("--active-reset-num", type=int, default=ACTIVE_RESET_NUM)
    parser.add_argument("--num-shots", type=int, default=NUM_SHOTS)
    parser.add_argument("--num-bins", type=int, default=NUM_BINS)
    parser.add_argument("--do-emulation", action="store_true", default=DO_EMULATION)
    parser.add_argument(
        "--export-iq-blobs",
        action="store_true",
        help="Also export the per-qubit IQ blob plots through the Qratena handler.",
    )
    parser.add_argument("--show", action="store_true")
    args, _unknown_args = parser.parse_known_args()
    return args


def validate_states(states: list[str]) -> None:
    if states != ["g", "e"]:
        raise ValueError(
            "This direct IQ blobs experiment currently supports only --states g e."
        )


def build_reset_settings(reset_mode: str, active_reset_num: int) -> Any:
    from qratena.system.components_params.reset_settings import ResetSettings
    from qratena.util.enums import ResetType

    if reset_mode == "active":
        return ResetSettings(ResetType.ACTIVE, reset_num=active_reset_num)
    if reset_mode == "passive":
        return ResetSettings(ResetType.PASSIVE)
    return ResetSettings()


def extract_fidelity_rows(
    qubit_names: list[str],
    iq_blob_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for qubit_name in qubit_names:
        qubit_result = iq_blob_results.get(qubit_name, {}) or {}
        fidelity_matrix = qubit_result.get("readout_fidelity_matrix")
        ground_fidelity, excited_fidelity = diagonal_fidelities(fidelity_matrix)
        readout_fidelity = optional_float(qubit_result.get("readout_fidelity"))

        rows.append(
            {
                "qubit": qubit_name,
                "readout_fidelity": readout_fidelity,
                "ground_readout_fidelity": ground_fidelity,
                "excited_readout_fidelity": excited_fidelity,
                "threshold": optional_float(qubit_result.get("threshold")),
                "blobs_angle": optional_float(qubit_result.get("blobs_angle")),
                "status": "measured" if readout_fidelity is not None else "missing",
            }
        )
    return rows


def diagonal_fidelities(value: Any) -> tuple[float | None, float | None]:
    import numpy as np

    if value is None:
        return None, None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, None
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None, None
    return float(matrix[0, 0]), float(matrix[1, 1])


def build_summary(
    rows: list[dict[str, Any]],
    profile_name: str,
    states: list[str],
    reset_mode: str,
    active_reset_num: int,
    do_emulation: bool,
) -> dict[str, Any]:
    import numpy as np

    fidelities = np.asarray(
        [row["readout_fidelity"] for row in rows if row["readout_fidelity"] is not None],
        dtype=float,
    )
    summary: dict[str, Any] = {
        "profile_name": profile_name,
        "states": states,
        "reset_mode": reset_mode,
        "active_reset_num": active_reset_num if reset_mode == "active" else None,
        "do_emulation": do_emulation,
        "num_qubits_requested": len(rows),
        "num_qubits_measured": int(fidelities.size),
    }
    if fidelities.size:
        summary.update(
            {
                "mean_readout_fidelity": float(np.mean(fidelities)),
                "median_readout_fidelity": float(np.median(fidelities)),
                "min_readout_fidelity": float(np.min(fidelities)),
                "max_readout_fidelity": float(np.max(fidelities)),
                "std_readout_fidelity": float(np.std(fidelities)),
            }
        )
    return summary


def plot_fidelity_summary(rows: list[dict[str, Any]], num_bins: int) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np

    measured_rows = [row for row in rows if row["readout_fidelity"] is not None]
    if not measured_rows:
        raise ValueError("No readout_fidelity values were found in the IQ blobs result.")

    labels = [row["qubit"] for row in measured_rows]
    fidelities = np.asarray(
        [row["readout_fidelity"] for row in measured_rows],
        dtype=float,
    )

    figure, (bar_axis, hist_axis) = plt.subplots(
        1,
        2,
        figsize=(max(11.0, len(labels) * 0.42), 5.8),
        gridspec_kw={"width_ratios": [4.0, 1.25]},
    )
    x_values = np.arange(len(labels))

    bar_axis.bar(
        x_values,
        fidelities,
        color=fidelity_colors(fidelities),
        edgecolor="black",
        linewidth=0.7,
    )
    add_reference_lines(bar_axis, fidelities)
    bar_axis.set_title("Readout fidelity by qubit")
    bar_axis.set_ylabel("Readout fidelity")
    bar_axis.set_xticks(x_values)
    bar_axis.set_xticklabels(labels, rotation=60, ha="right")
    bar_axis.set_ylim(fidelity_axis_limits(fidelities))
    bar_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)

    hist_axis.hist(
        fidelities,
        bins=histogram_bins(fidelities, num_bins),
        orientation="horizontal",
        color="#6a8caf",
        edgecolor="black",
        alpha=0.85,
    )
    add_reference_lines(hist_axis, fidelities)
    hist_axis.set_title("Histogram")
    hist_axis.set_xlabel("Qubits")
    hist_axis.set_ylim(bar_axis.get_ylim())
    hist_axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)

    figure.suptitle(
        (
            f"Measured {len(measured_rows)} qubits | "
            f"mean={np.mean(fidelities):.4f}, "
            f"median={np.median(fidelities):.4f}, "
            f"min={np.min(fidelities):.4f}"
        ),
        y=1.02,
    )
    figure.tight_layout()
    return figure


def histogram_bins(fidelities: np.ndarray, num_bins: int) -> np.ndarray:
    lower = max(0.5, float(np.min(fidelities)) - 0.02)
    return np.linspace(lower, 1.0, max(2, num_bins + 1))


def add_reference_lines(axis: Any, fidelities: np.ndarray) -> None:
    axis.axhline(float(np.mean(fidelities)), color="black", linewidth=1.2)
    axis.axhline(0.90, color="#b23a48", linestyle="--", linewidth=1.0)
    axis.axhline(0.95, color="#2f7d5c", linestyle="--", linewidth=1.0)


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
    return max(0.5, float(np.min(fidelities)) - 0.04), 1.0


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "qubit",
                "readout_fidelity",
                "ground_readout_fidelity",
                "excited_readout_fidelity",
                "threshold",
                "blobs_angle",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(value, json_file, indent=2)
        json_file.write("\n")


def make_run_dir(output_root: Path, qubit_names: list[str]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qubit_label = "all_qubits" if len(qubit_names) > 3 else "_".join(qubit_names)
    run_dir = output_root / f"{timestamp}_{qubit_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def qubit_sort_key(qubit_name: str) -> tuple[str, int | str]:
    prefix = "".join(character for character in qubit_name if not character.isdigit())
    suffix = qubit_name[len(prefix):]
    if suffix.isdigit():
        return prefix, int(suffix)
    return prefix, qubit_name


def count_measured(rows: list[dict[str, Any]]) -> int:
    return sum(row["readout_fidelity"] is not None for row in rows)


if __name__ == "__main__":
    main()
