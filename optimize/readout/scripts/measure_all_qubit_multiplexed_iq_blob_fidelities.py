from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[3]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from workbench_bootstrap import (
    QRATENA_DATA_ROOT,
    QRATENA_NINJA_PROFILE,
    setup_workbench_environment,
)

setup_workbench_environment()


PROFILE_NAME = "main_asaf"
OUTPUT_ROOT = Path("outputs/readout_multiplexed_iq_blob_fidelities")
ACTIVE_RESET_NUM = 5
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
        SUPPORTED_PULSE_TYPES,
        UpdateParamsMethod,
    )

    ensure_qratena_runtime_settings()

    from resources.load_profile import load_profile, load_task_manager

    validate_states(args.states)
    profile = load_profile(args.profile_name)
    qubit_names = args.qubits or sorted(profile.qubits.keys(), key=qubit_sort_key)
    run_datetime = datetime.now()

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

    common_settings = {
        "num_shots": args.num_shots,
        "acquisition_type": AcquisitionType.INTEGRATION,
        "averaging_mode": AveragingMode.SINGLE_SHOT,
        "exportation_method": ExportationMethod.NONE,
        "update_params_method": UpdateParamsMethod.NONE,
        "pulse_shape": SUPPORTED_PULSE_SHAPES.const,
        "do_emulation": True,
        "states": 2,
    }
    readout_metadata = collect_readout_metadata(
        profile=profile,
        qubit_names=qubit_names,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        pulse_type=SUPPORTED_PULSE_TYPES.readout,
    )

    task_manager = None if args.do_emulation else load_task_manager()
    start = perf_counter()
    no_active_rows = run_iq_blobs_condition(
        condition="without_active_reset",
        reset=build_reset_settings("passive", args.active_reset_num),
        handler_class=ProfileIQBlobsHandler,
        settings_class=ExperimentSettings,
        common_settings=common_settings,
        qubit_names=qubit_names,
        profile=profile,
        profile_name=args.profile_name,
        task_manager=task_manager,
        do_emulation=args.do_emulation,
        export_iq_blobs=args.export_iq_blobs,
        from_json=from_json,
    )
    active_rows = run_iq_blobs_condition(
        condition="with_active_reset",
        reset=build_reset_settings("active", args.active_reset_num),
        handler_class=ProfileIQBlobsHandler,
        settings_class=ExperimentSettings,
        common_settings=common_settings,
        qubit_names=qubit_names,
        profile=profile,
        profile_name=args.profile_name,
        task_manager=task_manager,
        do_emulation=args.do_emulation,
        export_iq_blobs=args.export_iq_blobs,
        from_json=from_json,
    )
    elapsed = perf_counter() - start

    rows = no_active_rows + active_rows
    comparison_rows = build_comparison_rows(qubit_names, no_active_rows, active_rows)
    run_dir = make_run_dir(args.output_root, qubit_names)
    save_csv(run_dir / "readout_fidelities.csv", rows)
    save_comparison_csv(run_dir / "readout_fidelity_comparison.csv", comparison_rows)
    save_json(
        run_dir / "summary.json",
        build_summary(
            rows=comparison_rows,
            profile_name=args.profile_name,
            states=args.states,
            active_reset_num=args.active_reset_num,
            do_emulation=args.do_emulation,
            num_shots=args.num_shots,
            run_datetime=run_datetime,
            readout_metadata=readout_metadata,
        ),
    )

    figure = plot_fidelity_comparison(
        comparison_rows,
        metadata=build_plot_metadata(
            profile_name=args.profile_name,
            run_datetime=run_datetime,
            qubit_names=qubit_names,
            states=args.states,
            num_shots=args.num_shots,
            active_reset_num=args.active_reset_num,
            do_emulation=args.do_emulation,
            elapsed=elapsed,
            readout_metadata=readout_metadata,
        ),
    )
    figure.savefig(run_dir / "readout_fidelity_comparison.png", dpi=200, bbox_inches="tight")

    print(f"Measured {count_measured(comparison_rows)} of {len(qubit_names)} qubits")
    print(f"Two IQ blobs experiments finished in {elapsed:.1f}s")
    print(f"Saved CSV, summary JSON, and comparison figure to {run_dir}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure multiplexed IQ blob readout fidelity with and without "
            "active reset, then save a per-qubit comparison plot."
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
    parser.add_argument("--active-reset-num", type=int, default=ACTIVE_RESET_NUM)
    parser.add_argument("--num-shots", type=int, default=NUM_SHOTS)
    parser.add_argument("--do-emulation", action="store_true", default=DO_EMULATION)
    parser.add_argument(
        "--export-iq-blobs",
        action="store_true",
        help="Also export the per-qubit IQ blob plots through the Qratena handler.",
    )
    parser.add_argument("--show", action="store_true")
    args, _unknown_args = parser.parse_known_args()
    return args


def ensure_qratena_runtime_settings() -> None:
    from qratena.util import settings

    if (settings.DATA_DIR_PATH / QRATENA_NINJA_PROFILE).exists():
        return

    try:
        settings.configure(data_dir=QRATENA_DATA_ROOT)
    except PermissionError:
        settings.DATA_DIR_PATH = QRATENA_DATA_ROOT
        settings._data_dir_overridden = True
        settings._resolve_derived_paths()

    if not (settings.DATA_DIR_PATH / QRATENA_NINJA_PROFILE).exists():
        raise FileNotFoundError(
            "Could not find Qratena ninja profile under "
            f"{settings.DATA_DIR_PATH / QRATENA_NINJA_PROFILE}"
        )


def run_iq_blobs_condition(
    condition: str,
    reset: Any,
    handler_class: Any,
    settings_class: Any,
    common_settings: dict[str, Any],
    qubit_names: list[str],
    profile: Any,
    profile_name: str,
    task_manager: Any,
    do_emulation: bool,
    export_iq_blobs: bool,
    from_json: Any,
) -> list[dict[str, Any]]:
    settings = settings_class(reset=reset, **common_settings)
    handler = handler_class(
        qubit_names=qubit_names,
        settings=settings,
        profile=profile,
    )

    print(f"Running IQ blobs {condition}...")
    if do_emulation:
        handler.run()
    else:
        compiled_experiment = handler.get_compiled_experiment()
        task_id = task_manager.submit_compiled_experiment(
            experiment_name=f"{handler.experiment_name}_{condition}",
            profile_name=profile_name,
            qubit_names=handler.qubit_names,
            compiled_experiment=compiled_experiment,
            do_emulation=False,
        )
        print(f"{condition} task_id={task_id}")
        task_result = task_manager.wait_for_result(task_id)
        handler.experiment_result = from_json(task_result.raw_data)
        handler.analysis_result = handler.analyze()
        handler.figs = []
        if export_iq_blobs:
            handler.figs = handler.plot()
            handler.export_data(figs=handler.figs)

    return extract_fidelity_rows(qubit_names, handler.data, condition)


def validate_states(states: list[str]) -> None:
    if states != ["g", "e"]:
        raise ValueError(
            "This direct IQ blobs experiment currently supports only --states g e."
        )


def build_reset_settings(reset_mode: str, active_reset_num: int) -> Any:
    from qratena.system.components_params.reset_settings import ResetSettings
    from qratena.util.enums import ResetType

    if reset_mode == "active":
        return ResetSettings(
            reset_type=ResetType.ACTIVE,
            reset_num=active_reset_num,
        )
    if reset_mode == "passive":
        return ResetSettings(reset_type=ResetType.PASSIVE)
    return ResetSettings()


def extract_fidelity_rows(
    qubit_names: list[str],
    iq_blob_results: dict[str, dict[str, Any]],
    condition: str,
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
                "condition": condition,
                "readout_fidelity": readout_fidelity,
                "ground_readout_fidelity": ground_fidelity,
                "excited_readout_fidelity": excited_fidelity,
                "threshold": optional_float(qubit_result.get("threshold")),
                "blobs_angle": optional_float(qubit_result.get("blobs_angle")),
                "status": "measured" if readout_fidelity is not None else "missing",
            }
        )
    return rows


def build_comparison_rows(
    qubit_names: list[str],
    no_active_rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    no_active_by_qubit = rows_by_qubit(no_active_rows)
    active_by_qubit = rows_by_qubit(active_rows)
    rows = []

    for qubit_name in qubit_names:
        no_active_fidelity = optional_float(
            no_active_by_qubit.get(qubit_name, {}).get("readout_fidelity")
        )
        active_fidelity = optional_float(
            active_by_qubit.get(qubit_name, {}).get("readout_fidelity")
        )
        rows.append(
            {
                "qubit": qubit_name,
                "fidelity_without_active_reset": no_active_fidelity,
                "fidelity_with_active_reset": active_fidelity,
                "fidelity_delta_active_minus_without": optional_difference(
                    active_fidelity,
                    no_active_fidelity,
                ),
                "status": (
                    "measured"
                    if no_active_fidelity is not None or active_fidelity is not None
                    else "missing"
                ),
            }
        )

    return rows


def rows_by_qubit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["qubit"]: row for row in rows}


def optional_difference(
    first_value: float | None,
    second_value: float | None,
) -> float | None:
    if first_value is None or second_value is None:
        return None
    return first_value - second_value


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


def collect_readout_metadata(
    profile: Any,
    qubit_names: list[str],
    pulse_shape: Any,
    pulse_type: Any,
) -> dict[str, Any]:
    pulse_lengths = []
    amplitudes = []
    resonator_frequencies = []

    for qubit_name in qubit_names:
        qubit = profile.qubits[qubit_name]
        pulse = qubit.pulses[pulse_type][pulse_shape]
        pulse_lengths.append(optional_float(getattr(pulse, "readout_duration", None)))
        amplitudes.append(optional_float(getattr(pulse, "readout_amplitude", None)))
        resonator_frequencies.append(
            optional_float(getattr(qubit.readout_resonator_frequency, "value", None))
        )

    return {
        "readout_pulse_length_s": summarize_values(pulse_lengths),
        "readout_amplitude": summarize_values(amplitudes),
        "readout_resonator_frequency_hz": summarize_values(resonator_frequencies),
    }


def summarize_values(values: list[float | None]) -> dict[str, Any]:
    import numpy as np

    measured = np.asarray([value for value in values if value is not None], dtype=float)
    if measured.size == 0:
        return {"count": 0, "common": None, "min": None, "max": None, "mean": None}

    min_value = float(np.min(measured))
    max_value = float(np.max(measured))
    common_value = float(measured[0]) if np.allclose(measured, measured[0]) else None

    return {
        "count": int(measured.size),
        "common": common_value,
        "min": min_value,
        "max": max_value,
        "mean": float(np.mean(measured)),
    }


def build_summary(
    rows: list[dict[str, Any]],
    profile_name: str,
    states: list[str],
    active_reset_num: int,
    do_emulation: bool,
    num_shots: int,
    run_datetime: datetime,
    readout_metadata: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    no_active_fidelities = np.asarray(
        [
            row["fidelity_without_active_reset"]
            for row in rows
            if row["fidelity_without_active_reset"] is not None
        ],
        dtype=float,
    )
    active_fidelities = np.asarray(
        [
            row["fidelity_with_active_reset"]
            for row in rows
            if row["fidelity_with_active_reset"] is not None
        ],
        dtype=float,
    )
    summary: dict[str, Any] = {
        "profile_name": profile_name,
        "states": states,
        "active_reset_num": active_reset_num,
        "num_shots_per_state": num_shots,
        "do_emulation": do_emulation,
        "run_datetime": run_datetime.isoformat(timespec="seconds"),
        "num_qubits_requested": len(rows),
        "num_qubits_measured_without_active_reset": int(no_active_fidelities.size),
        "num_qubits_measured_with_active_reset": int(active_fidelities.size),
        "readout": readout_metadata,
    }
    if no_active_fidelities.size:
        summary.update(
            {
                "mean_fidelity_without_active_reset": float(np.mean(no_active_fidelities)),
                "median_fidelity_without_active_reset": float(
                    np.median(no_active_fidelities)
                ),
                "min_fidelity_without_active_reset": float(np.min(no_active_fidelities)),
                "max_fidelity_without_active_reset": float(np.max(no_active_fidelities)),
            }
        )
    if active_fidelities.size:
        summary.update(
            {
                "mean_fidelity_with_active_reset": float(np.mean(active_fidelities)),
                "median_fidelity_with_active_reset": float(np.median(active_fidelities)),
                "min_fidelity_with_active_reset": float(np.min(active_fidelities)),
                "max_fidelity_with_active_reset": float(np.max(active_fidelities)),
            }
        )
    return summary


def plot_fidelity_comparison(rows: list[dict[str, Any]], metadata: list[str]) -> Any:
    import matplotlib.pyplot as plt
    import numpy as np

    measured_rows = [
        row
        for row in rows
        if row["fidelity_without_active_reset"] is not None
        or row["fidelity_with_active_reset"] is not None
    ]
    if not measured_rows:
        raise ValueError("No readout_fidelity values were found in the IQ blobs results.")

    labels = [row["qubit"] for row in measured_rows]
    no_active_fidelities = values_or_nan(
        measured_rows,
        "fidelity_without_active_reset",
    )
    active_fidelities = values_or_nan(
        measured_rows,
        "fidelity_with_active_reset",
    )
    all_fidelities = np.concatenate(
        [
            no_active_fidelities[~np.isnan(no_active_fidelities)],
            active_fidelities[~np.isnan(active_fidelities)],
        ]
    )

    figure, axis = plt.subplots(figsize=(max(11.0, len(labels) * 0.56), 6.8))
    x_values = np.arange(len(labels))
    width = 0.38

    axis.bar(
        x_values - width / 2,
        no_active_fidelities,
        width=width,
        color="#6a8caf",
        edgecolor="black",
        linewidth=0.7,
        label="Without active reset",
    )
    axis.bar(
        x_values + width / 2,
        active_fidelities,
        width=width,
        color="#3f8f63",
        edgecolor="black",
        linewidth=0.7,
        label="With active reset",
    )
    add_reference_lines(axis, all_fidelities)
    axis.set_title("Readout fidelity by qubit")
    axis.set_ylabel("Readout fidelity")
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=60, ha="right")
    axis.set_ylim(fidelity_axis_limits(all_fidelities))
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    axis.legend()
    add_delta_labels(axis, x_values, active_fidelities, no_active_fidelities)

    figure.suptitle(
        (
            f"Measured {len(measured_rows)} qubits | "
            f"mean without active reset={np.nanmean(no_active_fidelities):.4f}, "
            f"mean with active reset={np.nanmean(active_fidelities):.4f}"
        ),
        y=1.02,
    )
    figure.text(
        0.01,
        0.02,
        "\n".join(metadata),
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
    )
    figure.tight_layout(rect=(0.0, 0.16, 1.0, 0.98))
    return figure


def values_or_nan(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    import numpy as np

    return np.asarray(
        [np.nan if row[key] is None else float(row[key]) for row in rows],
        dtype=float,
    )


def build_plot_metadata(
    profile_name: str,
    run_datetime: datetime,
    qubit_names: list[str],
    states: list[str],
    num_shots: int,
    active_reset_num: int,
    do_emulation: bool,
    elapsed: float,
    readout_metadata: dict[str, Any],
) -> list[str]:
    return [
        (
            f"profile={profile_name} | date={run_datetime:%Y-%m-%d %H:%M:%S} | "
            f"qubits={len(qubit_names)} | states={','.join(states)} | "
            f"shots/state={num_shots:,} | emulation={do_emulation}"
        ),
        (
            f"active_reset_repetitions={active_reset_num} | "
            f"elapsed={elapsed:.1f}s | "
            f"readout_length={format_summary(readout_metadata['readout_pulse_length_s'], 's')} | "
            f"readout_amp={format_summary(readout_metadata['readout_amplitude'], '')}"
        ),
        (
            "readout_resonator="
            f"{format_summary(readout_metadata['readout_resonator_frequency_hz'], 'Hz')}"
        ),
    ]


def format_summary(summary: dict[str, Any], unit: str) -> str:
    common_value = summary.get("common")
    if common_value is not None:
        return f"{format_metric(common_value, unit)} (common)"

    min_value = summary.get("min")
    max_value = summary.get("max")
    mean_value = summary.get("mean")
    if min_value is None or max_value is None or mean_value is None:
        return "n/a"
    return (
        f"min={format_metric(min_value, unit)}, "
        f"mean={format_metric(mean_value, unit)}, "
        f"max={format_metric(max_value, unit)}"
    )


def format_metric(value: float, unit: str) -> str:
    if unit == "s":
        if abs(value) < 1e-6:
            return f"{value * 1e9:.1f} ns"
        if abs(value) < 1e-3:
            return f"{value * 1e6:.3g} us"
        return f"{value:.3g} s"
    if unit == "Hz":
        return f"{value * 1e-9:.6g} GHz"
    if unit:
        return f"{value:.6g} {unit}"
    return f"{value:.6g}"


def add_delta_labels(
    axis: Any,
    x_values: Any,
    active_fidelities: Any,
    no_active_fidelities: Any,
) -> None:
    import numpy as np

    y_min, y_max = axis.get_ylim()
    y_offset = 0.012 * (y_max - y_min)
    for x_value, active, no_active in zip(
        x_values,
        active_fidelities,
        no_active_fidelities,
    ):
        if np.isnan(active) or np.isnan(no_active):
            continue
        delta = active - no_active
        label = f"{delta:+.3f}"
        y_value = min(max(active, no_active) + y_offset, y_max - y_offset)
        axis.text(
            x_value,
            y_value,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f1f1f",
        )


def add_reference_lines(axis: Any, fidelities: np.ndarray) -> None:
    import numpy as np

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
    import numpy as np

    return max(0.5, float(np.min(fidelities)) - 0.04), 1.0


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "qubit",
                "condition",
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


def save_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "qubit",
                "fidelity_without_active_reset",
                "fidelity_with_active_reset",
                "fidelity_delta_active_minus_without",
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
    return sum(
        row["fidelity_without_active_reset"] is not None
        or row["fidelity_with_active_reset"] is not None
        for row in rows
    )


if __name__ == "__main__":
    main()
