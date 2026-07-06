"""Run many CPMG experiments over a list of pi-pulse counts."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from laboneq.serializers import from_json
from laboneq.simple import AcquisitionType, AveragingMode
from qratena.experiments.base_experiment import ExperimentSettings
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    UpdateParamsMethod,
)

from measure_resonator_thermal_population.cpmg import (
    INTERPULSE_DELAY_SWEEP_START,
    NUM_SWEEP_POINTS,
    PROFILE_NAME,
    CPMGHandler,
)
from measure_resonator_thermal_population.decay_fit import SECONDS_TO_MICROSECONDS
from resources.load_profile import load_profile, load_task_manager


matplotlib.use("Agg")


DEFAULT_NUM_PI_PULSES = range(0, 17, 1)
# Alternative dense/sparse sweep: list(chain(range(10), range(10, 100, 10)))
DEFAULT_QUBITS = ["q3", "q7", "q8"]
DEFAULT_TARGET_EVOLUTION_TIME = 100e-6
DEFAULT_INTERPULSE_DELAY_STOP = 100e-6
OUTPUT_DIR = Path("outputs/cpmg_sweep")
TASK_IDS_FILENAME = "task_ids_by_n.json"


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run_cpmg_sweep(args: argparse.Namespace) -> list[dict[str, Any]]:
    _load_dotenv()
    _validate_args(args)
    output_dir = _make_output_dir(args)

    profile = load_profile(args.profile)
    task_manager = load_task_manager()

    settings = ExperimentSettings(
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
        update_params_method=UpdateParamsMethod.NONE,
        exportation_method=ExportationMethod.FULL,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        num_shots=args.num_shots,
        reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=args.reset_num),
    )

    task_records_by_n = _load_task_records_by_n(args, output_dir)
    if not args.acquire_only:
        task_records_by_n = submit_cpmg_tasks(
            args=args,
            output_dir=output_dir,
            task_manager=task_manager,
            profile=profile,
            settings=settings,
            task_records_by_n=task_records_by_n,
        )

    task_ids_path = _task_ids_output_path(args, output_dir)
    _write_task_records(task_ids_path, task_records_by_n)
    print(f"Saved {task_ids_path}")

    if args.submit_only:
        return []

    return acquire_cpmg_results(
        args=args,
        output_dir=output_dir,
        task_manager=task_manager,
        profile=profile,
        settings=settings,
        task_records_by_n=task_records_by_n,
    )


def submit_cpmg_tasks(
    args: argparse.Namespace,
    output_dir: Path,
    task_manager: Any,
    profile: Any,
    settings: ExperimentSettings,
    task_records_by_n: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    for num_pi_pulses in args.num_pi_pulses:
        if num_pi_pulses in task_records_by_n:
            task_id = task_records_by_n[num_pi_pulses]["task_id"]
            print(f"Reusing existing task for N={num_pi_pulses}: {task_id}")
            continue

        interpulse_delay_stop = _get_interpulse_delay_stop(args, num_pi_pulses)
        evolution_time_stop = _get_evolution_time_stop(interpulse_delay_stop, num_pi_pulses)
        handler = CPMGHandler(
            qubit_names=args.qubits,
            interpulse_delay_sweep_start=args.interpulse_delay_start,
            interpulse_delay_sweep_stop=interpulse_delay_stop,
            num_sweep_points=args.num_sweep_points,
            num_pi_pulses=num_pi_pulses,
            settings=settings,
            configuration_params=profile,
        )
        experiment_name = (
            f"{handler.experiment_name}_{num_pi_pulses}_pulses_"
            f"{args.sweep_mode}"
        )

        compiled_experiment = handler.get_compiled_experiment()
        print(
            f"Submitting {experiment_name}: "
            f"interpulse delay {args.interpulse_delay_start * SECONDS_TO_MICROSECONDS:.3f}"
            f"-{interpulse_delay_stop * SECONDS_TO_MICROSECONDS:.3f} us, "
            f"evolution to {evolution_time_stop * SECONDS_TO_MICROSECONDS:.3f} us"
        )
        task_id = task_manager.submit_compiled_experiment(
            experiment_name=experiment_name,
            profile_name=args.profile,
            qubit_names=handler.qubit_names,
            compiled_experiment=compiled_experiment,
            do_emulation=args.do_emulation,
        )
        print(f"Task submitted for N={num_pi_pulses}: {task_id}")
        task_records_by_n[num_pi_pulses] = {
            "task_id": str(task_id),
            "experiment_name": experiment_name,
            "profile_name": args.profile,
            "qubits": handler.qubit_names,
            "num_pi_pulses": num_pi_pulses,
            "num_sweep_points": args.num_sweep_points,
            "num_shots": args.num_shots,
            "sweep_mode": args.sweep_mode,
            "interpulse_delay_start_us": (
                args.interpulse_delay_start * SECONDS_TO_MICROSECONDS
            ),
            "interpulse_delay_stop_us": (
                interpulse_delay_stop * SECONDS_TO_MICROSECONDS
            ),
            "evolution_time_stop_us": evolution_time_stop * SECONDS_TO_MICROSECONDS,
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_task_records(_task_ids_output_path(args, output_dir), task_records_by_n)

    return task_records_by_n


def acquire_cpmg_results(
    args: argparse.Namespace,
    output_dir: Path,
    task_manager: Any,
    profile: Any,
    settings: ExperimentSettings,
    task_records_by_n: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for num_pi_pulses in args.num_pi_pulses:
        if num_pi_pulses not in task_records_by_n:
            print(f"Skipping N={num_pi_pulses}: no task_id")
            continue

        task_record = task_records_by_n[num_pi_pulses]
        task_id = task_record["task_id"]
        interpulse_delay_stop = _get_interpulse_delay_stop(args, num_pi_pulses)
        evolution_time_stop = _get_evolution_time_stop(interpulse_delay_stop, num_pi_pulses)
        handler = CPMGHandler(
            qubit_names=args.qubits,
            interpulse_delay_sweep_start=args.interpulse_delay_start,
            interpulse_delay_sweep_stop=interpulse_delay_stop,
            num_sweep_points=args.num_sweep_points,
            num_pi_pulses=num_pi_pulses,
            settings=settings,
            configuration_params=profile,
        )
        experiment_name = task_record.get(
            "experiment_name",
            f"{handler.experiment_name}_{num_pi_pulses}_pulses_{args.sweep_mode}",
        )

        print(f"Acquiring {experiment_name} result from task {task_id}")
        handler.define_experiment()
        task_result = task_manager.get_result(task_id)
        if task_result is None:
            print(f"Task N={num_pi_pulses} is not ready; skipping without waiting")
            continue

        handler.experiment_result = from_json(task_result.raw_data)
        handler.analysis_result = handler.analyze()
        handler.figs = handler.plot()

        for index, fig in enumerate(handler.figs):
            qubit_name = handler.qubit_names[index]
            fig_path = output_dir / f"cpmg_{qubit_name}_N{num_pi_pulses}.png"
            fig.savefig(fig_path, dpi=180, bbox_inches="tight")
            print(f"Saved {fig_path}")
            plt.close(fig)

        for qubit_name in handler.qubit_names:
            qubit_data = handler.data[qubit_name]
            fitted_t2 = qubit_data["fitted_t2_cpmg"]
            fitted_t2_stderr = qubit_data["fitted_t2_cpmg_stderr"]
            summary_entry = {
                "task_id": task_id,
                "experiment_name": experiment_name,
                "profile_name": args.profile,
                "qubit": qubit_name,
                "num_pi_pulses": num_pi_pulses,
                "num_sweep_points": args.num_sweep_points,
                "num_shots": args.num_shots,
                "interpulse_delay_start_us": (
                    args.interpulse_delay_start * SECONDS_TO_MICROSECONDS
                ),
                "interpulse_delay_stop_us": (
                    interpulse_delay_stop * SECONDS_TO_MICROSECONDS
                ),
                "evolution_time_stop_us": (
                    evolution_time_stop * SECONDS_TO_MICROSECONDS
                ),
                "fitted_t2_cpmg_us": (
                    fitted_t2 * SECONDS_TO_MICROSECONDS if np.isfinite(fitted_t2) else None
                ),
                "fitted_t2_cpmg_stderr_us": (
                    fitted_t2_stderr * SECONDS_TO_MICROSECONDS
                    if np.isfinite(fitted_t2_stderr)
                    else None
                ),
                "fit_r2": qubit_data["fit_r2"],
                "fit_score": qubit_data["fit_score"],
                "contrast_estimate": qubit_data["contrast_estimate"],
            }
            summary.append(summary_entry)

            data_path = output_dir / f"cpmg_{qubit_name}_N{num_pi_pulses}.json"
            data_path.write_text(json.dumps(_json_safe(qubit_data), indent=2))
            print(f"Saved {data_path}")

    if summary:
        decay_plot_path = output_dir / "cpmg_t2_vs_n.png"
        _plot_decay_time_vs_n(summary, decay_plot_path, profile)
        print(f"Saved {decay_plot_path}")

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(_json_safe(summary), indent=2))
        print(f"Saved {summary_path}")
    else:
        print("No ready CPMG results found; leaving summary and T2 plot unchanged")
    return summary


def _make_output_dir(args: argparse.Namespace) -> Path:
    if args.acquire_only and args.task_ids_path is not None and args.run_name is None:
        output_dir = Path(args.task_ids_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Writing outputs to {output_dir}")
        return output_dir

    base_output_dir = Path(args.output_dir)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / run_name
    if args.acquire_only and output_dir.exists():
        print(f"Writing outputs to {output_dir}")
        return output_dir

    suffix = 1
    while output_dir.exists():
        output_dir = base_output_dir / f"{run_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    print(f"Writing outputs to {output_dir}")
    return output_dir


def _plot_decay_time_vs_n(
    summary: list[dict[str, Any]],
    output_path: Path,
    profile: Any | None = None,
) -> None:
    by_qubit: dict[str, list[dict[str, Any]]] = {}
    for entry in summary:
        if entry["fitted_t2_cpmg_us"] is None:
            continue
        by_qubit.setdefault(entry["qubit"], []).append(entry)

    if not by_qubit:
        return

    fig, axes = plt.subplots(
        len(by_qubit),
        1,
        figsize=(7.5, 3.2 * len(by_qubit)),
        sharex=True,
        squeeze=False,
    )
    for ax, (qubit_name, entries) in zip(axes[:, 0], sorted(by_qubit.items())):
        sorted_entries = sorted(entries, key=lambda item: item["num_pi_pulses"])
        num_pi_pulses = np.array(
            [entry["num_pi_pulses"] for entry in sorted_entries],
            dtype=float,
        )
        fitted_t2_us = np.array(
            [entry["fitted_t2_cpmg_us"] for entry in sorted_entries],
            dtype=float,
        )
        fitted_t2_stderr_us = np.array(
            [
                entry.get("fitted_t2_cpmg_stderr_us", np.nan)
                for entry in sorted_entries
            ],
            dtype=float,
        )
        valid = (num_pi_pulses > 0) & np.isfinite(fitted_t2_us) & (fitted_t2_us > 0)
        if not np.any(valid):
            ax.set_title(f"{qubit_name}: no positive N points for log-log plot")
            continue

        plot_n = num_pi_pulses[valid]
        plot_t2_us = fitted_t2_us[valid]
        plot_t2_stderr_us = fitted_t2_stderr_us[valid]
        yerr = _positive_log_yerr(plot_t2_us, plot_t2_stderr_us)
        ax.errorbar(
            plot_n,
            plot_t2_us,
            yerr=yerr,
            fmt="o-",
            capsize=3,
            label="CPMG fit",
        )
        _plot_log_log_fit(ax, plot_n, plot_t2_us)
        _plot_profile_reference_lines(ax, profile, qubit_name)

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylabel("Fitted T2 CPMG [us]")
        ax.set_title(f"CPMG decay time vs N - {qubit_name}")
        _set_qubit_ylim(ax, profile, qubit_name, plot_t2_us)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    axes[-1, 0].set_xlabel("Number of pi pulses, N")
    fig.suptitle("CPMG T2 vs N (log-log, N=0 omitted)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_log_log_fit(
    ax: plt.Axes,
    num_pi_pulses: np.ndarray,
    fitted_t2_us: np.ndarray,
) -> None:
    if len(num_pi_pulses) < 2:
        return

    log_n = np.log10(num_pi_pulses)
    log_t2 = np.log10(fitted_t2_us)
    slope, intercept = np.polyfit(log_n, log_t2, 1)
    fit_t2_us = 10 ** (intercept + slope * log_n)
    residuals = log_t2 - (intercept + slope * log_n)
    total = log_t2 - np.mean(log_t2)
    r2 = 1.0 - np.sum(residuals**2) / np.sum(total**2) if np.sum(total**2) > 0 else np.nan
    ax.plot(
        num_pi_pulses,
        fit_t2_us,
        "--",
        label=f"log fit: slope={slope:.2f}, R2={r2:.2f}",
    )


def _positive_log_yerr(
    y_values: np.ndarray,
    y_stderr: np.ndarray,
) -> np.ndarray | None:
    valid_stderr = np.isfinite(y_stderr) & (y_stderr > 0)
    if not np.any(valid_stderr):
        return None

    upper = np.where(valid_stderr, y_stderr, 0.0)
    lower = np.where(valid_stderr, np.minimum(y_stderr, y_values * 0.95), 0.0)
    return np.vstack((lower, upper))


def _plot_profile_reference_lines(
    ax: plt.Axes,
    profile: Any | None,
    qubit_name: str,
) -> None:
    t1_us = _profile_qubit_time_us(profile, qubit_name, "t1")
    t2_ramsey_us = _profile_qubit_time_us(profile, qubit_name, "t2_ramsey")
    if t1_us is not None:
        ax.axhline(
            2.0 * t1_us,
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            label=f"2*T1={2.0 * t1_us:.1f} us",
        )
    if t2_ramsey_us is not None:
        ax.axhline(
            t2_ramsey_us,
            color="tab:green",
            linestyle="-.",
            linewidth=1.2,
            label=f"T2 Ramsey={t2_ramsey_us:.1f} us",
        )


def _set_qubit_ylim(
    ax: plt.Axes,
    profile: Any | None,
    qubit_name: str,
    fitted_t2_us: np.ndarray,
) -> None:
    positive_t2 = fitted_t2_us[np.isfinite(fitted_t2_us) & (fitted_t2_us > 0)]
    if len(positive_t2) == 0:
        return

    lower = max(float(np.min(positive_t2)) * 0.8, 1e-3)
    t1_us = _profile_qubit_time_us(profile, qubit_name, "t1")
    if t1_us is None:
        ax.set_ylim(bottom=lower)
        return

    upper = max(2.0 * t1_us * 1.12, lower * 1.2)
    ax.set_ylim(lower, upper)


def _profile_qubit_time_us(
    profile: Any | None,
    qubit_name: str,
    field_name: str,
) -> float | None:
    if profile is None or not hasattr(profile, "qubits"):
        return None
    qubit = profile.qubits.get(qubit_name)
    if qubit is None or not hasattr(qubit, field_name):
        return None

    value = getattr(qubit, field_name)
    value = getattr(value, "value", value)
    try:
        time_s = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(time_s) or time_s <= 0:
        return None
    return time_s * SECONDS_TO_MICROSECONDS


def _get_interpulse_delay_stop(args: argparse.Namespace, num_pi_pulses: int) -> float:
    if args.sweep_mode == "total-evolution":
        if num_pi_pulses == 0:
            return args.target_evolution_time
        return args.target_evolution_time / num_pi_pulses
    return args.interpulse_delay_stop


def _get_evolution_time_stop(interpulse_delay_stop: float, num_pi_pulses: int) -> float:
    if num_pi_pulses == 0:
        return interpulse_delay_stop
    return interpulse_delay_stop * num_pi_pulses


def _validate_args(args: argparse.Namespace) -> None:
    if args.submit_only and args.acquire_only:
        raise ValueError("--submit-only and --acquire-only cannot be used together")
    for num_pi_pulses in args.num_pi_pulses:
        interpulse_delay_stop = _get_interpulse_delay_stop(args, num_pi_pulses)
        if interpulse_delay_stop < args.interpulse_delay_start:
            raise ValueError(
                f"N={num_pi_pulses} gives interpulse delay stop "
                f"{interpulse_delay_stop * SECONDS_TO_MICROSECONDS:.3f} us, below start "
                f"{args.interpulse_delay_start * SECONDS_TO_MICROSECONDS:.3f} us"
            )


def _task_ids_output_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.task_ids_path is not None:
        return Path(args.task_ids_path)
    return output_dir / TASK_IDS_FILENAME


def _load_task_records_by_n(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[int, dict[str, Any]]:
    records_by_n: dict[int, dict[str, Any]] = {}
    for num_pi_pulses, task_id in _parse_task_ids_by_n(args.task_ids_by_n).items():
        records_by_n[num_pi_pulses] = {
            "task_id": task_id,
            "num_pi_pulses": num_pi_pulses,
        }

    path = _task_ids_output_path(args, output_dir)
    if not path.exists() and args.task_ids_path is None:
        return records_by_n

    if not path.exists():
        if args.acquire_only:
            raise FileNotFoundError(f"Task IDs file not found: {path}")
        return records_by_n

    loaded = json.loads(path.read_text())
    if isinstance(loaded, dict) and "tasks" in loaded:
        loaded_records = loaded["tasks"]
    else:
        loaded_records = loaded

    for key, value in loaded_records.items():
        if isinstance(value, dict):
            record = dict(value)
            num_pi_pulses = int(record.get("num_pi_pulses", key))
            record["num_pi_pulses"] = num_pi_pulses
        else:
            num_pi_pulses = int(key)
            record = {
                "task_id": str(value),
                "num_pi_pulses": num_pi_pulses,
            }
        records_by_n[num_pi_pulses] = record
    return records_by_n


def _write_task_records(path: Path, records_by_n: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tasks": {
            str(num_pi_pulses): record
            for num_pi_pulses, record in sorted(records_by_n.items())
        }
    }
    path.write_text(json.dumps(_json_safe(payload), indent=2))


def _parse_task_ids_by_n(values: list[str]) -> dict[int, str]:
    task_ids_by_n: dict[int, str] = {}
    for value in values:
        num_pi_pulses, task_id = value.split("=", 1)
        task_ids_by_n[int(num_pi_pulses)] = task_id
    return task_ids_by_n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=PROFILE_NAME)
    parser.add_argument("--qubits", nargs="+", default=DEFAULT_QUBITS)
    parser.add_argument("--num-pi-pulses", nargs="+", type=int, default=DEFAULT_NUM_PI_PULSES)
    parser.add_argument(
        "--sweep-mode",
        choices=["total-evolution", "interpulse-delay"],
        default="total-evolution",
        help=(
            "total-evolution keeps N * interpulse_delay_stop fixed; "
            "interpulse-delay keeps interpulse_delay_stop fixed for every N."
        ),
    )
    parser.add_argument(
        "--target-evolution-time",
        type=float,
        default=DEFAULT_TARGET_EVOLUTION_TIME,
    )
    parser.add_argument(
        "--interpulse-delay-start",
        type=float,
        default=INTERPULSE_DELAY_SWEEP_START,
    )
    parser.add_argument(
        "--interpulse-delay-stop",
        type=float,
        default=DEFAULT_INTERPULSE_DELAY_STOP,
    )
    parser.add_argument("--num-sweep-points", type=int, default=NUM_SWEEP_POINTS)
    parser.add_argument("--num-shots", type=int, default=2500)
    parser.add_argument("--reset-num", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name. If it already exists, a numeric suffix is added.",
    )
    parser.add_argument("--do-emulation", action="store_true")
    parser.add_argument(
        "--task-ids-by-n",
        nargs="*",
        default=[],
        help="Existing task IDs keyed as N=task-id, e.g. 1=uuid.",
    )
    parser.add_argument(
        "--task-ids-path",
        type=Path,
        default=None,
        help=(
            "JSON file for task IDs. Submit mode writes it; acquire-only mode "
            "reads it. Defaults to task_ids_by_n.json inside the output run."
        ),
    )
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit all missing CPMG tasks, save task IDs, and exit without reading results.",
    )
    parser.add_argument(
        "--acquire-only",
        action="store_true",
        help="Read task IDs and download ready results without submitting new tasks or waiting.",
    )
    args, _unknown_args = parser.parse_known_args()
    return args


def main() -> None:
    run_cpmg_sweep(parse_args())


if __name__ == "__main__":
    main()
