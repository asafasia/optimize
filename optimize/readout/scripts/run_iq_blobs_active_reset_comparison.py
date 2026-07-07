from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[3]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from qigeon.io.task_submitter import TaskSubmitterAsync
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType, SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES

from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
    ReadoutFidelityWorkflowSettings,
)
from resources.load_profile import load_profile, load_task_manager


PROFILE_NAME = "main"
OUTPUT_ROOT = Path("data/readout_iq_blobs_active_reset_comparison")

CHOSEN_QUBITS: list[str] = []
ACTIVE_RESET_NUM = 5
DO_EMULATION = False
SHOW_HANDLER_OUTPUT = False
TASK_STATUS_POLL_INTERVAL = 10.0


@dataclass(slots=True)
class ActiveResetIQBlobsComparisonSettings:
    qubit_names: list[str] = field(default_factory=list)
    profile_name: str = PROFILE_NAME
    output_root: Path = OUTPUT_ROOT
    active_reset_num: int = ACTIVE_RESET_NUM
    do_emulation: bool = DO_EMULATION
    show_handler_output: bool = SHOW_HANDLER_OUTPUT
    task_status_poll_interval: float = TASK_STATUS_POLL_INTERVAL
    show_plot: bool = False
    readout_pulse_shape: SUPPORTED_PULSE_SHAPES = SUPPORTED_PULSE_SHAPES.const


class ActiveResetIQBlobsComparison:
    def __init__(
        self,
        profile: Profile,
        task_manager: TaskSubmitterAsync,
        settings: ActiveResetIQBlobsComparisonSettings | None = None,
    ) -> None:
        self.profile = profile
        self.task_manager = task_manager
        self.settings = settings or ActiveResetIQBlobsComparisonSettings()
        self.qubit_names = sorted(self._selected_qubits(),key=lambda q: int(q[1:]))
        self.no_reset_rows: list[dict[str, Any]] = []
        self.active_reset_rows: list[dict[str, Any]] = []
        self.report_rows: list[dict[str, Any]] = []
        self.run_dir: Path | None = None
        self.figure: Figure | None = None
        self.run_datetime = datetime.now()

    def run(self) -> dict[str, Any]:
        self.no_reset_rows = self._run_iq_blobs_measurement(
            reset=ResetSettings(),
            condition="without_active_reset",
        )
        self.active_reset_rows = self._run_iq_blobs_measurement(
            reset=ResetSettings(ResetType.ACTIVE, reset_num=self.settings.active_reset_num),
            condition="with_active_reset",
        )
        self.report_rows = self._build_report_rows()
        self.figure = self._plot_readout_fidelity_comparison()
        self.run_dir = self.save()

        if self.settings.show_plot:
            plt.show()
        elif self.figure is not None:
            plt.close(self.figure)

        return {
            "run_dir": self.run_dir,
            "no_reset_rows": self.no_reset_rows,
            "active_reset_rows": self.active_reset_rows,
            "report_rows": self.report_rows,
            "figure": self.figure,
        }

    def save(self) -> Path:
        run_dir = make_run_dir(
            self.settings.output_root,
            self.qubit_names,
            self.run_datetime,
        )
        all_rows = self.no_reset_rows + self.active_reset_rows

        save_dict_csv(
            run_dir / "readout_fidelities_active_reset_comparison.csv",
            all_rows,
            [
                "qubit",
                "condition",
                "readout_fidelity",
                "readout_fidelity_error",
                "separation",
                "status",
            ],
        )
        save_dict_csv(
            run_dir / "readout_report.csv",
            self.report_rows,
            [
                "qubit",
                "readout_resonator_frequency_hz",
                "readout_pulse_length_s",
                "readout_amplitude",
                "fidelity_without_active_reset",
                "fidelity_without_active_reset_error",
                "fidelity_with_active_reset",
                "fidelity_with_active_reset_error",
                "fidelity_delta_active_minus_no_reset",
                "separation_without_active_reset",
                "separation_with_active_reset",
            ],
        )
        save_json(run_dir / "readout_report.json", self.report_rows)
        save_markdown_report(
            run_dir / "readout_report.md",
            self.report_rows,
            self.settings.active_reset_num,
        )

        if self.figure is not None:
            self.figure.savefig(
                run_dir / "readout_fidelities_active_reset_comparison.png",
                dpi=200,
                bbox_inches="tight",
            )

        return run_dir

    def _selected_qubits(self) -> list[str]:
        if self.settings.qubit_names:
            return self.settings.qubit_names
        if CHOSEN_QUBITS:
            return CHOSEN_QUBITS
        return sorted(self.profile.qubits.keys(), key=qubit_sort_key)

    def _run_iq_blobs_measurement(
        self,
        reset: ResetSettings,
        condition: str,
    ) -> list[dict[str, Any]]:
        settings = ReadoutFidelityWorkflowSettings(
            profile_name=self.settings.profile_name,
            do_emulation=self.settings.do_emulation,
            run_resonator=False,
            run_kernels=True,
            run_iq_blobs=True,
            do_plotting=False,
            show_handler_output=self.settings.show_handler_output,
            report_timing=True,
            task_status_poll_interval=self.settings.task_status_poll_interval,
            reset=reset,
        )

        workflow = ReadoutFidelityWorkflow(
            qubit_names=self.qubit_names,
            profile=self.profile,
            task_manager=self.task_manager,
            settings=settings,
        )
        result = workflow.run()
        return extract_fidelity_rows(self.qubit_names, result["iq_blobs"], condition)

    def _build_report_rows(self) -> list[dict[str, Any]]:
        no_reset_by_qubit = rows_by_qubit(self.no_reset_rows)
        active_reset_by_qubit = rows_by_qubit(self.active_reset_rows)
        rows = []

        for qubit_name in self.qubit_names:
            readout_params = self._readout_params(qubit_name)
            no_reset_row = no_reset_by_qubit.get(qubit_name, {})
            active_reset_row = active_reset_by_qubit.get(qubit_name, {})
            no_reset_fidelity = optional_float(no_reset_row.get("readout_fidelity"))
            active_reset_fidelity = optional_float(active_reset_row.get("readout_fidelity"))

            rows.append(
                {
                    "qubit": qubit_name,
                    **readout_params,
                    "fidelity_without_active_reset": no_reset_fidelity,
                    "fidelity_without_active_reset_error": optional_float(
                        no_reset_row.get("readout_fidelity_error")
                    ),
                    "fidelity_with_active_reset": active_reset_fidelity,
                    "fidelity_with_active_reset_error": optional_float(
                        active_reset_row.get("readout_fidelity_error")
                    ),
                    "fidelity_delta_active_minus_no_reset": optional_difference(
                        active_reset_fidelity,
                        no_reset_fidelity,
                    ),
                    "separation_without_active_reset": optional_float(
                        no_reset_row.get("separation")
                    ),
                    "separation_with_active_reset": optional_float(
                        active_reset_row.get("separation")
                    ),
                }
            )

        return rows

    def _readout_params(self, qubit_name: str) -> dict[str, float | None]:
        qubit = self.profile.qubits[qubit_name]
        readout_pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][
            self.settings.readout_pulse_shape
        ]
        frequency = getattr(qubit.readout_resonator_frequency, "value", None)

        return {
            "readout_resonator_frequency_hz": optional_float(frequency),
            "readout_pulse_length_s": optional_float(readout_pulse.readout_duration),
            "readout_amplitude": optional_float(readout_pulse.readout_amplitude),
        }

    def _plot_readout_fidelity_comparison(self) -> Figure:
        return plot_readout_fidelity_comparison(
            qubit_names=self.qubit_names,
            no_reset_rows=self.no_reset_rows,
            active_reset_rows=self.active_reset_rows,
            active_reset_num=self.settings.active_reset_num,
            run_datetime=self.run_datetime,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IQ blobs once without active reset and once with active reset, "
            "then plot and report the readout fidelities per qubit."
        )
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        help="Qubits to measure. Defaults to CHOSEN_QUBITS, then every profile qubit.",
    )
    parser.add_argument("--profile-name", default=PROFILE_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where the run folder, plot, CSV, and report are saved.",
    )
    parser.add_argument(
        "--active-reset-num",
        type=int,
        default=ACTIVE_RESET_NUM,
        help="Number of active reset repetitions for the active-reset measurement.",
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
    condition: str,
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
                "condition": condition,
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


def optional_difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def first_optional_float(data: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = optional_float(data.get(key))
        if value is not None:
            return value
    return None


def make_run_dir(
    output_root: Path,
    qubit_names: list[str],
    run_datetime: datetime | None = None,
) -> Path:
    timestamp = (run_datetime or datetime.now()).strftime("%Y%m%d_%H%M%S")
    qubit_label = "all_qubits" if len(qubit_names) > 3 else "_".join(qubit_names)
    run_dir = output_root / f"{timestamp}_{qubit_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_dict_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8")


def save_markdown_report(
    path: Path,
    rows: list[dict[str, Any]],
    active_reset_num: int,
) -> None:
    lines = [
        "# IQ Blobs Active Reset Comparison",
        "",
        f"Active reset repetitions: {active_reset_num}",
        "",
        "| Qubit | Resonator freq (GHz) | Length (ns) | Amplitude | Fidelity no reset | Fidelity active reset | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"| {row['qubit']}",
                    format_frequency_ghz(row["readout_resonator_frequency_hz"]),
                    format_seconds_ns(row["readout_pulse_length_s"]),
                    format_optional(row["readout_amplitude"], digits=6),
                    format_optional(row["fidelity_without_active_reset"], digits=4),
                    format_optional(row["fidelity_with_active_reset"], digits=4),
                    format_optional(row["fidelity_delta_active_minus_no_reset"], digits=4),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_readout_fidelity_comparison(
    qubit_names: list[str],
    no_reset_rows: list[dict[str, Any]],
    active_reset_rows: list[dict[str, Any]],
    active_reset_num: int,
    run_datetime: datetime | None = None,
) -> Figure:
    no_reset_by_qubit = rows_by_qubit(no_reset_rows)
    active_reset_by_qubit = rows_by_qubit(active_reset_rows)

    no_reset_fidelities = values_for_qubits(no_reset_by_qubit, qubit_names)
    active_reset_fidelities = values_for_qubits(active_reset_by_qubit, qubit_names)
    no_reset_errors = errors_for_qubits(no_reset_by_qubit, qubit_names)
    active_reset_errors = errors_for_qubits(active_reset_by_qubit, qubit_names)

    measured_values = [
        value
        for value in [*no_reset_fidelities, *active_reset_fidelities]
        if value is not None
    ]
    if not measured_values:
        raise ValueError("No readout_fidelity values were found in the IQ blobs results.")

    fig, axis = plt.subplots(
        figsize=(max(10.5, len(qubit_names) * 0.48), 6.0),
    )
    x_values = np.arange(len(qubit_names))
    width = 0.38

    axis.bar(
        x_values - width / 2,
        nan_for_missing(no_reset_fidelities),
        width=width,
        yerr=zero_for_missing(no_reset_errors),
        color="#6a8caf",
        edgecolor="black",
        linewidth=0.7,
        capsize=3,
        label="without active reset",
    )
    axis.bar(
        x_values + width / 2,
        nan_for_missing(active_reset_fidelities),
        width=width,
        yerr=zero_for_missing(active_reset_errors),
        color="#d49a3a",
        edgecolor="black",
        linewidth=0.7,
        capsize=3,
        label=f"with active reset ({active_reset_num}x)",
    )

    mean_no_reset = mean_optional(no_reset_fidelities)
    mean_active_reset = mean_optional(active_reset_fidelities)
    if mean_no_reset is not None:
        axis.axhline(
            mean_no_reset,
            color="#315c7c",
            linewidth=1.1,
            alpha=0.85,
            label="mean without active reset",
        )
    if mean_active_reset is not None:
        axis.axhline(
            mean_active_reset,
            color="#9a681d",
            linewidth=1.1,
            alpha=0.85,
            label="mean with active reset",
        )

    axis.axhline(
        0.90,
        color="#b23a48",
        linestyle="--",
        linewidth=1.0,
        label="0.90 reference",
    )
    axis.axhline(
        0.95,
        color="#2f7d5c",
        linestyle="--",
        linewidth=1.0,
        label="0.95 reference",
    )
    axis.set_title("Readout fidelity with and without active reset")
    axis.set_ylabel("Readout fidelity")
    axis.set_xticks(x_values)
    axis.set_xticklabels(qubit_names, rotation=60, ha="right")
    axis.set_ylim(fidelity_axis_limits(measured_values))
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    title_parts = []
    if mean_no_reset is not None:
        title_parts.append(f"without reset mean={mean_no_reset:.4f}")
    if mean_active_reset is not None:
        title_parts.append(f"active reset mean={mean_active_reset:.4f}")
    if title_parts:
        timestamp = ""
        if run_datetime is not None:
            timestamp = f" | {run_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
        fig.suptitle(" | ".join(title_parts) + timestamp, y=1.02)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def rows_by_qubit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["qubit"]): row for row in rows}


def values_for_qubits(
    rows: dict[str, dict[str, Any]],
    qubit_names: list[str],
) -> list[float | None]:
    return [
        optional_float(rows.get(qubit_name, {}).get("readout_fidelity"))
        for qubit_name in qubit_names
    ]


def errors_for_qubits(
    rows: dict[str, dict[str, Any]],
    qubit_names: list[str],
) -> list[float | None]:
    return [
        optional_float(rows.get(qubit_name, {}).get("readout_fidelity_error"))
        for qubit_name in qubit_names
    ]


def nan_for_missing(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if value is None else value for value in values], dtype=float)


def zero_for_missing(values: list[float | None]) -> np.ndarray:
    return np.array([0.0 if value is None else value for value in values], dtype=float)


def mean_optional(values: list[float | None]) -> float | None:
    measured_values = [value for value in values if value is not None]
    if not measured_values:
        return None
    return float(np.mean(measured_values))


def fidelity_axis_limits(values: list[float]) -> tuple[float, float]:
    lower = max(0.5, min(values) - 0.04)
    return lower, 1.0


def format_frequency_ghz(value: Any) -> str:
    frequency = optional_float(value)
    if frequency is None:
        return ""
    return f"{frequency / 1e9:.6f}"


def format_seconds_ns(value: Any) -> str:
    seconds = optional_float(value)
    if seconds is None:
        return ""
    return f"{seconds * 1e9:.1f}"


def format_optional(value: Any, digits: int) -> str:
    number = optional_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


if __name__ == "__main__":
    args = parse_args()
    
    profile = load_profile()
    profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)
    
    # profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

    
    # prepare qubit list and exclude q20
    qubits = [q for q in profile.qubits.keys() if q != "q20"]
    
    
    qubits = ['q1','q3','q4']
    settings = ActiveResetIQBlobsComparisonSettings(
        qubit_names=['q5','q6','q9'],
        profile_name=args.profile_name,
        output_root=args.output_root,
        active_reset_num=5,
        do_emulation=args.do_emulation,
        show_handler_output=args.show_handler_output,
        task_status_poll_interval=args.task_status_poll_interval,
        show_plot=args.show,
    )

    runner = ActiveResetIQBlobsComparison(
        profile=profile,
        task_manager=load_task_manager(),
        settings=settings,
    )
    result = runner.run()

    print(f"Measured {len(runner.qubit_names)} qubits twice")
    print(f"Saved comparison and report to {result['run_dir']}")
