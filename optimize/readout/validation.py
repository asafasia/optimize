from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from qratena.system.components_params.profile import Profile
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType, SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES

from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
    ReadoutFidelityWorkflowSettings,
)

if TYPE_CHECKING:
    from qigeon.io.task_submitter import TaskSubmitterAsync
else:
    TaskSubmitterAsync = Any


@dataclass(slots=True)
class ReadoutOptimizerValidationSettings:
    profile_name: str = "main"
    active_reset_num: int = 5
    task_status_poll_interval: float = 10.0
    do_emulation: bool = False
    show_handler_output: bool = False
    states: list[str] | None = None
    output_name: str = "main_profile_iq_sweep"


class ReadoutOptimizerValidation:
    """Validate optimizer amplitudes with main-profile readout workflow runs."""

    def __init__(
        self,
        *,
        optimizer_run_dir: str | Path,
        profile: Profile,
        task_manager: TaskSubmitterAsync,
        settings: ReadoutOptimizerValidationSettings | None = None,
    ) -> None:
        self.optimizer_run_dir = Path(optimizer_run_dir)
        self.profile = profile
        self.task_manager = task_manager
        self.settings = settings or ReadoutOptimizerValidationSettings()
        self.optimizer_summary = self._load_optimizer_summary()
        self.qubit_names = list(self.optimizer_summary["qubits"])
        self.amplitudes = [
            float(amplitude) for amplitude in self.optimizer_summary["amplitudes"]
        ]
        self.best_mean_amplitude = float(
            self.optimizer_summary["best_mean_amplitude"]
        )
        self.rows: list[dict[str, Any]] = []
        self.figure: Figure | None = None
        self.validation_dir: Path | None = None

    def run(self) -> dict[str, Any]:
        self.rows = []
        for amplitude in self.amplitudes:
            self.rows.extend(self._run_amplitude(amplitude))

        self.figure = plot_validation(
            rows=self.rows,
            qubit_names=self.qubit_names,
            best_amplitude=self.best_mean_amplitude,
        )
        self.validation_dir = save_validation_artifacts(
            run_dir=self.optimizer_run_dir,
            output_name=self.settings.output_name,
            rows=self.rows,
            figure=self.figure,
            profile_name=self.settings.profile_name,
            qubit_names=self.qubit_names,
            amplitudes=self.amplitudes,
            best_mean_amplitude=self.best_mean_amplitude,
        )
        return {
            "validation_dir": self.validation_dir,
            "rows": self.rows,
            "figure": self.figure,
        }

    def _load_optimizer_summary(self) -> dict[str, Any]:
        summary_path = self.optimizer_run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing optimizer summary: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def _run_amplitude(self, amplitude: float) -> list[dict[str, Any]]:
        profile = copy.deepcopy(self.profile)
        set_readout_amplitude(profile, self.qubit_names, amplitude)

        workflow = ReadoutFidelityWorkflow(
            qubit_names=self.qubit_names,
            profile=profile,
            task_manager=self.task_manager,
            settings=ReadoutFidelityWorkflowSettings(
                profile_name=self.settings.profile_name,
                do_emulation=self.settings.do_emulation,
                run_resonator=False,
                run_kernels=True,
                run_iq_blobs=True,
                do_plotting=False,
                show_handler_output=self.settings.show_handler_output,
                report_timing=True,
                task_status_poll_interval=self.settings.task_status_poll_interval,
                reset=ResetSettings(
                    reset_type=ResetType.ACTIVE,
                    reset_num=self.settings.active_reset_num,
                ),
                states=self.settings.states or ["g", "e"],
            ),
        )
        result = workflow.run()
        return extract_fidelity_rows(
            amplitude=amplitude,
            qubit_names=self.qubit_names,
            iq_blob_results=result["iq_blobs"],
        )


def set_readout_amplitude(
    profile: Profile,
    qubit_names: list[str],
    amplitude: float,
) -> None:
    for qubit_name in qubit_names:
        pulse = profile.qubits[qubit_name].pulses[SUPPORTED_PULSE_TYPES.readout][
            SUPPORTED_PULSE_SHAPES.const
        ]
        pulse.readout_amplitude = float(amplitude)


def extract_fidelity_rows(
    *,
    amplitude: float,
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
            ["separation", "readout_separation", "iq_separation", "state_separation"],
        )
        rows.append(
            {
                "amplitude": float(amplitude),
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


def plot_validation(
    *,
    rows: list[dict[str, Any]],
    qubit_names: list[str],
    best_amplitude: float,
) -> Figure:
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for qubit_name in qubit_names:
        qubit_rows = [row for row in rows if row["qubit"] == qubit_name]
        x_values = [row["amplitude"] for row in qubit_rows]
        y_values = [row["readout_fidelity"] for row in qubit_rows]
        axis.plot(x_values, y_values, marker="o", linewidth=1.4, label=qubit_name)

    axis.axvline(
        best_amplitude,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="optimizer best mean",
    )
    axis.set_title("Main-profile IQ blob validation")
    axis.set_xlabel("Readout amplitude")
    axis.set_ylabel("Readout fidelity")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    axis.legend(loc="best", fontsize=8)
    return fig


def save_validation_artifacts(
    *,
    run_dir: Path,
    output_name: str,
    rows: list[dict[str, Any]],
    figure: Figure,
    profile_name: str,
    qubit_names: list[str],
    amplitudes: list[float],
    best_mean_amplitude: float,
) -> Path:
    validation_dir = run_dir / "validation" / output_name
    validation_dir.mkdir(parents=True, exist_ok=True)

    csv_path = validation_dir / "readout_validation.csv"
    fieldnames = [
        "amplitude",
        "qubit",
        "readout_fidelity",
        "readout_fidelity_error",
        "separation",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_by_amplitude = validation_mean_fidelity_by_amplitude(rows, amplitudes)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_optimizer_run_dir": str(run_dir),
        "profile_name": profile_name,
        "qubits": qubit_names,
        "amplitudes": amplitudes,
        "optimizer_best_mean_amplitude": best_mean_amplitude,
        "validation_mean_fidelity_by_amplitude": mean_by_amplitude,
        "rows": rows,
    }
    (validation_dir / "readout_validation.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    figure.savefig(validation_dir / "readout_validation.png", dpi=200, bbox_inches="tight")
    (validation_dir / "readout_validation.md").write_text(
        validation_markdown_report(
            report=report,
            mean_by_amplitude=mean_by_amplitude,
        ),
        encoding="utf-8",
    )
    return validation_dir


def validation_mean_fidelity_by_amplitude(
    rows: list[dict[str, Any]],
    amplitudes: list[float],
) -> dict[str, float | None]:
    measured = [row for row in rows if row["readout_fidelity"] is not None]
    mean_by_amplitude = {}
    for amplitude in amplitudes:
        values = [
            float(row["readout_fidelity"])
            for row in measured
            if row["amplitude"] == amplitude
        ]
        mean_by_amplitude[str(amplitude)] = float(np.mean(values)) if values else None
    return mean_by_amplitude


def validation_markdown_report(
    *,
    report: dict[str, Any],
    mean_by_amplitude: dict[str, float | None],
) -> str:
    lines = [
        "# Main-Profile Readout Validation",
        "",
        f"Created at: {report['created_at']}",
        f"Source optimizer run: `{report['source_optimizer_run_dir']}`",
        f"Profile: `{report['profile_name']}`",
        f"Qubits: {', '.join(report['qubits'])}",
        f"Optimizer best mean amplitude: {report['optimizer_best_mean_amplitude']}",
        "",
        "## Mean Fidelity By Amplitude",
        "",
        "| Amplitude | Mean validation fidelity |",
        "| ---: | ---: |",
    ]
    for amplitude, mean_fidelity in mean_by_amplitude.items():
        lines.append(f"| {amplitude} | {mean_fidelity} |")

    lines.extend(
        [
            "",
            "## Saved Files",
            "",
            "- `readout_validation.csv`: one row per amplitude and qubit.",
            "- `readout_validation.json`: machine-readable validation report.",
            "- `readout_validation.png`: fidelity versus amplitude figure.",
        ]
    )
    return "\n".join(lines) + "\n"
