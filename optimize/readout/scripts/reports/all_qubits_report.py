from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

WORKBENCH_ROOT = Path(__file__).resolve().parents[4]
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
OUTPUT_ROOT = Path("data/readout_all_qubits_report")


@dataclass(slots=True)
class AllQubitsReadoutReportSettings:
    qubit_names: list[str] = field(default_factory=list)
    profile_name: str = PROFILE_NAME
    output_root: Path = OUTPUT_ROOT
    active_reset_num: int = 5
    states: list[str] = field(default_factory=lambda: ["g", "e"])
    do_emulation: bool = False
    show_handler_output: bool = False
    task_status_poll_interval: float = 10.0
    run_resonator: bool = True
    run_kernels: bool = True
    run_passive_comparison: bool = True
    continue_on_error: bool = True
    show_plots: bool = False
    readout_pulse_shape: SUPPORTED_PULSE_SHAPES = SUPPORTED_PULSE_SHAPES.const


class AllQubitsReadoutReport:
    """Run the readout workflow for every selected qubit and save an HTML report."""

    def __init__(
        self,
        profile: Profile,
        task_manager: TaskSubmitterAsync,
        settings: AllQubitsReadoutReportSettings | None = None,
    ) -> None:
        self.profile = profile
        self.task_manager = task_manager
        self.settings = settings or AllQubitsReadoutReportSettings()
        selected = self.settings.qubit_names or list(profile.qubits)
        self.qubit_names = sorted(selected, key=qubit_sort_key)
        self.run_datetime = datetime.now()
        self.run_dir = make_run_dir(
            self.settings.output_root,
            self.qubit_names,
            self.run_datetime,
        )
        self.rows: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, dict[str, str]] = {}
        self.image_paths: dict[str, dict[str, list[Path]]] = {}
        self.summary_images: list[Path] = []

    def run(self) -> dict[str, Any]:
        for index, qubit_name in enumerate(self.qubit_names, start=1):
            print(f"[all-qubits report] {index}/{len(self.qubit_names)}: {qubit_name}")
            active_result = self._run_condition(
                qubit_name=qubit_name,
                condition="active",
                reset=ResetSettings(
                    ResetType.ACTIVE,
                    reset_num=self.settings.active_reset_num,
                ),
                run_full_workflow=True,
            )
            passive_result = None
            if self.settings.run_passive_comparison:
                passive_result = self._run_condition(
                    qubit_name=qubit_name,
                    condition="passive",
                    reset=ResetSettings(reset_type=ResetType.PASSIVE),
                    run_full_workflow=False,
                )

            self.rows.append(
                self._build_row(qubit_name, active_result, passive_result)
            )

        self._save()
        if self.settings.show_plots:
            plt.show()
        else:
            plt.close("all")

        return {
            "run_dir": self.run_dir,
            "rows": self.rows,
            "results": self.results,
            "errors": self.errors,
        }

    def _run_condition(
        self,
        qubit_name: str,
        condition: str,
        reset: ResetSettings,
        run_full_workflow: bool,
    ) -> dict[str, Any] | None:
        workflow = ReadoutFidelityWorkflow(
            qubit_names=[qubit_name],
            profile=self.profile,
            task_manager=self.task_manager,
            settings=ReadoutFidelityWorkflowSettings(
                profile_name=self.settings.profile_name,
                do_emulation=self.settings.do_emulation,
                run_resonator=self.settings.run_resonator and run_full_workflow,
                run_kernels=self.settings.run_kernels and run_full_workflow,
                run_iq_blobs=True,
                do_plotting=False,
                show_handler_output=self.settings.show_handler_output,
                report_timing=True,
                task_status_poll_interval=self.settings.task_status_poll_interval,
                reset=reset,
                states=self.settings.states,
            ),
        )

        try:
            result = workflow.run()
        except Exception as error:
            self.errors.setdefault(qubit_name, {})[condition] = (
                f"{type(error).__name__}: {error}"
            )
            print(
                f"[all-qubits report] {qubit_name} {condition} failed: "
                f"{type(error).__name__}: {error}"
            )
            if not self.settings.continue_on_error:
                raise
            return None

        self.results.setdefault(qubit_name, {})[condition] = result
        self._save_workflow_figures(qubit_name, condition, workflow)
        return result

    def _save_workflow_figures(
        self,
        qubit_name: str,
        condition: str,
        workflow: ReadoutFidelityWorkflow,
    ) -> None:
        handlers = {
            "resonator": workflow.resonator_handler,
            "kernel": workflow.kernel_handler,
            "iq_blobs": workflow.iq_blobs_handler,
        }
        for experiment, handler in handlers.items():
            figures = getattr(handler, "workflow_figures", []) if handler else []
            for figure_index, figure in enumerate(figures, start=1):
                if not isinstance(figure, Figure):
                    continue
                relative_path = (
                    Path("figures")
                    / qubit_name
                    / condition
                    / f"{experiment}_{figure_index:02d}.png"
                )
                path = self.run_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    figure.savefig(path, dpi=180, bbox_inches="tight")
                except Exception as error:
                    error_key = f"{condition}_{experiment}_figure_{figure_index}"
                    self.errors.setdefault(qubit_name, {})[error_key] = (
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                self.image_paths.setdefault(qubit_name, {}).setdefault(
                    f"{condition}_{experiment}", []
                ).append(relative_path)

    def _build_row(
        self,
        qubit_name: str,
        active_result: dict[str, Any] | None,
        passive_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        active_iq = qubit_result(active_result, "iq_blobs", qubit_name)
        passive_iq = qubit_result(passive_result, "iq_blobs", qubit_name)
        resonator = qubit_result(active_result, "resonator", qubit_name)
        kernel = qubit_result(active_result, "kernels", qubit_name)
        pulse_params = self._pulse_params(qubit_name)
        active_fidelity = first_float(
            active_iq, ["readout_fidelity", "average_readout_fidelity"]
        )
        passive_fidelity = first_float(
            passive_iq, ["readout_fidelity", "average_readout_fidelity"]
        )

        return {
            "qubit": qubit_name,
            **pulse_params,
            "active_fidelity": active_fidelity,
            "active_fidelity_error": fidelity_error(active_iq),
            "active_separation": separation(active_iq),
            "passive_fidelity": passive_fidelity,
            "passive_fidelity_error": fidelity_error(passive_iq),
            "passive_separation": separation(passive_iq),
            "fidelity_delta_active_minus_passive": optional_difference(
                active_fidelity, passive_fidelity
            ),
            "optimal_resonator_frequency_hz": first_float(
                resonator,
                ["optimal_resonance_freq", "optimal_resonator_frequency"],
            ),
            "resonator_data_keys": ", ".join(sorted(resonator)),
            "kernel_data_keys": ", ".join(sorted(kernel)),
            "iq_data_keys": ", ".join(sorted(active_iq)),
            "status": "failed" if qubit_name in self.errors else "measured",
            "errors": "; ".join(self.errors.get(qubit_name, {}).values()),
        }

    def _pulse_params(self, qubit_name: str) -> dict[str, float | None]:
        try:
            qubit = self.profile.qubits[qubit_name]
            pulse = qubit.pulses[SUPPORTED_PULSE_TYPES.readout][
                self.settings.readout_pulse_shape
            ]
            frequency = getattr(qubit.readout_resonator_frequency, "value", None)
        except Exception as error:
            self.errors.setdefault(qubit_name, {})["readout_parameters"] = (
                f"{type(error).__name__}: {error}"
            )
            return {
                "readout_resonator_frequency_hz": None,
                "readout_pulse_length_s": None,
                "readout_amplitude": None,
            }
        return {
            "readout_resonator_frequency_hz": optional_float(frequency),
            "readout_pulse_length_s": optional_float(pulse.readout_duration),
            "readout_amplitude": optional_float(pulse.readout_amplitude),
        }

    def _save(self) -> None:
        save_csv(self.run_dir / "readout_report.csv", self.rows)
        save_json(self.run_dir / "readout_report.json", self.rows)
        save_json(self.run_dir / "errors.json", self.errors)
        save_json(
            self.run_dir / "run_settings.json",
            json_ready(asdict(self.settings)),
        )
        self._save_raw_results()
        self._save_profile_snapshot()
        self.summary_images = self._save_summary_plots()
        save_html_report(
            self.run_dir / "readout_report.html",
            self.rows,
            self.image_paths,
            self.summary_images,
            self.run_datetime,
            self.settings,
        )

    def _save_raw_results(self) -> None:
        try:
            np.savez_compressed(
                self.run_dir / "raw_results.npz",
                results=np.array(self.results, dtype=object),
            )
        except Exception as error:
            save_json(
                self.run_dir / "raw_results_save_error.json",
                {"error": f"{type(error).__name__}: {error}"},
            )

    def _save_profile_snapshot(self) -> None:
        path = self.run_dir / "profile.json"
        try:
            if hasattr(self.profile, "model_dump_json"):
                path.write_text(self.profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
            elif hasattr(self.profile, "json"):
                path.write_text(self.profile.json(indent=2) + "\n", encoding="utf-8")
            else:
                save_json(path, self.profile)
        except Exception as error:
            save_json(
                self.run_dir / "profile_save_error.json",
                {"error": f"{type(error).__name__}: {error}"},
            )

    def _save_summary_plots(self) -> list[Path]:
        plots = {
            "fidelity_comparison.png": plot_fidelity_comparison(self.rows),
            "qubit_parameter_sweep.png": plot_parameter_sweep(self.rows),
        }
        paths = []
        for filename, figure in plots.items():
            relative_path = Path("figures") / "summary" / filename
            path = self.run_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=200, bbox_inches="tight")
            paths.append(relative_path)
        return paths


def plot_fidelity_comparison(rows: list[dict[str, Any]]) -> Figure:
    labels = [row["qubit"] for row in rows]
    active = float_array(rows, "active_fidelity")
    passive = float_array(rows, "passive_fidelity")
    x_values = np.arange(len(labels))
    width = 0.38

    figure, axis = plt.subplots(figsize=(max(11, len(labels) * 0.55), 6.2))
    axis.bar(x_values - width / 2, passive, width, label="Passive reset", color="#7189a6")
    axis.bar(x_values + width / 2, active, width, label="Active reset", color="#17a398")
    axis.axhline(0.90, color="#d45d5d", linestyle="--", linewidth=1, label="0.90")
    axis.axhline(0.95, color="#2f7d5c", linestyle="--", linewidth=1, label="0.95")
    axis.set_title("Readout fidelity sweep across qubits")
    axis.set_ylabel("Readout fidelity")
    axis.set_xticks(x_values, labels, rotation=55, ha="right")
    axis.set_ylim(fidelity_limits(active, passive))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    figure.tight_layout()
    return figure


def plot_parameter_sweep(rows: list[dict[str, Any]]) -> Figure:
    labels = [row["qubit"] for row in rows]
    x_values = np.arange(len(labels))
    figure, axes = plt.subplots(3, 1, figsize=(max(11, len(labels) * 0.55), 10), sharex=True)
    series = [
        ("readout_amplitude", "Readout amplitude", "#17a398", 1.0),
        ("readout_pulse_length_s", "Pulse length (ns)", "#d17b49", 1e9),
        ("readout_resonator_frequency_hz", "Resonator frequency (GHz)", "#5271a3", 1e-9),
    ]
    for axis, (key, label, color, scale) in zip(axes, series, strict=True):
        values = float_array(rows, key) * scale
        axis.plot(x_values, values, marker="o", linewidth=1.8, color=color)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xticks(x_values, labels, rotation=55, ha="right")
    figure.suptitle("Readout pulse and resonator parameter sweep")
    figure.tight_layout()
    return figure


def save_html_report(
    path: Path,
    rows: list[dict[str, Any]],
    image_paths: dict[str, dict[str, list[Path]]],
    summary_images: list[Path],
    run_datetime: datetime,
    settings: AllQubitsReadoutReportSettings,
) -> None:
    measured = sum(row["status"] == "measured" for row in rows)
    active_values = [row["active_fidelity"] for row in rows if row["active_fidelity"] is not None]
    mean_active = float(np.mean(active_values)) if active_values else None
    summary_html = "".join(image_card(path, image) for image in summary_images)
    rows_html = "".join(table_row(row) for row in rows)
    qubits_html = "".join(
        qubit_section(path, row, image_paths.get(str(row["qubit"]), {}))
        for row in rows
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>All-Qubit Readout Report</title>
<style>
:root{{--bg:#08111f;--panel:#101d30;--panel2:#14243a;--text:#e8f0fa;--muted:#91a4bb;--accent:#33c6b4;--line:#263b56;--bad:#ff7676}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07101c,#0c1930);color:var(--text);font:14px Inter,system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:30px}} h1,h2,h3{{margin-top:0}} h1{{font-size:32px}} h2{{margin-top:36px}} a{{color:var(--accent)}}
.muted{{color:var(--muted)}} .stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}}
.stat,.card,.qubit{{background:rgba(16,29,48,.94);border:1px solid var(--line);border-radius:14px;padding:18px;box-shadow:0 14px 36px #0004}}
.stat b{{display:block;font-size:25px;margin-top:7px;color:var(--accent)}} .gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
.card img{{width:100%;border-radius:8px;background:white}} .table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}}
table{{border-collapse:collapse;width:100%;background:var(--panel)}} th,td{{padding:11px 13px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;position:sticky;left:0;background:var(--panel)}} th{{color:var(--accent);background:var(--panel2)}} .failed{{color:var(--bad)}}
.qubit{{margin:18px 0}} .params{{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:16px}} .params span{{color:var(--muted)}} .params b{{color:var(--text)}}
details{{margin-top:14px}} summary{{cursor:pointer;color:var(--accent);font-weight:650}} code{{white-space:pre-wrap;color:#b9cae0}}
@media(max-width:700px){{main{{padding:16px}}.gallery{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>All-Qubit Readout Report</h1>
<p class="muted">Generated {html.escape(run_datetime.isoformat(sep=" ", timespec="seconds"))} · profile {html.escape(settings.profile_name)} · states {html.escape(", ".join(settings.states))}</p>
<section class="stats">
<div class="stat">Selected qubits<b>{len(rows)}</b></div>
<div class="stat">Successful qubits<b>{measured}</b></div>
<div class="stat">Mean active fidelity<b>{format_number(mean_active, 4)}</b></div>
<div class="stat">Active reset repetitions<b>{settings.active_reset_num}</b></div>
</section>
<h2>Experiment Sweeps</h2><section class="gallery">{summary_html}</section>
<h2>Readout Data</h2><div class="table-wrap"><table>
<thead><tr><th>Qubit</th><th>Amplitude</th><th>Pulse length (ns)</th><th>Resonator (GHz)</th><th>Active fidelity</th><th>Passive fidelity</th><th>Delta</th><th>Active separation</th><th>Status</th></tr></thead>
<tbody>{rows_html}</tbody></table></div>
<h2>Per-Qubit Experiments</h2>{qubits_html}
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def qubit_section(
    report_path: Path,
    row: dict[str, Any],
    images: dict[str, list[Path]],
) -> str:
    galleries = []
    for category, paths in images.items():
        cards = "".join(image_card(report_path, image) for image in paths)
        galleries.append(f"<h3>{html.escape(category.replace('_', ' ').title())}</h3><div class='gallery'>{cards}</div>")
    error = html.escape(str(row["errors"])) if row["errors"] else "None"
    return f"""<section class="qubit">
<h2>{html.escape(str(row["qubit"]))}</h2>
<div class="params">
<span>Amplitude <b>{format_number(row["readout_amplitude"], 6)}</b></span>
<span>Pulse length <b>{format_scaled(row["readout_pulse_length_s"], 1e9, 1)} ns</b></span>
<span>Resonator <b>{format_scaled(row["readout_resonator_frequency_hz"], 1e-9, 6)} GHz</b></span>
<span>Active fidelity <b>{format_number(row["active_fidelity"], 4)}</b></span>
<span>Passive fidelity <b>{format_number(row["passive_fidelity"], 4)}</b></span>
</div>
{''.join(galleries) or "<p class='muted'>No experiment figures were available.</p>"}
<details><summary>Data keys and errors</summary><code>Resonator: {html.escape(str(row["resonator_data_keys"]))}
Kernel: {html.escape(str(row["kernel_data_keys"]))}
IQ: {html.escape(str(row["iq_data_keys"]))}
Errors: {error}</code></details>
</section>"""


def image_card(report_path: Path, relative_path: Path) -> str:
    image_path = report_path.parent / relative_path
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return (
        "<article class='card'>"
        f"<h3>{html.escape(relative_path.stem.replace('_', ' ').title())}</h3>"
        f"<img loading='lazy' src='data:image/png;base64,{encoded}' "
        f"alt='{html.escape(relative_path.stem)}'>"
        "</article>"
    )


def table_row(row: dict[str, Any]) -> str:
    status_class = "failed" if row["status"] == "failed" else ""
    cells = [
        html.escape(str(row["qubit"])),
        format_number(row["readout_amplitude"], 6),
        format_scaled(row["readout_pulse_length_s"], 1e9, 1),
        format_scaled(row["readout_resonator_frequency_hz"], 1e-9, 6),
        format_number(row["active_fidelity"], 4),
        format_number(row["passive_fidelity"], 4),
        format_number(row["fidelity_delta_active_minus_passive"], 4),
        format_number(row["active_separation"], 4),
        html.escape(str(row["status"])),
    ]
    return "<tr>" + "".join(
        f"<td class='{status_class if index == len(cells) - 1 else ''}'>{cell}</td>"
        for index, cell in enumerate(cells)
    ) + "</tr>"


def qubit_result(
    result: dict[str, Any] | None,
    experiment: str,
    qubit_name: str,
) -> dict[str, Any]:
    if not result:
        return {}
    experiment_result = result.get(experiment, {}) or {}
    value = experiment_result.get(qubit_name, {}) if isinstance(experiment_result, dict) else {}
    return value if isinstance(value, dict) else {}


def fidelity_error(data: dict[str, Any]) -> float | None:
    return first_float(
        data,
        [
            "readout_fidelity_std",
            "readout_fidelity_error",
            "readout_fidelity_err",
            "average_readout_fidelity_std",
            "fidelity_std",
        ],
    )


def separation(data: dict[str, Any]) -> float | None:
    return first_float(
        data,
        ["separation", "readout_separation", "iq_separation", "state_separation"],
    )


def first_float(data: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = optional_float(data.get(key))
        if value is not None:
            return value
    return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def optional_difference(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def float_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array(
        [np.nan if row.get(key) is None else float(row[key]) for row in rows],
        dtype=float,
    )


def fidelity_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    return (max(0.5, float(values.min()) - 0.04), 1.0) if values.size else (0.5, 1.0)


def qubit_sort_key(qubit_name: str) -> tuple[str, int | str]:
    prefix = "".join(character for character in qubit_name if not character.isdigit())
    suffix = qubit_name[len(prefix):]
    return (prefix, int(suffix)) if suffix.isdigit() else (prefix, qubit_name)


def make_run_dir(output_root: Path, qubits: list[str], run_datetime: datetime) -> Path:
    date_dir = output_root / run_datetime.strftime("%Y-%m-%d")
    label = "all_qubits" if len(qubits) > 3 else "_".join(qubits)
    run_dir = date_dir / f"{run_datetime.strftime('%H-%M-%S')}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def format_number(value: Any, digits: int) -> str:
    number = optional_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def format_scaled(value: Any, scale: float, digits: int) -> str:
    number = optional_float(value)
    return "—" if number is None else f"{number * scale:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-qubit readout experiments and create a standalone HTML report."
    )
    parser.add_argument("--qubits", nargs="+")
    parser.add_argument("--profile-name", default=PROFILE_NAME)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--active-reset-num", type=int, default=5)
    parser.add_argument("--states", nargs="+", default=["g", "e"])
    parser.add_argument("--do-emulation", action="store_true")
    parser.add_argument("--show-handler-output", action="store_true")
    parser.add_argument("--task-status-poll-interval", type=float, default=10.0)
    parser.add_argument("--skip-resonator", action="store_true")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument("--skip-passive-comparison", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--show", action="store_true")
    args, _unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile_name)
    runner = AllQubitsReadoutReport(
        profile=profile,
        task_manager=load_task_manager(),
        settings=AllQubitsReadoutReportSettings(
            qubit_names=args.qubits or [],
            profile_name=args.profile_name,
            output_root=args.output_root,
            active_reset_num=args.active_reset_num,
            states=args.states,
            do_emulation=args.do_emulation,
            show_handler_output=args.show_handler_output,
            task_status_poll_interval=args.task_status_poll_interval,
            run_resonator=not args.skip_resonator,
            run_kernels=not args.skip_kernels,
            run_passive_comparison=not args.skip_passive_comparison,
            continue_on_error=not args.stop_on_error,
            show_plots=args.show,
        ),
    )
    result = runner.run()
    print(f"[all-qubits report] Saved report to {result['run_dir'] / 'readout_report.html'}")


if __name__ == "__main__":
    main()
