from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
import numpy as np

from optimize.readout.utils.readout_sweep_analysis import ReadoutAmplitudeSweepAnalysis
from optimize.readout.utils.readout_sweep_plotter import ReadoutAmplitudeSweepPlotter


def create_readout_run_dir(
    output_dir: str | Path,
    scan_method: str,
    qubit_names: list[str],
) -> Path:
    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M-%S")
    qubit_slug = "_".join(qubit_names)
    method_slug = scan_method.lower().replace(" ", "_")
    run_name = f"{timestamp}_{method_slug}_{qubit_slug}"
    day_dir = Path(output_dir) / date_folder
    run_dir = day_dir / run_name

    suffix = 1
    while run_dir.exists():
        run_dir = day_dir / f"{run_name}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


class ReadoutAmplitudeSweepSaver:
    def __init__(
        self,
        qubit_names: list[str],
        amplitudes: Any,
        fidelities: dict[str, list[float]],
        results: dict[float, dict[str, Any]],
        profile: Any,
        initial_amplitudes: dict[str, float] | None = None,
        readout_lengths: dict[str, float] | None = None,
        fidelity_errors: dict[str, list[float | None]] | None = None,
        separations: dict[str, list[float | None]] | None = None,
        roundnesses: dict[str, list[float | None]] | None = None,
        resonator_frequencies: dict[str, list[float | None]] | None = None,
        readout_frequencies: dict[str, float] | None = None,
        iq_blob_figures: dict[float, list[Figure]] | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        self.qubit_names = qubit_names
        self.amplitudes = amplitudes
        self.fidelities = fidelities
        self.results = results
        self.profile = profile
        self.initial_amplitudes = initial_amplitudes or {}
        self.readout_lengths = readout_lengths or {}
        self.fidelity_errors = fidelity_errors or {}
        self.separations = separations or {}
        self.roundnesses = roundnesses or {}
        self.resonator_frequencies = resonator_frequencies or {}
        self.readout_frequencies = readout_frequencies or {}
        self.iq_blob_figures = iq_blob_figures or {}
        self.profile_path = profile_path
        self.interrupted = False
        self.interrupt_reason: str | None = None
        self.reset_label: str | None = None
        self.scan_method: str = "unknown"
        self.measurement_errors: dict[float, str] = {}
        self.kernel_figures: dict[float, list[Figure]] = {}
        self.resonator_figures: dict[float, list[Figure]] = {}

    def save(
        self,
        output_dir: str | Path = Path("data") / "readout_optimize",
        figure: Figure | None = None,
        run_dir: str | Path | None = None,
    ) -> str:
        run_dir = (
            Path(run_dir)
            if run_dir is not None
            else self._create_run_dir(output_dir)
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = ReadoutAmplitudeSweepAnalysis(
            qubit_names=self.qubit_names,
            amplitudes=self.amplitudes,
            fidelities=self.fidelities,
            initial_amplitudes=self.initial_amplitudes,
        ).summary()
        summary["interrupted"] = self.interrupted
        summary["interrupt_reason"] = self.interrupt_reason
        summary["scan_method"] = self.scan_method
        summary["measurement_errors"] = self.measurement_errors
        summary["resonator_frequencies"] = self.resonator_frequencies
        summary["readout_frequencies"] = self.readout_frequencies
        summary["roundnesses"] = self.roundnesses

        np.savez_compressed(
            run_dir / "data.npz",
            amplitudes=self.amplitudes,
            fidelities=np.array([self.fidelities], dtype=object),
            fidelity_errors=np.array([self.fidelity_errors], dtype=object),
            separations=np.array([self.separations], dtype=object),
            roundnesses=np.array([self.roundnesses], dtype=object),
            resonator_frequencies=np.array([self.resonator_frequencies], dtype=object),
            results=np.array([self.results], dtype=object),
            measurement_errors=np.array([self.measurement_errors], dtype=object),
        )

        profile_status = self._save_profile(run_dir)
        self._save_csv(run_dir / "fidelities.csv")
        self._save_summary(run_dir / "summary.json", summary)
        self._save_report(run_dir / "report.md", summary, profile_status)
        self._save_plot(run_dir / "plot.png", figure)
        self._save_iq_blob_figures(run_dir / "iq_blobs")
        self._save_kernel_figures(run_dir / "kernels")
        self._save_resonator_figures(run_dir / "resonator")

        return str(run_dir)

    def _create_run_dir(self, output_dir: str | Path) -> Path:
        return create_readout_run_dir(
            output_dir=output_dir,
            scan_method=self.scan_method,
            qubit_names=self.qubit_names,
        )

    def _save_csv(self, path: Path) -> None:
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "amplitude",
                    *self.qubit_names,
                    *[f"{qubit}_fidelity_error" for qubit in self.qubit_names],
                    *[f"{qubit}_separation" for qubit in self.qubit_names],
                    *[f"{qubit}_roundness" for qubit in self.qubit_names],
                    *[f"{qubit}_resonator_frequency" for qubit in self.qubit_names],
                    "mean_fidelity",
                    "status",
                    "error",
                ]
            )

            for index, amplitude in enumerate(self.amplitudes):
                row_fidelities = [
                    float(self.fidelities[qubit_name][index])
                    for qubit_name in self.qubit_names
                ]
                row_errors = [
                    self._optional_metric_at(self.fidelity_errors, qubit_name, index)
                    for qubit_name in self.qubit_names
                ]
                row_separations = [
                    self._optional_metric_at(self.separations, qubit_name, index)
                    for qubit_name in self.qubit_names
                ]
                row_resonator_frequencies = [
                    self._optional_metric_at(
                        self.resonator_frequencies,
                        qubit_name,
                        index,
                    )
                    for qubit_name in self.qubit_names
                ]
                row_roundnesses = [
                    self._optional_metric_at(self.roundnesses, qubit_name, index)
                    for qubit_name in self.qubit_names
                ]
                writer.writerow(
                    [
                        float(amplitude),
                        *row_fidelities,
                        *row_errors,
                        *row_separations,
                        *row_roundnesses,
                        *row_resonator_frequencies,
                        float(np.mean(row_fidelities)),
                        self._measurement_status(float(amplitude)),
                        self.measurement_errors.get(float(amplitude), ""),
                    ]
                )

    def _optional_metric_at(
        self,
        metrics: dict[str, list[float | None]],
        qubit_name: str,
        index: int,
    ) -> float | None:
        values = metrics.get(qubit_name, [])
        if index >= len(values) or values[index] is None:
            return None
        return float(values[index])

    def _save_summary(self, path: Path, summary: dict[str, Any]) -> None:
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _save_report(
        self,
        path: Path,
        summary: dict[str, Any],
        profile_status: str,
    ) -> None:
        lines = [
            "# Readout Amplitude Sweep Optimization",
            "",
            f"Created at: {summary['created_at']}",
            f"Qubits: {', '.join(summary['qubits'])}",
            f"Scan method: {summary['scan_method']}",
            f"Interrupted: {summary.get('interrupted', False)}",
            f"Best mean amplitude: {summary['best_mean_amplitude']}",
            f"Best mean fidelity: {summary['best_mean_fidelity']}",
            "",
            "## Per-Qubit Best Results",
            "",
            "| Qubit | Readout frequency (GHz) | Initial amplitude | Best amplitude | Best fidelity | Final fidelity |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        if summary.get("interrupt_reason"):
            lines.insert(5, f"Interrupt reason: {summary['interrupt_reason']}")

        for qubit_name, qubit_summary in summary["qubit_summaries"].items():
            lines.append(
                "| "
                f"{qubit_name} | "
                f"{self._frequency_ghz(qubit_name)} | "
                f"{qubit_summary['initial_amplitude']} | "
                f"{qubit_summary['best_amplitude']} | "
                f"{qubit_summary['best_fidelity']} | "
                f"{qubit_summary['final_fidelity']} |"
            )

        if summary.get("measurement_errors"):
            lines.extend(
                [
                    "",
                    "## Measurement Errors",
                    "",
                    "| Amplitude | Error |",
                    "| ---: | --- |",
                ]
            )
            for amplitude, error in summary["measurement_errors"].items():
                lines.append(f"| {float(amplitude)} | {error} |")

        lines.extend(
            [
                "",
                "## Saved Files",
                "",
                "- `data.npz`: amplitudes, fidelities, and raw workflow results.",
                "- `fidelities.csv`: tabular amplitudes, fidelity values, and resonator frequencies.",
                "- `summary.json`: machine-readable analysis summary.",
                "- `plot.png`: readout fidelity plot.",
                "- `iq_blobs/`: IQ blobs plots for each measured amplitude.",
                "- `kernels/`: kernel plots for each measured amplitude.",
                "- `resonator/`: resonator plots for each measured amplitude.",
                f"- `profile.json`: {profile_status}.",
            ]
        )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _measurement_status(self, amplitude: float) -> str:
        if amplitude in self.measurement_errors:
            return "failed"

        return "ok"

    def _save_plot(self, path: Path, figure: Figure | None) -> None:
        if figure is None:
            plotter = ReadoutAmplitudeSweepPlotter(
                self.qubit_names,
                self.amplitudes,
                self.fidelities,
            )
            plotter.initial_amplitudes = self.initial_amplitudes
            plotter.readout_lengths = self.readout_lengths
            plotter.reset_label = self.reset_label
            plotter.fidelity_errors = self.fidelity_errors
            plotter.separations = self.separations
            plotter.roundnesses = self.roundnesses
            figure = plotter.plot()

        fig = figure
        fig.savefig(path, dpi=200, bbox_inches="tight")

    def _frequency_ghz(self, qubit_name: str) -> str:
        frequency = self.readout_frequencies.get(qubit_name)
        return "not available" if frequency is None else f"{frequency / 1e9:.6f}"

    def _save_iq_blob_figures(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, amplitude in enumerate(self.amplitudes):
            figures = self.iq_blob_figures.get(float(amplitude), [])
            for figure_index, figure in enumerate(figures):
                suffix = "" if len(figures) == 1 else f"_fig{figure_index + 1}"
                figure.savefig(
                    output_dir
                    / f"{index:03d}_amplitude_{float(amplitude):.6g}{suffix}.png",
                    dpi=200,
                )

    def _save_kernel_figures(self, output_dir: Path) -> None:
        if not self.kernel_figures:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        for index, amplitude in enumerate(self.amplitudes):
            figures = self.kernel_figures.get(float(amplitude), [])
            for figure_index, figure in enumerate(figures):
                suffix = "" if len(figures) == 1 else f"_fig{figure_index + 1}"
                figure.savefig(
                    output_dir
                    / f"{index:03d}_amplitude_{float(amplitude):.6g}{suffix}.png",
                    dpi=200,
                )

    def _save_resonator_figures(self, output_dir: Path) -> None:
        if not self.resonator_figures:
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        for index, amplitude in enumerate(self.amplitudes):
            figures = self.resonator_figures.get(float(amplitude), [])
            for figure_index, figure in enumerate(figures):
                suffix = "" if len(figures) == 1 else f"_fig{figure_index + 1}"
                figure.savefig(
                    output_dir
                    / f"{index:03d}_amplitude_{float(amplitude):.6g}{suffix}.png",
                    dpi=200,
                )

    def _save_profile(self, run_dir: Path) -> str:
        profile_path = self._find_profile_path()
        destination = run_dir / "profile.json"

        if profile_path is not None:
            shutil.copy2(profile_path, destination)
            return f"copied from `{profile_path}`"

        snapshot = self._profile_snapshot()
        if snapshot is None:
            return "not saved because no profile JSON was found and the profile object could not be serialized"

        destination.write_text(
            json.dumps(snapshot, indent=2, default=str),
            encoding="utf-8",
        )
        return "saved as a best-effort snapshot from the in-memory profile object"

    def _profile_snapshot(self) -> Any:
        if hasattr(self.profile, "model_dump"):
            return self.profile.model_dump(mode="json")
        if hasattr(self.profile, "dict"):
            return self.profile.dict()
        if is_dataclass(self.profile):
            return asdict(self.profile)
        if hasattr(self.profile, "__dict__"):
            return self.profile.__dict__
        return None

    def _find_profile_path(self) -> Path | None:
        if self.profile_path is not None:
            profile_path = Path(self.profile_path)
            return profile_path if profile_path.exists() else None

        search_roots = [Path.cwd(), Path.cwd().parent]
        for root in search_roots:
            for candidate in root.glob("**/profile.json"):
                if "data" not in candidate.parts:
                    return candidate

        return None
