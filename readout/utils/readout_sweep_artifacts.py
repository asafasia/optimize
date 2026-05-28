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
        self.iq_blob_figures = iq_blob_figures or {}
        self.profile_path = profile_path
        self.interrupted = False
        self.interrupt_reason: str | None = None
        self.reset_label: str | None = None
        self.scan_method: str = "unknown"

    def save(
        self,
        output_dir: str | Path = Path("data") / "readout_optimize",
        figure: Figure | None = None,
    ) -> str:
        run_dir = self._create_run_dir(output_dir)
        summary = ReadoutAmplitudeSweepAnalysis(
            qubit_names=self.qubit_names,
            amplitudes=self.amplitudes,
            fidelities=self.fidelities,
            initial_amplitudes=self.initial_amplitudes,
        ).summary()
        summary["interrupted"] = self.interrupted
        summary["interrupt_reason"] = self.interrupt_reason
        summary["scan_method"] = self.scan_method

        np.savez_compressed(
            run_dir / "data.npz",
            amplitudes=self.amplitudes,
            fidelities=np.array([self.fidelities], dtype=object),
            fidelity_errors=np.array([self.fidelity_errors], dtype=object),
            separations=np.array([self.separations], dtype=object),
            results=np.array([self.results], dtype=object),
        )

        profile_status = self._save_profile(run_dir)
        self._save_csv(run_dir / "fidelities.csv")
        self._save_summary(run_dir / "summary.json", summary)
        self._save_report(run_dir / "report.md", summary, profile_status)
        self._save_plot(run_dir / "plot.png", figure)
        self._save_iq_blob_figures(run_dir / "iq_blobs")

        return str(run_dir)

    def _create_run_dir(self, output_dir: str | Path) -> Path:
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H-%M-%S")
        qubit_slug = "_".join(self.qubit_names)
        method_slug = self._slug(self.scan_method)
        run_name = f"{timestamp}_{method_slug}_{qubit_slug}"
        day_dir = Path(output_dir) / date_folder
        run_dir = day_dir / run_name

        suffix = 1
        while run_dir.exists():
            run_dir = day_dir / f"{run_name}_{suffix}"
            suffix += 1

        run_dir.mkdir(parents=True)
        return run_dir

    def _slug(self, value: str) -> str:
        return value.lower().replace(" ", "_")

    def _save_csv(self, path: Path) -> None:
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "amplitude",
                    *self.qubit_names,
                    *[f"{qubit}_fidelity_error" for qubit in self.qubit_names],
                    *[f"{qubit}_separation" for qubit in self.qubit_names],
                    "mean_fidelity",
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
                writer.writerow(
                    [
                        float(amplitude),
                        *row_fidelities,
                        *row_errors,
                        *row_separations,
                        float(np.mean(row_fidelities)),
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
            "| Qubit | Initial amplitude | Best amplitude | Best fidelity | Final fidelity |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        if summary.get("interrupt_reason"):
            lines.insert(5, f"Interrupt reason: {summary['interrupt_reason']}")

        for qubit_name, qubit_summary in summary["qubit_summaries"].items():
            lines.append(
                "| "
                f"{qubit_name} | "
                f"{qubit_summary['initial_amplitude']} | "
                f"{qubit_summary['best_amplitude']} | "
                f"{qubit_summary['best_fidelity']} | "
                f"{qubit_summary['final_fidelity']} |"
            )

        lines.extend(
            [
                "",
                "## Saved Files",
                "",
                "- `data.npz`: amplitudes, fidelities, and raw workflow results.",
                "- `fidelities.csv`: tabular amplitudes and fidelity values.",
                "- `summary.json`: machine-readable analysis summary.",
                "- `plot.png`: readout fidelity plot.",
                "- `iq_blobs/`: IQ blobs plots for each measured amplitude.",
                f"- `profile.json`: {profile_status}.",
            ]
        )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
            figure = plotter.plot()

        fig = figure
        fig.savefig(path, dpi=200, bbox_inches="tight")

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
                    bbox_inches="tight",
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
