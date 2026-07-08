from __future__ import annotations

import base64
import html
import json
import platform
import subprocess
import webbrowser
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from optimize.readout.optimizer.plotter import ReadoutAmplitudeSweepPlotter


class ReadoutLiveHtmlPlotter:
    """Experimental file-based live monitor for readout amplitude optimization."""

    def __init__(
        self,
        output_dir: str | Path,
        refresh_interval_seconds: float = 1.0,
        open_browser: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.refresh_interval_seconds = float(refresh_interval_seconds)
        self.open_browser = open_browser
        self.html_path = self.output_dir / "live_readout_optimizer.html"
        self.iq_blobs_dir = self.output_dir / "iq_blobs"
        self.kernels_dir = self.output_dir / "kernels"
        self.resonator_dir = self.output_dir / "resonator"
        self.fidelity_history_dir = self.output_dir / "fidelity_history"
        self.fidelity_path = self.output_dir / "plot.png"
        self.resonator_frequency_path = self.output_dir / "resonator_frequency.png"
        self._opened = False
        self._title = "Readout amplitude optimizer"
        self._latest_amplitude: float | None = None
        self._points = 0
        self._version = 0
        self._iq_history: list[dict[str, str | float]] = []
        self._kernel_history: list[dict[str, str | float]] = []
        self._resonator_history: list[dict[str, str | float]] = []
        self._report_html = "<p>No measurements yet.</p>"
        self._workflow_label = "not started"
        self._scan_method = "unknown"
        self._optimization_parameters: dict[str, Any] = {}

    def start(
        self,
        title: str = "Readout amplitude optimizer",
        workflow_label: str = "not started",
        scan_method: str = "unknown",
        optimization_parameters: dict[str, Any] | None = None,
    ) -> None:
        self._title = title
        self._workflow_label = workflow_label
        self._scan_method = scan_method
        self._optimization_parameters = optimization_parameters or {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iq_blobs_dir.mkdir(parents=True, exist_ok=True)
        self.kernels_dir.mkdir(parents=True, exist_ok=True)
        self.resonator_dir.mkdir(parents=True, exist_ok=True)
        self.fidelity_history_dir.mkdir(parents=True, exist_ok=True)
        self._save_placeholder(self.fidelity_path, "Waiting for fidelity data")
        self._save_placeholder(
            self.resonator_frequency_path,
            "Waiting for resonator frequency data",
        )
        self._write_html(title=self._title)

        if self.open_browser and not self._opened:
            self._open_html()
            self._opened = True
        print(f"Live readout monitor: {self.html_path.resolve()}")

    def update(
        self,
        *,
        qubit_names: list[str],
        amplitudes: list[float],
        fidelities: dict[str, list[float]],
        fidelity_errors: dict[str, list[float | None]],
        separations: dict[str, list[float | None]],
        roundnesses: dict[str, list[float | None]],
        resonator_frequencies: dict[str, list[float | None]],
        initial_amplitudes: dict[str, float],
        readout_lengths: dict[str, float],
        readout_frequencies: dict[str, float],
        reset_label: str,
        latest_amplitude: float | None,
        latest_iq_figures: list[Figure] | None,
        latest_kernel_figures: list[Figure] | None = None,
        latest_resonator_figures: list[Figure] | None = None,
        workflow_label: str = "not started",
        scan_method: str = "unknown",
        optimization_parameters: dict[str, Any] | None = None,
    ) -> None:
        self._version += 1
        self._workflow_label = workflow_label
        self._scan_method = scan_method
        self._optimization_parameters = optimization_parameters or {}
        if latest_amplitude is not None:
            self._latest_amplitude = latest_amplitude
        self._points = len(amplitudes)

        if latest_iq_figures:
            if latest_amplitude is not None:
                point_index = max(0, len(amplitudes) - 1)
                for index, figure in enumerate(latest_iq_figures, start=1):
                    label = f"A={float(latest_amplitude):.4g}"
                    if len(latest_iq_figures) > 1:
                        label = f"{label}, fig {index}"
                    self._save_iq_history_figure(
                        label=label,
                        amplitude=float(latest_amplitude),
                        point_index=point_index,
                        figure=figure,
                        figure_index=index - 1,
                        figure_count=len(latest_iq_figures),
                    )
        if latest_kernel_figures:
            if latest_amplitude is not None:
                point_index = max(0, len(amplitudes) - 1)
                for index, figure in enumerate(latest_kernel_figures, start=1):
                    label = f"A={float(latest_amplitude):.4g}"
                    if len(latest_kernel_figures) > 1:
                        label = f"{label}, fig {index}"
                    self._save_kernel_history_figure(
                        label=label,
                        amplitude=float(latest_amplitude),
                        point_index=point_index,
                        figure=figure,
                        figure_index=index - 1,
                        figure_count=len(latest_kernel_figures),
                    )
        if latest_resonator_figures:
            if latest_amplitude is not None:
                point_index = max(0, len(amplitudes) - 1)
                for index, figure in enumerate(latest_resonator_figures, start=1):
                    label = f"A={float(latest_amplitude):.4g}"
                    if len(latest_resonator_figures) > 1:
                        label = f"{label}, fig {index}"
                    self._save_resonator_history_figure(
                        label=label,
                        amplitude=float(latest_amplitude),
                        point_index=point_index,
                        figure=figure,
                        figure_index=index - 1,
                        figure_count=len(latest_resonator_figures),
                    )

        self._save_fidelity_plot(
            qubit_names=qubit_names,
            amplitudes=amplitudes,
            fidelities=fidelities,
            fidelity_errors=fidelity_errors,
            separations=separations,
            roundnesses=roundnesses,
            initial_amplitudes=initial_amplitudes,
            readout_lengths=readout_lengths,
            reset_label=reset_label,
        )
        self._save_fidelity_history_plots(
            qubit_names=qubit_names,
            amplitudes=amplitudes,
            fidelities=fidelities,
            fidelity_errors=fidelity_errors,
            separations=separations,
            roundnesses=roundnesses,
            initial_amplitudes=initial_amplitudes,
            readout_lengths=readout_lengths,
            reset_label=reset_label,
        )
        self._save_resonator_frequency_plot(
            qubit_names=qubit_names,
            amplitudes=amplitudes,
            resonator_frequencies=resonator_frequencies,
        )
        self._update_report(
            qubit_names=qubit_names,
            amplitudes=amplitudes,
            fidelities=fidelities,
            fidelity_errors=fidelity_errors,
            separations=separations,
            initial_amplitudes=initial_amplitudes,
            readout_lengths=readout_lengths,
            readout_frequencies=readout_frequencies,
            reset_label=reset_label,
            latest_amplitude=latest_amplitude,
        )
        self._write_html(
            title=self._title,
            latest_amplitude=self._latest_amplitude,
            points=self._points,
        )

    def finish(self) -> None:
        self._version += 1
        self.write_standalone_html(status="finished")

    def write_standalone_html(self, status: str = "saved") -> None:
        self._write_html(
            title=self._title,
            latest_amplitude=self._latest_amplitude,
            points=self._points,
            status=status,
            embed_images=True,
            auto_refresh=False,
        )

    def _open_html(self) -> None:
        html_uri = self.html_path.resolve().as_uri()
        opened = webbrowser.open(html_uri, new=2)
        if opened:
            return

        if platform.system() == "Darwin":
            subprocess.run(["open", html_uri], check=False)

    def _save_fidelity_plot(
        self,
        *,
        qubit_names: list[str],
        amplitudes: list[float],
        fidelities: dict[str, list[float]],
        fidelity_errors: dict[str, list[float | None]],
        separations: dict[str, list[float | None]],
        roundnesses: dict[str, list[float | None]],
        initial_amplitudes: dict[str, float],
        readout_lengths: dict[str, float],
        reset_label: str,
        output_path: Path | None = None,
        selected_amplitude: float | None = None,
    ) -> None:
        path = self.fidelity_path if output_path is None else output_path
        if not amplitudes:
            self._save_placeholder(path, "Waiting for fidelity data")
            return

        plotter = ReadoutAmplitudeSweepPlotter(
            qubit_names=qubit_names,
            amplitudes=amplitudes,
            fidelities=fidelities,
            initial_amplitudes=initial_amplitudes,
            readout_lengths=readout_lengths,
        )
        plotter.fidelity_errors = fidelity_errors
        plotter.separations = separations
        plotter.roundnesses = roundnesses
        plotter.reset_label = reset_label
        plotter.selected_amplitude = selected_amplitude

        figure = plotter.plot()
        figure.savefig(path, dpi=140)
        plt.close(figure)

    def _save_iq_history_figure(
        self,
        label: str,
        amplitude: float,
        point_index: int,
        figure: Figure,
        figure_index: int,
        figure_count: int,
    ) -> None:
        suffix = "" if figure_count == 1 else f"_fig{figure_index + 1}"
        path = (
            self.iq_blobs_dir
            / f"{point_index:03d}_amplitude_{float(amplitude):.6g}{suffix}.png"
        )
        figure.savefig(path, dpi=140)
        relative_path = path.relative_to(self.output_dir).as_posix()
        self._iq_history.append(
            {
                "label": label,
                "amplitude": float(amplitude),
                "iq_src": relative_path,
                "fidelity_src": "",
            }
        )

    def _save_kernel_history_figure(
        self,
        label: str,
        amplitude: float,
        point_index: int,
        figure: Figure,
        figure_index: int,
        figure_count: int,
    ) -> None:
        suffix = "" if figure_count == 1 else f"_fig{figure_index + 1}"
        path = (
            self.kernels_dir
            / f"{point_index:03d}_amplitude_{float(amplitude):.6g}{suffix}.png"
        )
        figure.savefig(path, dpi=140)
        relative_path = path.relative_to(self.output_dir).as_posix()
        self._kernel_history.append(
            {
                "label": label,
                "amplitude": float(amplitude),
                "kernel_src": relative_path,
                "fidelity_src": "",
            }
        )

    def _save_resonator_history_figure(
        self,
        label: str,
        amplitude: float,
        point_index: int,
        figure: Figure,
        figure_index: int,
        figure_count: int,
    ) -> None:
        suffix = "" if figure_count == 1 else f"_fig{figure_index + 1}"
        path = (
            self.resonator_dir
            / f"{point_index:03d}_amplitude_{float(amplitude):.6g}{suffix}.png"
        )
        figure.savefig(path, dpi=140)
        relative_path = path.relative_to(self.output_dir).as_posix()
        self._resonator_history.append(
            {
                "label": label,
                "amplitude": float(amplitude),
                "resonator_src": relative_path,
                "fidelity_src": "",
            }
        )

    def _save_fidelity_history_plots(
        self,
        *,
        qubit_names: list[str],
        amplitudes: list[float],
        fidelities: dict[str, list[float]],
        fidelity_errors: dict[str, list[float | None]],
        separations: dict[str, list[float | None]],
        roundnesses: dict[str, list[float | None]],
        initial_amplitudes: dict[str, float],
        readout_lengths: dict[str, float],
        reset_label: str,
    ) -> None:
        if not amplitudes:
            return

        for index, item in enumerate(self._iq_history, start=1):
            if item["fidelity_src"]:
                continue
            path = self.fidelity_history_dir / f"fidelity_{index:04d}.png"
            self._save_fidelity_plot(
                qubit_names=qubit_names,
                amplitudes=amplitudes,
                fidelities=fidelities,
                fidelity_errors=fidelity_errors,
                separations=separations,
                roundnesses=roundnesses,
                initial_amplitudes=initial_amplitudes,
                readout_lengths=readout_lengths,
                reset_label=reset_label,
                output_path=path,
                selected_amplitude=float(item["amplitude"]),
            )
            item["fidelity_src"] = path.relative_to(self.output_dir).as_posix()

        for index, item in enumerate(self._kernel_history, start=1):
            if item["fidelity_src"]:
                continue
            path = self.fidelity_history_dir / f"kernel_fidelity_{index:04d}.png"
            self._save_fidelity_plot(
                qubit_names=qubit_names,
                amplitudes=amplitudes,
                fidelities=fidelities,
                fidelity_errors=fidelity_errors,
                separations=separations,
                roundnesses=roundnesses,
                initial_amplitudes=initial_amplitudes,
                readout_lengths=readout_lengths,
                reset_label=reset_label,
                output_path=path,
                selected_amplitude=float(item["amplitude"]),
            )
            item["fidelity_src"] = path.relative_to(self.output_dir).as_posix()

        for index, item in enumerate(self._resonator_history, start=1):
            if item["fidelity_src"]:
                continue
            path = self.fidelity_history_dir / f"resonator_fidelity_{index:04d}.png"
            self._save_fidelity_plot(
                qubit_names=qubit_names,
                amplitudes=amplitudes,
                fidelities=fidelities,
                fidelity_errors=fidelity_errors,
                separations=separations,
                roundnesses=roundnesses,
                initial_amplitudes=initial_amplitudes,
                readout_lengths=readout_lengths,
                reset_label=reset_label,
                output_path=path,
                selected_amplitude=float(item["amplitude"]),
            )
            item["fidelity_src"] = path.relative_to(self.output_dir).as_posix()

    def _save_resonator_frequency_plot(
        self,
        *,
        qubit_names: list[str],
        amplitudes: list[float],
        resonator_frequencies: dict[str, list[float | None]],
    ) -> None:
        has_data = any(
            value is not None
            for qubit_name in qubit_names
            for value in resonator_frequencies.get(qubit_name, [])
        )
        if not amplitudes or not has_data:
            self._save_placeholder(
                self.resonator_frequency_path,
                "Waiting for resonator frequency data",
            )
            return

        figure, axis = plt.subplots(figsize=(4.6, 3.1))
        for qubit_name in qubit_names:
            values = resonator_frequencies.get(qubit_name, [])
            baseline = next(
                (float(value) for value in values if value is not None),
                None,
            )
            if baseline is None:
                continue
            x_values = []
            y_values = []
            for index, amplitude in enumerate(amplitudes):
                if index >= len(values) or values[index] is None:
                    continue
                x_values.append(float(amplitude))
                y_values.append((float(values[index]) - baseline) / 1e6)
            if x_values:
                axis.plot(x_values, y_values, marker="o", label=qubit_name)

        axis.set_xlabel("Readout amplitude")
        axis.set_ylabel("Drift from initial frequency (MHz)")
        axis.set_title("Resonator frequency drift")
        axis.axhline(0.0, color="#526071", linewidth=0.8, alpha=0.6)
        axis.grid(True, alpha=0.25)
        if len(qubit_names) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(self.resonator_frequency_path, dpi=180)
        plt.close(figure)

    def _save_placeholder(self, path: Path, text: str) -> None:
        figure, axis = plt.subplots(figsize=(7.5, 4.5))
        axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=14)
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)

    def _update_report(
        self,
        *,
        qubit_names: list[str],
        amplitudes: list[float],
        fidelities: dict[str, list[float]],
        fidelity_errors: dict[str, list[float | None]],
        separations: dict[str, list[float | None]],
        initial_amplitudes: dict[str, float],
        readout_lengths: dict[str, float],
        readout_frequencies: dict[str, float],
        reset_label: str,
        latest_amplitude: float | None,
    ) -> None:
        if not amplitudes:
            self._report_html = "<p>No measurements yet.</p>"
            return

        mean_fidelities = []
        for index in range(len(amplitudes)):
            values = [
                float(fidelities[qubit_name][index])
                for qubit_name in qubit_names
                if index < len(fidelities.get(qubit_name, []))
            ]
            mean_fidelities.append(sum(values) / len(values) if values else 0.0)

        best_mean_index = max(range(len(mean_fidelities)), key=mean_fidelities.__getitem__)
        best_mean_amplitude = float(amplitudes[best_mean_index])
        best_mean_fidelity = float(mean_fidelities[best_mean_index])
        latest_label = "none" if latest_amplitude is None else f"{float(latest_amplitude):.6g}"
        qubits_label = ", ".join(qubit_names)
        lengths_label = ", ".join(
            f"{qubit_name}: {readout_lengths[qubit_name] * 1e9:.0f} ns"
            for qubit_name in qubit_names
            if qubit_name in readout_lengths
        ) or "not available"
        frequencies_label = ", ".join(
            f"{qubit_name}: {readout_frequencies[qubit_name] / 1e9:.6f} GHz"
            for qubit_name in qubit_names
            if qubit_name in readout_frequencies
        ) or "not available"
        initial_label = ", ".join(
            f"{qubit_name}: {initial_amplitudes[qubit_name]:.6g}"
            for qubit_name in qubit_names
            if qubit_name in initial_amplitudes
        ) or "not available"
        parameter_label = ", ".join(
            f"{key}: {value}"
            for key, value in self._optimization_parameters.items()
        ) or "none"

        rows = []
        for qubit_name in qubit_names:
            qubit_fidelities = [float(value) for value in fidelities.get(qubit_name, [])]
            if not qubit_fidelities:
                continue

            best_index = max(range(len(qubit_fidelities)), key=qubit_fidelities.__getitem__)
            best_amplitude = float(amplitudes[best_index])
            best_fidelity = qubit_fidelities[best_index]
            latest_fidelity = qubit_fidelities[-1]
            latest_error = self._metric_at(fidelity_errors, qubit_name, len(qubit_fidelities) - 1)
            latest_separation = self._metric_at(separations, qubit_name, len(qubit_fidelities) - 1)
            rows.append(
                "<tr>"
                f"<td>{html.escape(qubit_name)}</td>"
                f"<td>{best_amplitude:.6g}</td>"
                f"<td>{best_fidelity:.4f}</td>"
                f"<td>{latest_fidelity:.4f}</td>"
                f"<td>{self._format_optional(latest_error)}</td>"
                f"<td>{self._format_optional(latest_separation)}</td>"
                "</tr>"
            )

        self._report_html = (
            "<section class=\"report\">"
            "<h2>Run Summary</h2>"
            "<div class=\"report-grid\">"
            f"<div><span>Qubits</span><strong>{html.escape(qubits_label)}</strong></div>"
            f"<div><span>Workflow</span><strong>{html.escape(self._workflow_label)}</strong></div>"
            f"<div><span>Optimization method</span><strong>{html.escape(self._scan_method)}</strong></div>"
            f"<div><span>Optimization parameters</span><strong>{html.escape(parameter_label)}</strong></div>"
            f"<div><span>Measured points</span><strong>{len(amplitudes)}</strong></div>"
            f"<div><span>Latest amplitude</span><strong>{latest_label}</strong></div>"
            f"<div><span>Best mean amplitude</span><strong>{best_mean_amplitude:.6g}</strong></div>"
            f"<div><span>Best mean fidelity</span><strong>{best_mean_fidelity:.4f}</strong></div>"
            f"<div><span>Initial amplitudes</span><strong>{html.escape(initial_label)}</strong></div>"
            f"<div><span>Readout length</span><strong>{html.escape(lengths_label)}</strong></div>"
            f"<div><span>Readout frequency</span><strong>{html.escape(frequencies_label)}</strong></div>"
            f"<div><span>Reset</span><strong>{html.escape(reset_label)}</strong></div>"
            "</div>"
            "<table>"
            "<thead><tr><th>Qubit</th><th>Best A</th><th>Best F</th><th>Latest F</th><th>Latest err</th><th>Latest sep</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</section>"
        )

    def _metric_at(
        self,
        metrics: dict[str, list[float | None]],
        qubit_name: str,
        index: int,
    ) -> float | None:
        values = metrics.get(qubit_name, [])
        if index >= len(values):
            return None
        return values[index]

    def _format_optional(self, value: float | None) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4g}"

    def _write_html(
        self,
        title: str,
        latest_amplitude: float | None = None,
        points: int | None = None,
        status: str = "running",
        embed_images: bool = False,
        auto_refresh: bool = True,
    ) -> None:
        safe_title = html.escape(title)
        safe_status = html.escape(status)
        amplitude_label = (
            "none"
            if latest_amplitude is None
            else f"{float(latest_amplitude):.6g}"
        )
        points_label = "0" if points is None else str(points)
        refresh_ms = max(250, int(self.refresh_interval_seconds * 1000))
        history_count = len(self._iq_history)
        iq_items = [
            {
                "label": f"Result {index} / {history_count} | {item['label']}",
                "imageSrc": self._image_src(str(item["iq_src"]), embed_images),
                "fidelitySrc": self._image_src(
                    str(item["fidelity_src"]),
                    embed_images,
                ),
            }
            for index, item in enumerate(self._iq_history, start=1)
        ]
        kernel_count = len(self._kernel_history)
        kernel_items = [
            {
                "label": f"Result {index} / {kernel_count} | {item['label']}",
                "imageSrc": self._image_src(str(item["kernel_src"]), embed_images),
                "fidelitySrc": self._image_src(
                    str(item["fidelity_src"]),
                    embed_images,
                ),
            }
            for index, item in enumerate(self._kernel_history, start=1)
        ]
        resonator_count = len(self._resonator_history)
        resonator_items = [
            {
                "label": f"Result {index} / {resonator_count} | {item['label']}",
                "imageSrc": self._image_src(str(item["resonator_src"]), embed_images),
                "fidelitySrc": self._image_src(
                    str(item["fidelity_src"]),
                    embed_images,
                ),
            }
            for index, item in enumerate(self._resonator_history, start=1)
        ]
        iq_items_js = json.dumps(iq_items)
        kernel_items_js = json.dumps(kernel_items)
        resonator_items_js = json.dumps(resonator_items)
        active_items = iq_items or kernel_items or resonator_items
        latest_experiment_src = (
            active_items[-1]["imageSrc"]
            if active_items
            else ""
        )
        fidelity_src = (
            active_items[-1]["fidelitySrc"]
            if active_items and active_items[-1]["fidelitySrc"]
            else self._image_src(self.fidelity_path.name, embed_images)
        )
        resonator_frequency_src = self._image_src(
            self.resonator_frequency_path.name,
            embed_images,
        )
        auto_refresh_script = (
            f"""
    const refreshMs = {refresh_ms};
    function refreshImage(id) {{
      const image = document.getElementById(id);
      const base = image.src.split("?")[0];
      image.src = base + "?t=" + Date.now();
    }}
    setInterval(() => {{
      refreshImage("fidelity");
      refreshImage("resonator-frequency");
      window.location.reload();
    }}, refreshMs);
"""
            if auto_refresh
            else ""
        )
        slider_max = max(0, len(active_items) - 1)
        slider_value = slider_max
        slider_disabled = "disabled" if len(active_items) <= 1 else ""
        selected_iq_label = (
            active_items[-1]["label"]
            if active_items
            else "waiting for experiment figures"
        )
        if iq_items:
            active_tab = "iq"
            experiment_title = "Acquired IQ blobs"
        elif kernel_items:
            active_tab = "kernel"
            experiment_title = "Kernel experiments"
        else:
            active_tab = "resonator"
            experiment_title = "Resonator results"

        self.html_path.write_text(
            f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #17202a;
    }}
    body {{
      margin: 0;
      padding: 18px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 14px;
    }}
    h1 {{
      font-size: 20px;
      margin: 0;
      font-weight: 650;
    }}
    .meta {{
      font-size: 13px;
      color: #526071;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 14px;
      margin-bottom: 14px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .tab-button {{
      border: 1px solid #cfd8e5;
      background: #ffffff;
      color: #526071;
      border-radius: 8px;
      padding: 7px 12px;
      font-size: 14px;
      font-weight: 650;
      cursor: pointer;
    }}
    .tab-button.active {{
      background: #2563eb;
      border-color: #2563eb;
      color: #ffffff;
    }}
    figure {{
      margin: 0;
      background: #ffffff;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 12px;
    }}
    figcaption {{
      font-size: 15px;
      font-weight: 650;
      color: #526071;
      margin-bottom: 10px;
    }}
    .iq-controls {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      align-items: center;
      margin-bottom: 12px;
      padding: 10px 12px;
      background: #f7f9fc;
      border: 1px solid #dce3ec;
      border-radius: 8px;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #2563eb;
      cursor: pointer;
    }}
    .iq-label {{
      min-width: 190px;
      text-align: right;
      font-size: 15px;
      font-weight: 600;
      color: #17202a;
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .report {{
      background: #ffffff;
      border: 1px solid #d8dde6;
      border-radius: 8px;
      padding: 14px;
    }}
    .report h2 {{
      margin: 0 0 12px;
      font-size: 17px;
      font-weight: 650;
    }}
    .report-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .report-grid div {{
      background: #f7f9fc;
      border: 1px solid #dce3ec;
      border-radius: 8px;
      padding: 9px 10px;
    }}
    .report-grid span {{
      display: block;
      color: #526071;
      font-size: 12px;
      margin-bottom: 3px;
    }}
    .report-grid strong {{
      display: block;
      color: #17202a;
      font-size: 14px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      border-top: 1px solid #e4e9f1;
      padding: 7px 8px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: #526071;
      font-weight: 650;
      background: #fbfcfe;
    }}
    .lower-grid {{
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(300px, 0.85fr);
      gap: 14px;
      align-items: start;
    }}
    .resonator-frequency img {{
      max-height: 330px;
      object-fit: contain;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .lower-grid {{
        grid-template-columns: 1fr;
      }}
      .report-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 640px) {{
      .report-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{safe_title}</h1>
    <div class="meta">
      status={safe_status} | points={points_label} | latest amplitude={amplitude_label}
    </div>
  </header>
  <main class="grid">
    <figure>
      <div class="tabs">
        <button id="tab-iq" class="tab-button" type="button">IQ blobs</button>
        <button id="tab-kernel" class="tab-button" type="button">Kernels</button>
        <button id="tab-resonator" class="tab-button" type="button">Resonator</button>
      </div>
      <figcaption id="experiment-title">{experiment_title}</figcaption>
      <div class="iq-controls">
        <input
          id="experiment-slider"
          type="range"
          min="0"
          max="{slider_max}"
          value="{slider_value}"
          step="1"
          {slider_disabled}
        >
        <span id="experiment-label" class="iq-label">{selected_iq_label}</span>
      </div>
      <img id="experiment-image" src="{latest_experiment_src}" alt="Experiment figure by amplitude">
    </figure>
    <figure>
      <figcaption>Readout fidelity scan</figcaption>
      <img id="fidelity" src="{fidelity_src}" alt="Fidelity versus amplitude">
    </figure>
  </main>
  <section class="lower-grid">
    {self._report_html}
    <figure class="resonator-frequency">
      <figcaption>Resonator frequency drift</figcaption>
      <img id="resonator-frequency" src="{resonator_frequency_src}" alt="Resonator frequency drift from initial frequency versus readout amplitude">
    </figure>
  </section>
  <script>
    const iqItems = {iq_items_js};
    const kernelItems = {kernel_items_js};
    const resonatorItems = {resonator_items_js};
    const itemsByTab = {{ iq: iqItems, kernel: kernelItems, resonator: resonatorItems }};
    const titleByTab = {{ iq: "Acquired IQ blobs", kernel: "Kernel experiments", resonator: "Resonator results" }};
    const slider = document.getElementById("experiment-slider");
    const label = document.getElementById("experiment-label");
    const experimentImage = document.getElementById("experiment-image");
    const experimentTitle = document.getElementById("experiment-title");
    const fidelityImage = document.getElementById("fidelity");
    const tabButtons = {{
      iq: document.getElementById("tab-iq"),
      kernel: document.getElementById("tab-kernel"),
      resonator: document.getElementById("tab-resonator"),
    }};
    let activeTab = "{active_tab}";
    const sliderStoragePrefix = "readout-live-slider-index-";
    const tabStorageKey = "readout-live-active-tab";

    function activeItems() {{
      return itemsByTab[activeTab] || [];
    }}

    function clampIndex(index) {{
      const items = activeItems();
      return Math.max(0, Math.min(items.length - 1, index));
    }}

    function showIqIndex(index) {{
      const items = activeItems();
      if (!items.length) {{
        label.textContent = activeTab === "kernel" ? "waiting for kernel figures" : "waiting for IQ blobs";
        experimentImage.removeAttribute("src");
        slider.disabled = true;
        return;
      }}
      const clampedIndex = clampIndex(index);
      const item = items[clampedIndex];
      slider.min = "0";
      slider.max = String(Math.max(0, items.length - 1));
      slider.disabled = items.length <= 1;
      slider.value = String(clampedIndex);
      window.localStorage.setItem(sliderStoragePrefix + activeTab, String(clampedIndex));
      experimentImage.src = item.imageSrc;
      if (item.fidelitySrc) {{
        fidelityImage.src = item.fidelitySrc;
      }}
      label.textContent = item.label;
    }}

    function setActiveTab(tabName) {{
      if (!itemsByTab[tabName]) {{
        return;
      }}
      activeTab = tabName;
      window.localStorage.setItem(tabStorageKey, tabName);
      experimentTitle.textContent = titleByTab[activeTab];
      for (const [name, button] of Object.entries(tabButtons)) {{
        button.classList.toggle("active", name === activeTab);
      }}
      const storedIndex = Number(window.localStorage.getItem(sliderStoragePrefix + activeTab));
      showIqIndex(Number.isInteger(storedIndex) ? storedIndex : activeItems().length - 1);
    }}

    for (const [name, button] of Object.entries(tabButtons)) {{
      button.disabled = !itemsByTab[name].length;
      button.addEventListener("click", () => setActiveTab(name));
    }}

    const storedTab = window.localStorage.getItem(tabStorageKey);
    if (storedTab && itemsByTab[storedTab] && itemsByTab[storedTab].length) {{
      activeTab = storedTab;
    }}
    if (!itemsByTab[activeTab].length && kernelItems.length) {{
      activeTab = "kernel";
    }}
    if (!itemsByTab[activeTab].length && resonatorItems.length) {{
      activeTab = "resonator";
    }}
    setActiveTab(activeTab);

    slider.addEventListener("input", () => {{
      showIqIndex(Number(slider.value));
    }});

    window.addEventListener("keydown", (event) => {{
      if (!activeItems().length) {{
        return;
      }}
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {{
        return;
      }}
      event.preventDefault();
      const step = event.key === "ArrowRight" ? 1 : -1;
      showIqIndex(Number(slider.value) + step);
    }});
{auto_refresh_script}
  </script>
</body>
</html>
""",
            encoding="utf-8",
        )

    def _image_src(self, relative_path: str, embed_images: bool) -> str:
        if not relative_path:
            return ""

        if not embed_images:
            return f"{relative_path}?v={self._version}"

        path = self.output_dir / relative_path
        if not path.exists():
            return ""

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
