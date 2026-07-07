from __future__ import annotations

from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np


class ReadoutAmplitudeSweepPlotter:
    def __init__(
        self,
        qubit_names: list[str],
        amplitudes: Any,
        fidelities: dict[str, list[float]],
        initial_amplitudes: dict[str, float] | None = None,
        readout_lengths: dict[str, float] | None = None,
    ) -> None:
        self.qubit_names = qubit_names
        self.amplitudes = amplitudes
        self.fidelities = fidelities
        self.initial_amplitudes = initial_amplitudes or {}
        self.readout_lengths = readout_lengths or {}
        self.reset_label: str | None = None
        self.fidelity_errors: dict[str, list[float | None]] = {}
        self.separations: dict[str, list[float | None]] = {}
        self.roundnesses: dict[str, list[float | None]] = {}
        self.selected_amplitude: float | None = None

    def plot(self) -> Figure:
        amplitudes = [float(amplitude) for amplitude in self.amplitudes]
        sorted_indices = np.argsort(amplitudes)
        sorted_amplitudes = [amplitudes[index] for index in sorted_indices]

        if len(self.qubit_names) == 1:
            return self._plot_single_figure(sorted_amplitudes, sorted_indices)

        return self._plot_qubit_panels(sorted_amplitudes, sorted_indices)

    def _plot_single_figure(
        self,
        sorted_amplitudes: list[float],
        sorted_indices: np.ndarray,
    ) -> Figure:
        fig, (fidelity_ax, separation_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.5, 7.0),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        self._plot_qubit_sweep(
            fidelity_ax,
            separation_ax,
            qubit_name=self.qubit_names[0],
            sorted_amplitudes=sorted_amplitudes,
            sorted_indices=sorted_indices,
            include_qubit_in_label=True,
        )
        self._plot_roundness(
            fidelity_ax,
            self.qubit_names[0],
            sorted_amplitudes,
            sorted_indices,
        )
        self._format_axes(
            fidelity_ax,
            separation_ax,
            title=self._title(),
            show_xlabel=True,
        )
        fig.tight_layout()
        return fig

    def _plot_qubit_panels(
        self,
        sorted_amplitudes: list[float],
        sorted_indices: np.ndarray,
    ) -> Figure:
        qubit_count = len(self.qubit_names)
        fig, axes = plt.subplots(
            qubit_count,
            2,
            sharex=True,
            figsize=(12.0, max(4.2, 3.2 * qubit_count)),
            gridspec_kw={"width_ratios": [2, 1]},
            squeeze=False,
        )
        fig.suptitle(self._title(), fontsize=13)

        for row_index, qubit_name in enumerate(self.qubit_names):
            fidelity_ax = axes[row_index][0]
            separation_ax = axes[row_index][1]
            self._plot_qubit_sweep(
                fidelity_ax,
                separation_ax,
                qubit_name=qubit_name,
                sorted_amplitudes=sorted_amplitudes,
                sorted_indices=sorted_indices,
                include_qubit_in_label=False,
            )
            self._plot_roundness(
                fidelity_ax,
                qubit_name,
                sorted_amplitudes,
                sorted_indices,
            )
            title = self._qubit_title(qubit_name)
            self._format_axes(
                fidelity_ax,
                separation_ax,
                title=title,
                show_xlabel=row_index == qubit_count - 1,
            )

        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        return fig

    def _plot_qubit_sweep(
        self,
        fidelity_ax,
        separation_ax,
        *,
        qubit_name: str,
        sorted_amplitudes: list[float],
        sorted_indices: np.ndarray,
        include_qubit_in_label: bool,
    ) -> None:
        measured_fidelities = [float(value) for value in self.fidelities[qubit_name]]
        fidelity_values = [measured_fidelities[index] for index in sorted_indices]
        best_index = int(np.argmax(fidelity_values))
        best_amplitude = sorted_amplitudes[best_index]
        best_fidelity = fidelity_values[best_index]
        qubit_label = f" {qubit_name}" if include_qubit_in_label else ""

        line = fidelity_ax.plot(
            sorted_amplitudes,
            fidelity_values,
            marker="o",
            linewidth=2.4,
            label=f"Qubit {qubit_name}" if include_qubit_in_label else "Fidelity",
            zorder=5,
        )[0]
        color = line.get_color()
        fidelity_errors = self._sorted_optional_values(
            self.fidelity_errors.get(qubit_name, []),
            sorted_indices,
        )
        if any(error is not None for error in fidelity_errors):
            lower, upper = self._error_band(fidelity_values, fidelity_errors)
            fidelity_ax.fill_between(
                sorted_amplitudes,
                lower,
                upper,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
            fidelity_ax.errorbar(
                sorted_amplitudes,
                fidelity_values,
                yerr=[0.0 if error is None else error for error in fidelity_errors],
                fmt="none",
                ecolor=color,
                alpha=0.45,
                capsize=3,
                linewidth=0.8,
            )

        fidelity_ax.scatter(
            [best_amplitude],
            [best_fidelity],
            color="green",
            marker="*",
            s=130,
            zorder=4,
        )
        fidelity_ax.axvline(
            best_amplitude,
            color="green",
            linestyle=":",
            linewidth=1.8,
            label=self._marker_label(
                f"Best{qubit_label}",
                best_amplitude,
                best_fidelity,
                approximate=False,
            ),
        )

        initial_amplitude = self.initial_amplitudes.get(qubit_name)
        if initial_amplitude is not None:
            initial_fidelity = self._fidelity_at_amplitude(
                initial_amplitude,
                sorted_amplitudes,
                fidelity_values,
            )
            fidelity_ax.scatter(
                [initial_amplitude],
                [initial_fidelity],
                color="red",
                marker="o",
                s=70,
                zorder=4,
            )
            fidelity_ax.axvline(
                initial_amplitude,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=self._marker_label(
                    f"Init{qubit_label}",
                    initial_amplitude,
                    initial_fidelity,
                    approximate=True,
                ),
            )

        if self.selected_amplitude is not None:
            selected_amplitude = float(self.selected_amplitude)
            selected_fidelity = self._fidelity_at_amplitude(
                selected_amplitude,
                sorted_amplitudes,
                fidelity_values,
            )
            self._plot_selected_point(
                fidelity_ax,
                selected_amplitude,
                selected_fidelity,
                size=175,
                label=f"Selected A={selected_amplitude:.4g}",
            )

        self._plot_separation_sweep(
            separation_ax,
            qubit_name=qubit_name,
            sorted_amplitudes=sorted_amplitudes,
            sorted_indices=sorted_indices,
            best_amplitude=best_amplitude,
            color=color,
            include_qubit_in_label=include_qubit_in_label,
        )

    def _plot_separation_sweep(
        self,
        separation_ax,
        *,
        qubit_name: str,
        sorted_amplitudes: list[float],
        sorted_indices: np.ndarray,
        best_amplitude: float,
        color: str,
        include_qubit_in_label: bool,
    ) -> None:
        separation_values = self._sorted_optional_values(
            self.separations.get(qubit_name, []),
            sorted_indices,
        )
        if not any(value is not None for value in separation_values):
            return

        numeric_separations = [
            np.nan if value is None else value for value in separation_values
        ]
        separation_ax.plot(
            sorted_amplitudes,
            numeric_separations,
            marker="o",
            color=color,
            label=f"Qubit {qubit_name}" if include_qubit_in_label else "Separation",
        )
        best_separation = self._optional_value_at_amplitude(
            best_amplitude,
            sorted_amplitudes,
            numeric_separations,
        )
        if best_separation is not None:
            separation_ax.scatter(
                [best_amplitude],
                [best_separation],
                color="green",
                marker="*",
                s=95,
                zorder=4,
            )
        separation_ax.axvline(
            best_amplitude,
            color="green",
            linestyle=":",
            linewidth=1.5,
        )

        initial_amplitude = self.initial_amplitudes.get(qubit_name)
        if initial_amplitude is not None:
            initial_separation = self._optional_value_at_amplitude(
                initial_amplitude,
                sorted_amplitudes,
                numeric_separations,
            )
            if initial_separation is not None:
                separation_ax.scatter(
                    [initial_amplitude],
                    [initial_separation],
                    color="red",
                    marker="o",
                    s=55,
                    zorder=4,
                )
            separation_ax.axvline(
                initial_amplitude,
                color="red",
                linestyle="--",
                linewidth=1.2,
            )

        if self.selected_amplitude is not None:
            selected_amplitude = float(self.selected_amplitude)
            selected_separation = self._optional_value_at_amplitude(
                selected_amplitude,
                sorted_amplitudes,
                numeric_separations,
            )
            if selected_separation is not None:
                self._plot_selected_point(
                    separation_ax,
                    selected_amplitude,
                    selected_separation,
                    size=135,
                )

    def _plot_roundness(
        self,
        fidelity_ax,
        qubit_name: str,
        sorted_amplitudes: list[float],
        sorted_indices: np.ndarray,
    ) -> None:
        roundness_values = self._sorted_optional_values(
            self.roundnesses.get(qubit_name, []),
            sorted_indices,
        )
        if not any(value is not None for value in roundness_values):
            return

        fidelity_ax.plot(
            sorted_amplitudes,
            [np.nan if value is None else value for value in roundness_values],
            color="red",
            marker="s",
            linewidth=1.5,
            alpha=0.75,
            label="Roundness",
            zorder=2,
        )

    def _format_axes(
        self,
        fidelity_ax,
        separation_ax,
        *,
        title: str,
        show_xlabel: bool,
    ) -> None:
        fidelity_ax.set_title(title)
        fidelity_ax.set_ylabel("Readout Fidelity")
        fidelity_ax.set_ylim(0.5, 1.0)
        fidelity_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        fidelity_ax.legend()
        if show_xlabel:
            separation_ax.set_xlabel("Readout Amplitude")
        separation_ax.set_ylabel("Separation")
        separation_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        if separation_ax.has_data():
            separation_ax.legend()

    def _plot_selected_point(
        self,
        axis,
        amplitude: float,
        value: float,
        size: int,
        label: str | None = None,
    ) -> None:
        axis.scatter(
            [amplitude],
            [value],
            color="#2563eb",
            marker="o",
            s=max(32, int(size * 0.24)),
            zorder=5,
        )
        axis.scatter(
            [amplitude],
            [value],
            facecolors="none",
            edgecolors="#2563eb",
            marker="o",
            s=size,
            linewidths=2.2,
            zorder=6,
            label=label,
        )

    def _sorted_optional_values(
        self,
        values: list[float | None],
        sorted_indices: np.ndarray,
    ) -> list[float | None]:
        if len(values) != len(sorted_indices):
            return [None for _ in sorted_indices]

        return [values[index] for index in sorted_indices]

    def _average_optional_values(
        self,
        metrics: dict[str, list[float | None]],
        sorted_indices: np.ndarray,
    ) -> list[float | None]:
        averages = []
        for index in sorted_indices:
            values = [
                float(qubit_values[index])
                for qubit_values in metrics.values()
                if index < len(qubit_values) and qubit_values[index] is not None
            ]
            averages.append(float(np.mean(values)) if values else None)
        return averages

    def _error_band(
        self,
        values: list[float],
        errors: list[float | None],
    ) -> tuple[list[float], list[float]]:
        lower = []
        upper = []
        for value, error in zip(values, errors):
            error_value = 0.0 if error is None else float(error)
            lower.append(value - error_value)
            upper.append(value + error_value)
        return lower, upper

    def _fidelity_at_amplitude(
        self,
        amplitude: float,
        amplitudes: list[float],
        fidelities: list[float],
    ) -> float:
        if amplitude <= amplitudes[0]:
            return fidelities[0]
        if amplitude >= amplitudes[-1]:
            return fidelities[-1]

        return float(np.interp(amplitude, amplitudes, fidelities))

    def _optional_value_at_amplitude(
        self,
        amplitude: float,
        amplitudes: list[float],
        values: list[float],
    ) -> float | None:
        finite_points = [
            (x, y)
            for x, y in zip(amplitudes, values)
            if not np.isnan(y)
        ]
        if not finite_points:
            return None

        finite_amplitudes = [point[0] for point in finite_points]
        finite_values = [point[1] for point in finite_points]
        if amplitude <= finite_amplitudes[0]:
            return finite_values[0]
        if amplitude >= finite_amplitudes[-1]:
            return finite_values[-1]

        return float(np.interp(amplitude, finite_amplitudes, finite_values))

    def _marker_label(
        self,
        prefix: str,
        amplitude: float,
        fidelity: float,
        approximate: bool,
    ) -> str:
        fidelity_prefix = "~" if approximate else ""
        return f"{prefix}: A={amplitude:.4g}, F{fidelity_prefix}={fidelity:.3f}"

    def _title(self) -> str:
        qubits = ", ".join(self.qubit_names)
        lengths = [
            f"{self.readout_lengths[qubit_name] * 1e9:.0f} ns"
            for qubit_name in self.qubit_names
            if qubit_name in self.readout_lengths
        ]

        if not lengths:
            title_parts = ["Readout Fidelity vs Amplitude", qubits]
        else:
            length_label = ", ".join(lengths)
            title_parts = [
                "Readout Fidelity vs Amplitude",
                qubits,
                length_label,
            ]

        if self.reset_label:
            title_parts.append(self.reset_label)

        return " - ".join(title_parts)

    def _qubit_title(self, qubit_name: str) -> str:
        title_parts = [f"Qubit {qubit_name}"]
        if qubit_name in self.readout_lengths:
            title_parts.append(f"{self.readout_lengths[qubit_name] * 1e9:.0f} ns")
        if self.reset_label:
            title_parts.append(self.reset_label)
        return " - ".join(title_parts)
