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
        self.selected_amplitude: float | None = None

    def plot(self) -> Figure:
        fig, (fidelity_ax, separation_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.5, 7.0),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        amplitudes = [float(amplitude) for amplitude in self.amplitudes]
        sorted_indices = np.argsort(amplitudes)
        sorted_amplitudes = [amplitudes[index] for index in sorted_indices]
        selected_label_added = False

        for qubit_name in self.qubit_names:
            measured_fidelities = [
                float(value) for value in self.fidelities[qubit_name]
            ]
            fidelity_values = [
                measured_fidelities[index] for index in sorted_indices
            ]
            best_index = int(np.argmax(fidelity_values))
            best_amplitude = sorted_amplitudes[best_index]
            best_fidelity = fidelity_values[best_index]

            line = fidelity_ax.plot(
                sorted_amplitudes,
                fidelity_values,
                marker="o",
                label=f"Qubit {qubit_name}",
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
                    f"Best {qubit_name}",
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
                        f"Init {qubit_name}",
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
                selected_label = None
                if not selected_label_added:
                    selected_label = f"Selected A={selected_amplitude:.4g}"
                    selected_label_added = True
                self._plot_selected_point(
                    fidelity_ax,
                    selected_amplitude,
                    selected_fidelity,
                    size=175,
                    label=selected_label,
                )

            separation_values = self._sorted_optional_values(
                self.separations.get(qubit_name, []),
                sorted_indices,
            )
            if any(value is not None for value in separation_values):
                numeric_separations = [
                    np.nan if value is None else value for value in separation_values
                ]
                separation_ax.plot(
                    sorted_amplitudes,
                    numeric_separations,
                    marker="o",
                    color=color,
                    label=f"Qubit {qubit_name}",
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

        fidelity_ax.set_title(self._title())
        fidelity_ax.set_ylabel("Readout Fidelity")
        fidelity_ax.set_ylim(0.5, 1.0)
        fidelity_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        fidelity_ax.legend()
        separation_ax.set_xlabel("Readout Amplitude")
        separation_ax.set_ylabel("Separation")
        separation_ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        if separation_ax.has_data():
            separation_ax.legend()
        fig.tight_layout()

        return fig

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
