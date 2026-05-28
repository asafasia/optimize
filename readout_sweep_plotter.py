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

    def plot(self) -> Figure:
        fig, ax = plt.subplots()
        amplitudes = [float(amplitude) for amplitude in self.amplitudes]
        sorted_indices = np.argsort(amplitudes)
        sorted_amplitudes = [amplitudes[index] for index in sorted_indices]

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

            ax.plot(
                sorted_amplitudes,
                fidelity_values,
                marker="o",
                label=f"Qubit {qubit_name}",
            )
            ax.scatter(
                [best_amplitude],
                [best_fidelity],
                color="green",
                marker="*",
                s=130,
                zorder=4,
            )
            ax.axvline(
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
                ax.scatter(
                    [initial_amplitude],
                    [initial_fidelity],
                    color="red",
                    marker="o",
                    s=70,
                    zorder=4,
                )
                ax.axvline(
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

        ax.set_title(self._title())
        ax.set_xlabel("Readout Amplitude")
        ax.set_ylabel("Readout Fidelity")
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax.legend()
        fig.tight_layout()

        return fig

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
            return f"Readout Fidelity vs Amplitude - {qubits}"

        length_label = ", ".join(lengths)
        return f"Readout Fidelity vs Amplitude - {qubits} - {length_label}"
