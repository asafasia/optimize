from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np


class ReadoutAmplitudeSweepAnalysis:
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

    def summary(self) -> dict[str, Any]:
        amplitudes = [float(amplitude) for amplitude in self.amplitudes]
        qubit_summaries = {}

        for qubit_name, fidelities in self.fidelities.items():
            fidelity_values = [float(fidelity) for fidelity in fidelities]
            best_index = int(np.argmax(fidelity_values))
            qubit_summaries[qubit_name] = {
                "initial_amplitude": self.initial_amplitudes.get(qubit_name),
                "best_amplitude": amplitudes[best_index],
                "best_fidelity": fidelity_values[best_index],
                "final_fidelity": fidelity_values[-1],
                "fidelities": fidelity_values,
            }

        mean_fidelities = [
            float(
                np.mean(
                    [self.fidelities[qubit][index] for qubit in self.qubit_names]
                )
            )
            for index in range(len(amplitudes))
        ]
        best_mean_index = int(np.argmax(mean_fidelities))

        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "qubits": self.qubit_names,
            "amplitudes": amplitudes,
            "mean_fidelities": mean_fidelities,
            "best_mean_amplitude": amplitudes[best_mean_index],
            "best_mean_fidelity": mean_fidelities[best_mean_index],
            "qubit_summaries": qubit_summaries,
        }
