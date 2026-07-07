from __future__ import annotations

from typing import Any

import numpy as np


class ReadoutMetricsMixin:
    def _score_result(self, result: dict[str, Any]) -> float:
        if "iq_blobs" not in result:
            return float(self.settings.failed_measurement_fidelity)

        iq_results = result["iq_blobs"]
        return float(
            np.mean(
                [
                    iq_results[qubit_name]["readout_fidelity"]
                    for qubit_name in self.qubit_names
                ]
            )
        )

    def _record_failed_measurement(
        self,
        amplitude: float,
        error: Exception,
    ) -> float:
        error_message = f"{type(error).__name__}: {error}"
        self.results[amplitude] = {"error": error_message}
        self.measurement_errors[amplitude] = error_message
        self.measured_amplitudes.append(amplitude)

        for qubit_name in self.qubit_names:
            self.fidelities[qubit_name].append(
                float(self.settings.failed_measurement_fidelity)
            )
            self.fidelity_errors[qubit_name].append(None)
            self.separations[qubit_name].append(None)
            self.roundnesses[qubit_name].append(None)
            self.resonator_frequencies[qubit_name].append(None)

        print(
            f"\nMeasurement failed at amplitude {amplitude:.6g}; "
            f"recorded fidelity={self.settings.failed_measurement_fidelity}. "
            f"{error_message}"
        )
        self._update_live_plotter(latest_amplitude=amplitude)
        return float(self.settings.failed_measurement_fidelity)

    def _record_fidelities(self, result: dict[str, Any]) -> None:
        iq_results = result["iq_blobs"]

        for qubit_name in self.qubit_names:
            qubit_result = iq_results[qubit_name]
            fidelity = qubit_result["readout_fidelity"]
            self.fidelities[qubit_name].append(fidelity)
            self.fidelity_errors[qubit_name].append(
                self._first_metric_value(
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
            )
            self.separations[qubit_name].append(
                self._first_metric_value(
                    qubit_result,
                    [
                        "separation",
                        "readout_separation",
                        "iq_separation",
                        "state_separation",
                    ],
                )
            )
            self.roundnesses[qubit_name].append(
                self._first_metric_value(
                    qubit_result,
                    [
                        "average_roundness",
                        "averaged_roundness",
                        "roundness",
                    ],
                )
            )

    def _record_resonator_frequencies(self, workflow: Any) -> None:
        handlers = self._workflow_handlers(workflow, "resonator")
        for qubit_name in self.qubit_names:
            frequency = None
            for handler in handlers:
                handler_data = getattr(handler, "data", {}) or {}
                qubit_data = handler_data.get(qubit_name, {}) or {}
                if "optimal_resonance_freq" in qubit_data:
                    frequency = float(qubit_data["optimal_resonance_freq"])
                    break
            self.resonator_frequencies[qubit_name].append(frequency)

    def _first_metric_value(
        self,
        data: dict[str, Any],
        keys: list[str],
    ) -> float | None:
        for key in keys:
            if key in data and data[key] is not None:
                return float(data[key])
        return None
