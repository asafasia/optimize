from __future__ import annotations

from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES, ResetType


class ReadoutProfileAccessMixin:
    def _set_readout_amplitude(self, amplitude: float) -> None:
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout
            ][SUPPORTED_PULSE_SHAPES.const]
            readout_pulse.readout_amplitude = amplitude

    def _readout_amplitudes(self) -> dict[str, float]:
        amplitudes = {}
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout
            ][SUPPORTED_PULSE_SHAPES.const]
            amplitudes[qubit_name] = float(readout_pulse.readout_amplitude)
        return amplitudes

    def _readout_lengths(self) -> dict[str, float]:
        lengths = {}
        for qubit_name in self.qubit_names:
            readout_pulse = self.profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout
            ][SUPPORTED_PULSE_SHAPES.const]
            lengths[qubit_name] = float(readout_pulse.readout_duration)
        return lengths

    def _readout_frequencies(self) -> dict[str, float]:
        frequencies = {}
        for qubit_name in self.qubit_names:
            frequency = getattr(
                self.profile.qubits[qubit_name].readout_resonator_frequency,
                "value",
                None,
            )
            if frequency is not None:
                frequencies[qubit_name] = float(frequency)
        return frequencies

    def _reset_label(self) -> str:
        reset = self.settings.workflow_settings.reset
        if reset.reset_type == ResetType.ACTIVE:
            return f"active reset on ({reset.reset_num}x)"

        return "active reset off"
