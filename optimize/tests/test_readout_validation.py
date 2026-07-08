from __future__ import annotations

import json
from types import SimpleNamespace

from optimize.readout import validation as validation_mod
from optimize.readout.validation import (
    ReadoutOptimizerValidation,
    ReadoutOptimizerValidationSettings,
)
from qratena.util.enums import SUPPORTED_PULSE_SHAPES, SUPPORTED_PULSE_TYPES


class FakeProfile:
    def __init__(self, qubit_names=("q1", "q2")):
        self.qubits = {}
        for qubit_name in qubit_names:
            readout = SimpleNamespace(readout_amplitude=0.01)
            self.qubits[qubit_name] = SimpleNamespace(
                pulses={
                    SUPPORTED_PULSE_TYPES.readout: {
                        SUPPORTED_PULSE_SHAPES.const: readout,
                    },
                },
            )


def test_validation_runs_optimizer_amplitudes_and_writes_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "optimizer_run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "qubits": ["q1", "q2"],
                "amplitudes": [0.02, 0.04],
                "best_mean_amplitude": 0.04,
            }
        ),
        encoding="utf-8",
    )
    seen_amplitudes = []

    class FakeWorkflow:
        def __init__(self, qubit_names, profile, task_manager, settings):
            self.qubit_names = qubit_names
            self.profile = profile
            self.settings = settings

        def run(self):
            pulse = self.profile.qubits["q1"].pulses[SUPPORTED_PULSE_TYPES.readout][
                SUPPORTED_PULSE_SHAPES.const
            ]
            seen_amplitudes.append(pulse.readout_amplitude)
            assert self.settings.run_kernels is True
            assert self.settings.run_iq_blobs is True
            return {
                "iq_blobs": {
                    qubit_name: {
                        "readout_fidelity": 0.9 + pulse.readout_amplitude,
                        "readout_fidelity_error": 0.01,
                        "separation": 3.0,
                    }
                    for qubit_name in self.qubit_names
                }
            }

    monkeypatch.setattr(validation_mod, "ReadoutFidelityWorkflow", FakeWorkflow)
    validator = ReadoutOptimizerValidation(
        optimizer_run_dir=run_dir,
        profile=FakeProfile(),
        task_manager=object(),
        settings=ReadoutOptimizerValidationSettings(do_emulation=True),
    )

    result = validator.run()

    assert seen_amplitudes == [0.02, 0.04]
    assert len(result["rows"]) == 4
    validation_dir = result["validation_dir"]
    assert (validation_dir / "readout_validation.csv").exists()
    assert (validation_dir / "readout_validation.json").exists()
    assert (validation_dir / "readout_validation.md").exists()
    assert (validation_dir / "readout_validation.png").exists()
