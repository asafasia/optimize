from __future__ import annotations

import json

from optimize.readout import readout_workflow
from optimize.readout.readout_workflow import ReadoutFidelityWorkflow


def test_deserialize_laboneq_result_fills_missing_real_data(monkeypatch):
    workflow = object.__new__(ReadoutFidelityWorkflow)
    raw_data = json.dumps(
        {
            "__data__": {
                "acquired_results": {
                    "handle_q3": {
                        "data.real": None,
                        "data.imag": [[1.0, 2.0], [3.0, 4.0]],
                    }
                }
            }
        }
    ).encode()
    calls = []

    def fake_from_json(data):
        calls.append(json.loads(data))
        if calls[-1]["__data__"]["acquired_results"]["handle_q3"]["data.real"] is None:
            raise TypeError("unsupported operand type(s) for +: 'NoneType' and 'complex'")
        return "decoded"

    monkeypatch.setattr(readout_workflow, "from_json", fake_from_json)

    result = workflow._deserialize_laboneq_result(raw_data)

    assert result == "decoded"
    assert len(calls) == 2
    assert calls[1]["__data__"]["acquired_results"]["handle_q3"]["data.real"] == [
        [0.0, 0.0],
        [0.0, 0.0],
    ]


def test_deserialize_laboneq_result_reraises_unrelated_type_errors(monkeypatch):
    workflow = object.__new__(ReadoutFidelityWorkflow)

    def fake_from_json(data):
        raise TypeError("different failure")

    monkeypatch.setattr(readout_workflow, "from_json", fake_from_json)

    try:
        workflow._deserialize_laboneq_result("{}")
    except TypeError as error:
        assert str(error) == "different failure"
    else:
        raise AssertionError("expected TypeError")
