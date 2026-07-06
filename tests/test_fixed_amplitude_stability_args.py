from __future__ import annotations

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

import amplitude_stability.fixed_amplitude_stability as stability
from amplitude_stability.fixed_amplitude_stability import parse_args


def test_parse_args_ignores_ipykernel_argv(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ipykernel_launcher.py",
            "--f=/Users/example/Library/Jupyter/runtime/kernel-example.json",
        ],
    )

    args = parse_args()

    assert args.qubit == "q8"
    assert args.duration_min == 10.0
    assert args.point_parity == "odd"


def test_parse_args_keeps_cli_validation_strict():
    try:
        parse_args(["--not-a-real-option"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("parse_args accepted an unknown command-line option.")


def test_selected_repetition_mask_uses_odd_superposition_points():
    repetitions = np.arange(0, 10, 1)

    mask = stability.selected_repetition_mask(repetitions, "odd", drop_edges=False)

    assert repetitions[mask].tolist() == [1, 3, 5, 7, 9]


def test_save_plot_writes_repetition_time_heatmap(tmp_path, monkeypatch):
    monkeypatch.setattr(stability, "np", np, raising=False)
    monkeypatch.setattr(stability, "plt", plt, raising=False)
    rows = [
        {
            "timestamp": "2026-07-04T12:00:00",
            "run_index": run_index,
            "elapsed_s": elapsed_s,
            "qubit": "q4",
            "repetition": repetition,
            "point_parity": "odd",
            "value": 0.1 * run_index + 0.01 * repetition,
        }
        for run_index, elapsed_s in enumerate([0.0, 60.0])
        for repetition in [1, 3, 5]
    ]
    path = tmp_path / "stability.png"

    stability.save_plot(path, rows, np.asarray([1, 3, 5]), "odd")

    assert path.exists()
    assert path.stat().st_size > 0
