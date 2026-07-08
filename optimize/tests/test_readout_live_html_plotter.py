from optimize.readout.optimizer.live_html_plotter import ReadoutLiveHtmlPlotter


def test_fidelity_history_plots_are_generated_only_once(tmp_path, monkeypatch):
    plotter = ReadoutLiveHtmlPlotter(tmp_path, open_browser=False)
    plotter.fidelity_history_dir.mkdir(parents=True)
    plotter._iq_history.append(
        {"label": "A=0.01", "amplitude": 0.01, "iq_src": "iq.png", "fidelity_src": ""}
    )
    saved_paths = []
    monkeypatch.setattr(
        plotter,
        "_save_fidelity_plot",
        lambda **kwargs: saved_paths.append(kwargs["output_path"]),
    )
    kwargs = {
        "qubit_names": ["q1"],
        "amplitudes": [0.01],
        "fidelities": {"q1": [0.9]},
        "fidelity_errors": {"q1": [None]},
        "separations": {"q1": [None]},
        "roundnesses": {"q1": [None]},
        "initial_amplitudes": {"q1": 0.02},
        "readout_lengths": {"q1": 1e-6},
        "reset_label": "active reset off",
    }

    plotter._save_fidelity_history_plots(**kwargs)
    plotter._save_fidelity_history_plots(**kwargs)

    assert len(saved_paths) == 1
    assert plotter._iq_history[0]["fidelity_src"] == "fidelity_history/fidelity_0001.png"
