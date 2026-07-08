# Readout Optimization Layout

This package is organized around the reusable optimizer and workflow code.

## Core

- `readout_amplitude_optimizer.py`: main readout amplitude optimizer.
- `optimizer_settings.py`: optimizer settings dataclass.
- `profile_access.py`: profile read/write helpers for readout parameters.
- `optimizer_metrics.py`: fidelity, separation, roundness, and resonator metric extraction.
- `optimizer_figures.py`: figure extraction from nested workflows and handlers.
- `submitted_runs.py`: submit-only manifests, status checks, and later result collection.
- `readout_workflow.py`: single-amplitude readout fidelity workflow.
- `utils/`: scan strategies, plotting, analysis, live HTML, and artifact helpers.

These modules are importable library code and should stay free of one-off lab
entrypoint behavior where possible.

## Scripts

Runnable reports and narrow lab utilities live in `scripts/`:

- `scripts/run_all_qubits_report.py`
- `scripts/run_all_qubits.py`
- `scripts/run_all_qubit_iq_blobs.py`
- `scripts/run_all_qubit_iq_blobs_experiment.py`
- `scripts/run_iq_blobs_active_reset_comparison.py`
- `scripts/measure_all_qubit_multiplexed_iq_blob_fidelities.py`
- `scripts/load_optimizer_results.py`

Old one-off script files were moved out of `optimize/readout/`; use the
`scripts/` paths directly.

## Docs

- `docs/submit_only_results.md`: submit-only sweep mode, task manifests,
  low-priority submissions, status checks, and later result collection.
