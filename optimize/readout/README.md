# Readout Optimization Layout

This package is organized around the reusable optimizer and workflow code.

## Core

- `readout_workflow.py`: single-amplitude readout fidelity workflow.
- `validation.py`: post-optimizer validation against a profile.
- `optimizer/`: readout amplitude optimizer, settings, scan strategies, plotting,
  live HTML, artifacts, submit-only manifests, and metric extraction.
- `utils/`: shared utility code that is not optimizer-specific.

These modules are importable library code and should stay free of one-off lab
entrypoint behavior where possible.

## Scripts

Runnable reports and narrow lab utilities live in `scripts/`:

- `scripts/optimizer/run_optimizer.py`
- `scripts/optimizer/collect_optimizer_results.py`
- `scripts/validation/validate_optimizer_iq_sweep.py`
- `scripts/validation/active_reset_comparison.py`
- `scripts/reports/all_qubits_report.py`
- `scripts/reports/multiplexed_iq_blob_report.py`

Old one-off script files were moved out of `optimize/readout/`; use the
`scripts/` paths directly.

## Docs

- `docs/submit_only_results.md`: submit-only sweep mode, task manifests,
  low-priority submissions, status checks, and later result collection.
