# Qarakal Workbench

This folder is the focused day-to-day workbench for Qarakal experiments,
analysis, and optimization. The package repositories live one directory up and
are treated as editable dependencies unless a task explicitly calls for package
changes.

## Layout

- `experiments/` - runnable experiment scripts and quick hardware workflows
- `notebooks/` - exploratory notebooks
- `optimize/` - readout optimization workflows and helper code
- `resources/` - local workbench bootstrap and profile/task-manager access
- `outputs/` - generated figures, logs, and result files

## Repository Boundaries

The workbench root is the operational workspace. The `optimize/` directory is a
nested Git repository with its own history. Keep that split intentional:

- Workbench-wide setup, docs, quality gates, credentials, and generated-output
  policy live at the root.
- Optimization workflow source and tests live under `optimize/`.
- Parent package repositories such as `../qhipu-lab`, `../qigeon`,
  `../qratena`, and `../q-b2c` are dependencies/reference code. Do not edit
  them during normal workbench changes without explicitly choosing to do so.

## Environment

Use Python 3.12 or newer. The local source checkouts one directory up are added
to `sys.path` by `workbench_bootstrap.setup_workbench_environment()` and by the
pytest config in `pyproject.toml`.

Required secret values are read from environment variables. Start from:

```bash
cp .env.example .env
```

Then load those values into your shell before running hardware/profile code.
Real `.env` files are ignored and must not be committed.

Required variables:

- `QTASKBOARD_USERNAME`
- `QTASKBOARD_PASSWORD`
- `QARAKAL_MONGO_URI`
- `QARAKAL_BLOB_CONN_STR`

Optional variables with defaults:

- `QTASKBOARD_API_URI`
- `QTASKBOARD_REDIS_URI`
- `QARAKAL_DB_NAME`
- `QARAKAL_BLOB_CONTAINER`

## Quality Gates

The standard commands are:

```bash
make test
make lint
make format
make check-secrets
```

`make check-secrets` expects `gitleaks` to be installed. The pre-commit config
also runs Ruff and Gitleaks for teams that use `pre-commit`.

## Running Workflows

Prefer explicit runner scripts over importing modules interactively. Examples:

```bash
python3 run_with_workbench.py optimize/readout/scripts/run_all_qubits_report.py --help
python3 run_with_workbench.py optimize/readout/scripts/run_all_qubit_iq_blobs.py --help
```

Reusable modules such as `optimize/readout/readout_workflow.py` and
`optimize/readout/readout_amplitude_optimizer.py` are library code. Import their
classes from a runner or notebook; do not use them as ad-hoc executable files.

## Generated Artifacts

Generated outputs belong under `outputs/`, `data/`, or `laboneq_output/`.
Bytecode caches, logs, plots, binary arrays, rendered reports, and local env
files are ignored. Clean transient files with:

```bash
make clean-artifacts
```

Keep small, intentional examples in source control only when they are useful as
fixtures or documentation.
