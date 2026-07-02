# Codex Instructions

This is the focused working folder for Qarakal experiments and optimization.

## Active Areas

- `experiments/` for runnable experiment scripts
- `notebooks/` for exploratory notebooks
- `optimize/` for optimization workflows and helper code
- `outputs/` for generated figures, logs, and result files

## Parent Package Repositories

The package repositories live one directory up. Treat them as dependencies or
reference code unless the user explicitly asks to edit them.

- `../qhipu-lab/`
- `../qigeon/`
- `../qratena/`
- `../q-b2c/`

When code from those packages is needed, inspect or import it first. Do not
make package changes as part of normal experiment work without confirmation.

## Data

Large data and generated outputs also live mostly one directory up. Avoid
scanning them unless the task specifically needs them.

- `../DATA/`
- `../WIP/`
- `../laboneq_output/`
- `outputs/`
