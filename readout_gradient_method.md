# Readout Gradient Method

This document explains the current primitive `method="gradient"` behavior in
`ReadoutAmplitudeSweepWorkflow`.

## Purpose

The gradient method tries to find a readout amplitude with high readout fidelity
without measuring every amplitude in the configured sweep list.

The configured `amplitudes` are still important, but in gradient mode they are
used mainly as:

- the allowed amplitude range
- the maximum number of measurements allowed
- the default scale for the initial step size

## Settings

Example:

```python
optimizer_settings = ReadoutAmplitudeSweepSettings(
    amplitudes=np.linspace(0.001, 0.1, 10),
    method="gradient",
    gradient_max_iterations=5,
    gradient_initial_step=0.01,
    gradient_min_step=0.001,
    gradient_fidelity_tolerance=0.01,
    workflow_settings=workflow_settings,
)
```

### `method`

Use:

```python
method="gradient"
```

to select the primitive gradient ascent scan.

### `amplitudes`

In gradient mode, `amplitudes` defines:

- `upper_bound = max(amplitudes)`
- `lower_bound = min(0.0, min(amplitudes))`
- maximum number of measured amplitudes: `len(amplitudes)`

For example:

```python
amplitudes=np.linspace(0.001, 0.1, 10)
```

means the gradient scan can measure at most 10 unique amplitudes and will not go
above `0.1`.

### `gradient_max_iterations`

Maximum number of gradient iterations.

Each iteration may measure up to three amplitudes:

- `current - step`
- `current`
- `current + step`

The actual number can be smaller if an amplitude was already measured or if the
maximum measurement count was reached.

### `gradient_initial_step`

Initial step size.

If this is `None`, the code uses the median spacing of the configured
`amplitudes`.

Example:

```python
amplitudes=np.linspace(0.001, 0.1, 10)
```

gives a default step close to `0.011`.

### `gradient_min_step`

The scan stops when the step size becomes smaller than this value.

This happens when the current amplitude is already better than both neighboring
trial points, so the algorithm halves the step.

### `gradient_fidelity_tolerance`

The scan stops when the best fidelity improvement between iterations is smaller
than this value.

Default:

```python
gradient_fidelity_tolerance=0.01
```

This means the scan stops if the best fidelity changes by less than 1 percentage
point.

## Algorithm

The current implementation is intentionally simple.

1. Start at amplitude `0.0`.
2. Choose an initial step.
3. Measure up to three amplitudes:

   ```text
   current - step
   current
   current + step
   ```

4. Clip amplitudes to the allowed bounds.
5. Skip amplitudes that were already measured.
6. Pick the amplitude with the best mean fidelity across all selected qubits.
7. If the best amplitude is the current amplitude, halve the step.
8. Otherwise, move `current` to the best amplitude.
9. Stop when one of the stop conditions is reached.

## Score

For each measured amplitude, the score is:

```python
mean(readout_fidelity for each qubit)
```

For one qubit, this is simply that qubit's readout fidelity.

For multiple qubits, the optimizer chooses the amplitude with the best average
fidelity.

## Stop Conditions

Gradient mode stops when any of these happens:

1. `gradient_max_iterations` is reached.
2. The number of unique measured amplitudes reaches `len(amplitudes)`.
3. The step is smaller than `gradient_min_step`.
4. The change in best fidelity is smaller than `gradient_fidelity_tolerance`.

## Output

The workflow still saves and plots the actual measured amplitudes:

```python
optimizer.measured_amplitudes
optimizer.fidelities
optimizer.results
```

The saved `data.npz`, `fidelities.csv`, `summary.json`, and `plot.png` are based
on the amplitudes that were really measured, not the original configured sweep
list.

## Limitations

This is not a robust optimizer yet. It is a first primitive version.

Known limitations:

- It can stop early if the fidelity is noisy.
- It only tries one-dimensional amplitude changes.
- It assumes higher mean fidelity is always better.
- It does not fit a curve.
- It does not estimate uncertainty.
- It does not retry suspicious measurements.
- It may miss a better amplitude if the fidelity landscape is not smooth.

## Future Improvements

Useful next steps:

- repeat each amplitude and average the fidelity
- add smoothing or robust scoring
- do a coarse sweep first, then gradient/refinement around the best point
- fit a curve around the best region
- stop only after several small improvements in a row
- warn when the best point is at a boundary
- support separate best amplitudes per qubit
