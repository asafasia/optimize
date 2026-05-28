from __future__ import annotations

from typing import Any

import numpy as np


class ReadoutSweepScan:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def run(self) -> None:
        amplitudes = [
            float(amplitude)
            for amplitude in self.optimizer.settings.amplitudes
        ]

        for index, amplitude in enumerate(amplitudes, start=1):
            self.optimizer._show_progress(index, len(amplitudes), amplitude)
            self.optimizer._measure_amplitude(amplitude)


class ReadoutGradientAscentScan:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def run(self) -> None:
        candidates = [
            float(amplitude)
            for amplitude in self.optimizer.settings.amplitudes
        ]
        max_measurements = len(candidates)
        lower_bound = min(0.0, min(candidates))
        upper_bound = max(candidates)
        current_amplitude = 0.0
        step = self._initial_step(candidates)
        total = min(
            self.optimizer.settings.gradient_max_iterations,
            max_measurements,
        )
        previous_best_score: float | None = None

        for iteration in range(1, total + 1):
            if len(self.optimizer.measured_amplitudes) >= max_measurements:
                break

            self.optimizer._show_progress(iteration, total, current_amplitude)

            trial_amplitudes = [
                self._clip(current_amplitude - step, lower_bound, upper_bound),
                self._clip(current_amplitude, lower_bound, upper_bound),
                self._clip(current_amplitude + step, lower_bound, upper_bound),
            ]
            trial_amplitudes = self._limit_new_measurements(
                trial_amplitudes,
                max_measurements,
            )
            if not trial_amplitudes:
                break

            scores = {
                amplitude: self.optimizer._measure_amplitude(amplitude)
                for amplitude in trial_amplitudes
            }

            best_amplitude = max(scores, key=scores.get)
            best_score = scores[best_amplitude]
            if (
                previous_best_score is not None
                and abs(best_score - previous_best_score)
                < self.optimizer.settings.gradient_fidelity_tolerance
            ):
                break

            previous_best_score = best_score
            if best_amplitude == current_amplitude:
                step *= 0.5
            else:
                current_amplitude = best_amplitude

            if step < self.optimizer.settings.gradient_min_step:
                break

    def _initial_step(self, amplitudes: list[float]) -> float:
        if self.optimizer.settings.gradient_initial_step is not None:
            return float(self.optimizer.settings.gradient_initial_step)

        if len(amplitudes) < 2:
            return self.optimizer.settings.gradient_min_step

        return float(np.median(np.diff(sorted(amplitudes))))

    def _clip(
        self,
        amplitude: float,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        return float(np.clip(amplitude, lower_bound, upper_bound))

    def _limit_new_measurements(
        self,
        amplitudes: list[float],
        max_measurements: int,
    ) -> list[float]:
        unique_amplitudes = list(dict.fromkeys(amplitudes))
        remaining = max_measurements - len(self.optimizer.measured_amplitudes)
        limited_amplitudes = []

        for amplitude in unique_amplitudes:
            if amplitude in self.optimizer.results:
                limited_amplitudes.append(amplitude)
                continue

            if remaining <= 0:
                continue

            limited_amplitudes.append(amplitude)
            remaining -= 1

        return limited_amplitudes


def scan_method_for(optimizer: Any) -> ReadoutSweepScan | ReadoutGradientAscentScan:
    if optimizer.settings.method == "sweep":
        return ReadoutSweepScan(optimizer)
    if optimizer.settings.method == "gradient":
        return ReadoutGradientAscentScan(optimizer)

    raise ValueError(f"Unsupported scan method: {optimizer.settings.method}")
