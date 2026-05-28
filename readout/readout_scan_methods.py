from __future__ import annotations

from typing import Any

import numpy as np

from optimize.readout.readout_scan_types import ReadoutScanMethod


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


class ReadoutGoldenSectionScan:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def run(self) -> None:
        candidates = [
            float(amplitude)
            for amplitude in self.optimizer.settings.amplitudes
        ]
        max_measurements = len(candidates)
        lower_bound = min(candidates)
        upper_bound = max(candidates)
        total = min(
            self.optimizer.settings.golden_section_max_iterations,
            max_measurements,
        )
        previous_best_score: float | None = None

        self._measure_if_possible(lower_bound, max_measurements, 1, total)
        self._measure_if_possible(upper_bound, max_measurements, 2, total)

        for iteration in range(3, total + 1):
            if len(self.optimizer.measured_amplitudes) >= max_measurements:
                break
            if (
                upper_bound - lower_bound
                < self.optimizer.settings.golden_section_interval_tolerance
            ):
                break

            left_probe, right_probe = self._golden_probes(lower_bound, upper_bound)
            self._measure_if_possible(left_probe, max_measurements, iteration, total)
            if len(self.optimizer.measured_amplitudes) >= max_measurements:
                break
            self._measure_if_possible(
                right_probe,
                max_measurements,
                iteration,
                total,
            )

            best_amplitude, best_score = self._best_measured_point()
            if (
                previous_best_score is not None
                and abs(best_score - previous_best_score)
                < self.optimizer.settings.gradient_fidelity_tolerance
            ):
                break

            previous_best_score = best_score
            lower_bound, upper_bound = self._shrink_interval(
                lower_bound,
                upper_bound,
                best_amplitude,
            )

    def _measure_if_possible(
        self,
        amplitude: float,
        max_measurements: int,
        index: int,
        total: int,
    ) -> None:
        if (
            amplitude not in self.optimizer.results
            and len(self.optimizer.measured_amplitudes) >= max_measurements
        ):
            return

        self.optimizer._show_progress(index, total, amplitude)
        self.optimizer._measure_amplitude(amplitude)

    def _golden_probes(
        self,
        lower_bound: float,
        upper_bound: float,
    ) -> tuple[float, float]:
        golden_ratio = (np.sqrt(5) - 1) / 2
        interval = upper_bound - lower_bound
        left_probe = upper_bound - golden_ratio * interval
        right_probe = lower_bound + golden_ratio * interval
        return float(left_probe), float(right_probe)

    def _best_measured_point(self) -> tuple[float, float]:
        scores = {
            amplitude: self.optimizer._score_result(result)
            for amplitude, result in self.optimizer.results.items()
        }
        best_amplitude = max(scores, key=scores.get)
        return best_amplitude, scores[best_amplitude]

    def _shrink_interval(
        self,
        lower_bound: float,
        upper_bound: float,
        best_amplitude: float,
    ) -> tuple[float, float]:
        measured = sorted(self.optimizer.measured_amplitudes)
        best_index = measured.index(best_amplitude)
        new_lower = measured[max(0, best_index - 1)]
        new_upper = measured[min(len(measured) - 1, best_index + 1)]

        if new_lower == new_upper:
            return lower_bound, upper_bound

        return new_lower, new_upper


def scan_method_for(
    optimizer: Any,
) -> ReadoutSweepScan | ReadoutGradientAscentScan | ReadoutGoldenSectionScan:
    method = ReadoutScanMethod(optimizer.settings.method)

    if method == ReadoutScanMethod.SWEEP:
        return ReadoutSweepScan(optimizer)
    if method == ReadoutScanMethod.GRADIENT:
        return ReadoutGradientAscentScan(optimizer)
    if method == ReadoutScanMethod.GOLDEN_SECTION:
        return ReadoutGoldenSectionScan(optimizer)

    raise ValueError(f"Unsupported scan method: {method}")
