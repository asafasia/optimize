from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure


class ReadoutFigureMixin:
    def _record_iq_blob_figures(self, amplitude: float, workflow: Any) -> None:
        handler = workflow.iq_blobs_handler
        if handler is None:
            return

        figures = self._handler_figures(handler)
        if figures:
            self.iq_blob_figures[float(amplitude)] = figures

    def _record_kernel_figures(self, amplitude: float, workflow: Any) -> None:
        figures = self._workflow_figures(workflow, "kernel")
        if figures:
            self.kernel_figures[float(amplitude)] = figures

    def _record_resonator_figures(self, amplitude: float, workflow: Any) -> None:
        figures = self._workflow_figures(workflow, "resonator")
        if figures:
            self.resonator_figures[float(amplitude)] = figures

    def _workflow_handlers(self, workflow: Any, experiment: str) -> list[Any]:
        plural_name = f"{experiment}_handlers"
        handlers = list(getattr(workflow, plural_name, []) or [])
        if handlers:
            return handlers

        handler = getattr(workflow, f"{experiment}_handler", None)
        return [handler] if handler is not None else []

    def _workflow_figures(self, workflow: Any, experiment: str) -> list[Figure]:
        figures = []
        for handler in self._workflow_handlers(workflow, experiment):
            figures.extend(self._handler_figures(handler))
        return self._unique_figures(figures)

    def _handler_figures(self, handler: Any) -> list[Figure]:
        figures = []
        for attribute_name in ("workflow_figures", "figs", "figures"):
            figures.extend(self._extract_figures(getattr(handler, attribute_name, None)))

        figure = getattr(handler, "fig", None)
        figures.extend(self._extract_figures(figure))

        return self._unique_figures(figures)

    def _unique_figures(self, figures: list[Figure]) -> list[Figure]:
        unique_figures = []
        seen_ids = set()
        for figure in figures:
            figure_id = id(figure)
            if figure_id in seen_ids:
                continue
            unique_figures.append(figure)
            seen_ids.add(figure_id)

        return unique_figures

    def _extract_figures(self, value: Any) -> list[Figure]:
        if isinstance(value, Figure):
            return [value]
        if hasattr(value, "figure") and isinstance(value.figure, Figure):
            return [value.figure]
        if isinstance(value, dict):
            figures = []
            for item in value.values():
                figures.extend(self._extract_figures(item))
            return figures
        if isinstance(value, (list, tuple, set)):
            figures = []
            for item in value:
                figures.extend(self._extract_figures(item))
            return figures
        return []
