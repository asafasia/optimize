from __future__ import annotations

from enum import StrEnum


class ReadoutScanMethod(StrEnum):
    SWEEP = "sweep"
    ZOOM_IN = "zoom_in"
    GRADIENT = "gradient"
    GOLDEN_SECTION = "golden_section"
