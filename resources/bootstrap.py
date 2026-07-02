from __future__ import annotations

try:
    from workbench_bootstrap import (
        DEPENDENCY_ROOTS,
        PROJECT_ROOT,
        QRATENA_DATA_ROOT,
        QRATENA_NINJA_PROFILE,
        WORKBENCH_ROOT,
        setup_workbench_environment,
    )
except ModuleNotFoundError:
    from workbench.workbench_bootstrap import (
        DEPENDENCY_ROOTS,
        PROJECT_ROOT,
        QRATENA_DATA_ROOT,
        QRATENA_NINJA_PROFILE,
        WORKBENCH_ROOT,
        setup_workbench_environment,
    )

__all__ = [
    "DEPENDENCY_ROOTS",
    "PROJECT_ROOT",
    "QRATENA_DATA_ROOT",
    "QRATENA_NINJA_PROFILE",
    "WORKBENCH_ROOT",
    "setup_workbench_environment",
]

