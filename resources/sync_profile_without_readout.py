from __future__ import annotations

from dataclasses import dataclass

from qratena.system.components_params.profile import Profile
from qratena.util.enums import SUPPORTED_PULSE_TYPES

from resources.load_profile import load_profile, push_profile

SOURCE_BRANCH = "main"
TARGET_BRANCH = "main_asaf"
DRY_RUN = True


@dataclass(frozen=True)
class SyncSummary:
    source_branch: str
    target_branch: str
    copied_qubits: int
    preserved_readout_pulses: int
    removed_readout_pulses: int


def copy_profile_without_readout_pulses(
    source: Profile,
    target: Profile,
    *,
    source_branch: str = SOURCE_BRANCH,
    target_branch: str = TARGET_BRANCH,
) -> tuple[Profile, SyncSummary]:
    """Copy source profile data while preserving target readout pulse params."""

    copied = source.model_copy(deep=True)
    copied.name = target.name

    preserved_readout_pulses = 0
    removed_readout_pulses = 0

    for qubit_name, copied_qubit in copied.qubits.items():
        source_has_readout = SUPPORTED_PULSE_TYPES.readout in copied_qubit.pulses
        target_qubit = target.qubits.get(qubit_name)
        target_readout_pulses = None

        if target_qubit is not None:
            target_readout_pulses = target_qubit.pulses.get(SUPPORTED_PULSE_TYPES.readout)

        if target_readout_pulses is not None:
            copied_qubit.pulses[SUPPORTED_PULSE_TYPES.readout] = target_readout_pulses
            preserved_readout_pulses += 1
        elif source_has_readout:
            copied_qubit.pulses.pop(SUPPORTED_PULSE_TYPES.readout, None)
            removed_readout_pulses += 1

    summary = SyncSummary(
        source_branch=source_branch,
        target_branch=target_branch,
        copied_qubits=len(copied.qubits),
        preserved_readout_pulses=preserved_readout_pulses,
        removed_readout_pulses=removed_readout_pulses,
    )
    return copied, summary


def sync_profile_without_readout_pulses(
    *,
    source_branch: str = SOURCE_BRANCH,
    target_branch: str = TARGET_BRANCH,
    dry_run: bool = DRY_RUN,
) -> SyncSummary:
    source = load_profile(source_branch)
    target = load_profile(target_branch)

    profile, summary = copy_profile_without_readout_pulses(
        source,
        target,
        source_branch=source_branch,
        target_branch=target_branch,
    )
    if not dry_run:
        push_profile(profile, target_branch)

    return summary


if __name__ == "__main__":
    result = sync_profile_without_readout_pulses()
    action = "would update" if DRY_RUN else "updated"
    print(
        f"{action} {result.target_branch} from {result.source_branch}: "
        f"{result.copied_qubits} qubits, "
        f"preserved {result.preserved_readout_pulses} readout pulse sets, "
        f"removed {result.removed_readout_pulses} source-only readout pulse sets"
    )
