from __future__ import annotations

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType

from optimize.readout.readout_workflow import (
    ReadoutFidelityWorkflow,
    ReadoutFidelityWorkflowSettings,
)
from resources import load_profile, load_task_manager


OUTPUT_ROOT = Path("outputs/readout_fidelity_comparison")


def main() -> None:
    profile = load_profile("main_asaf")
    task_manager = load_task_manager()

    settings = ReadoutFidelityWorkflowSettings(
        profile_name="main",
        do_emulation=False,
        run_resonator=True,
        run_kernels=True,
        run_iq_blobs=True,
        show_handler_output=True,
        reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
        do_plotting=True,
        states=["g", "e"],
    )

    qubit_names = [
        qubit_name
        for qubit_name in sorted(profile.qubits.keys(), key=lambda item: int(item[1:]))
        if qubit_name != "q2"
    ]

    active_fidelities = measure_fidelities(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
        reset=ResetSettings(ResetType.ACTIVE, reset_num=5),
        label="active reset",
    )
    passive_fidelities = measure_fidelities(
        qubit_names=qubit_names,
        profile=profile,
        task_manager=task_manager,
        settings=settings,
        reset=ResetSettings(ResetType.PASSIVE),
        label="passive reset",
    )

    save_comparison(qubit_names, active_fidelities, passive_fidelities)


def measure_fidelities(
    *,
    qubit_names: list[str],
    profile,
    task_manager,
    settings: ReadoutFidelityWorkflowSettings,
    reset: ResetSettings,
    label: str,
) -> dict[str, float]:
    fidelities: dict[str, float] = {}
    settings.reset = reset

    for qubit_name in qubit_names:
        try:
            workflow = ReadoutFidelityWorkflow(
                qubit_names=[qubit_name],
                profile=profile,
                task_manager=task_manager,
                settings=settings,
            )
            workflow.run()
            fidelity = workflow.results["iq_blobs"][qubit_name]["readout_fidelity"]
            fidelities[qubit_name] = fidelity
            print(f"Fidelity for {qubit_name} with {label}: {fidelity}")
        except Exception as error:
            print(f"Workflow failed for {qubit_name} with {label}: {error}")
            fidelities[qubit_name] = 0.0

    return fidelities


def save_comparison(
    qubit_names: list[str],
    active_fidelities: dict[str, float],
    passive_fidelities: dict[str, float],
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plot_qubits = [
        qubit_name
        for qubit_name in qubit_names
        if qubit_name in active_fidelities or qubit_name in passive_fidelities
    ]
    active_values = [active_fidelities.get(qubit_name, 0.0) for qubit_name in plot_qubits]
    passive_values = [passive_fidelities.get(qubit_name, 0.0) for qubit_name in plot_qubits]

    avg_active = sum(active_values) / len(active_values) if active_values else 0.0
    avg_passive = sum(passive_values) / len(passive_values) if passive_values else 0.0

    print(f"Average fidelity with active reset: {avg_active:.4f}")
    print(f"Average fidelity with passive reset: {avg_passive:.4f}")

    np.savez(
        OUTPUT_ROOT / "readout_fidelity_comparison.npz",
        qubits=plot_qubits,
        active_fidelities=active_values,
        passive_fidelities=passive_values,
        avg_active_fidelity=avg_active,
        avg_passive_fidelity=avg_passive,
    )

    colors = ["C4", "C2"]
    x_values = list(range(len(plot_qubits)))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(
        [index + width / 2 for index in x_values],
        passive_values,
        width=width,
        label="Passive reset",
        color=colors[1],
    )
    plt.bar(
        [index - width / 2 for index in x_values],
        active_values,
        width=width,
        label="Active reset",
        color=colors[0],
    )
    plt.axhline(
        avg_active,
        color=colors[0],
        linestyle="--",
        linewidth=1.5,
        label=f"Avg Active ({avg_active:.4f})",
    )
    plt.axhline(
        avg_passive,
        color=colors[1],
        linestyle="--",
        linewidth=1.5,
        label=f"Avg Passive ({avg_passive:.4f})",
    )
    plt.xticks(x_values, plot_qubits, rotation=45)
    plt.ylabel("Readout Fidelity")
    plt.xlabel("Qubit")
    plt.title("Readout Fidelity per Qubit: Active vs Passive Reset")
    plt.text(
        0.01,
        0.99,
        f"Avg Active: {avg_active:.4f}\nAvg Passive: {avg_passive:.4f}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.ylim(0.5, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "readout_fidelity_comparison.png", dpi=300)


if __name__ == "__main__":
    main()
