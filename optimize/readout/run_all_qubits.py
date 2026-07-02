

from matplotlib import pyplot as plt
import numpy as np
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType

from workbench.optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from resources import load_profile
from workbench.optimize.readout.readout_workflow import ReadoutFidelityWorkflow
from workbench.resources import load_task_manager


profile = load_profile('main_asaf')
task_manager = load_task_manager()

settings = ReadoutFidelityWorkflowSettings(
    profile_name="main",
    do_emulation=False,
    run_resonator=True,  # already ran once to update the profile with resonator frequencies
    run_kernels=True,
    run_iq_blobs=True,
    show_handler_output=True,
    reset=ResetSettings(
        ResetType.ACTIVE,
        reset_num=5,
    ),
    do_plotting=True,
    states=['g', 'e'],
)

qubit_names = [q for q in sorted(
    profile.qubits.keys(), key=lambda x: int(x[1:])) if q != "q2"]



# qubit_names = qubit_names[:1]

fidelities_with_active_reset = {}
fidelities_with_passive_reset = {}

for qubit_name in qubit_names:
    try:
        workflow = ReadoutFidelityWorkflow(
            qubit_names=[qubit_name],
            profile=profile,
            task_manager=task_manager,
            settings=settings,
        )

        workflow.run()

        fidelity = workflow.results['iq_blobs'][qubit_name]['readout_fidelity']
        fidelities_with_active_reset[qubit_name] = fidelity
        print(f"Fidelity for {qubit_name} with active reset: {fidelity}")
    except Exception as e:
        print(f"Workflow failed with error: {e}")
        fidelities_with_active_reset[qubit_name] = 0.0  # Assign a default value in case of failure


for qubit_name in qubit_names:
    try:

        settings.reset = ResetSettings(
            ResetType.PASSIVE
        )
        workflow = ReadoutFidelityWorkflow(
            qubit_names=[qubit_name],
            profile=profile,
            task_manager=task_manager,
            settings=settings
        )

        workflow.run()

        fidelity = workflow.results['iq_blobs'][qubit_name]['readout_fidelity']
        fidelities_with_passive_reset[qubit_name] = fidelity
        print(f"Fidelity for {qubit_name} with passive reset: {fidelity}")
    except Exception as e:
        print(f"Workflow failed with error: {e}")
        fidelities_with_passive_reset[qubit_name] = 0.0  # Assign a default value in case of failure


# %%
import numpy as np
plot_qubits = [
    q for q in qubit_names if q in fidelities_with_active_reset or q in fidelities_with_passive_reset]
active_vals = [fidelities_with_active_reset.get(q, 0.0) for q in plot_qubits]
passive_vals = [fidelities_with_passive_reset.get(q, 0.0) for q in plot_qubits]

avg_active_fidelity = sum(active_vals) / len(active_vals) if active_vals else 0.0
avg_passive_fidelity = sum(passive_vals) / len(passive_vals) if passive_vals else 0.0

print(f"Average fidelity with active reset: {avg_active_fidelity:.4f}")
print(f"Average fidelity with passive reset: {avg_passive_fidelity:.4f}")

np.savez('readout_fidelity_comparison.npz', qubits=plot_qubits, active_fidelities=active_vals, passive_fidelities=passive_vals,
         avg_active_fidelity=avg_active_fidelity, avg_passive_fidelity=avg_passive_fidelity)

# print(active_vals)
# print(passive_vals)

colors = ['C4', 'C2']
# if plot_qubits:
x = list(range(len(plot_qubits)))
width = 0.38
plt.figure(figsize=(12, 6))
plt.bar([i + width / 2 for i in x], passive_vals,
        width=width, label='Passive reset', color=colors[1])
plt.bar([i - width / 2 for i in x], active_vals,
        width=width, label='Active reset', color=colors[0])


plt.axhline(avg_active_fidelity, color=colors[0], linestyle='--', linewidth=1.5,
        label=f'Avg Active ({avg_active_fidelity:.4f})')
plt.axhline(avg_passive_fidelity, color=colors[1], linestyle='--', linewidth=1.5,
        label=f'Avg Passive ({avg_passive_fidelity:.4f})')

plt.xticks(x, plot_qubits, rotation=45)
plt.ylabel('Readout Fidelity')
plt.xlabel('Qubit')
plt.title('Readout Fidelity per Qubit: Active vs Passive Reset')
plt.text(
    0.01,
    0.99,
    f"Avg Active: {avg_active_fidelity:.4f}\nAvg Passive: {avg_passive_fidelity:.4f}",
    transform=plt.gca().transAxes,
    va='top',
    ha='left',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
)
plt.ylim(0.5, 1)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
# plt.show()

plt.savefig('readout_fidelity_comparison.png', dpi=300)




# %%
