import networkx as nx

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode

from qratena.experiments.base_experiment import ExperimentSettings, ResetSettings
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.kernel_traces_calculation import KernelTracesCalculationHandler
from qratena.experiments.sequence import ExperimentNode, ExperimentSequence, SequencePolicy, OnFailure
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    SUPPORTED_PULSE_TYPES,
    UpdateParamsMethod,
)
from laboneq.dsl.session import Session

from qratena.system.components_params.profile import Profile
from qratena.system.qratena_platform import create_platform

from resources.load_profile import load_profile, load_task_manager


profile = load_profile('main_asaf')

task_manager = load_task_manager()


platform = create_platform(profile)

device_setup = platform.setup

session = Session(
    device_setup,
)
session.connect(do_emulation=True)


class ReadoutAmplitudeExperimentNode(ExperimentNode):
    def __init__(self, *, readout_amplitudes: dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self._readout_amplitudes = readout_amplitudes

    def _execute(
        self,
        session,
        profile,
        update_params_method_override,
        sequence_data=None,
        qubit_names_override=None,
    ) -> bool:
        for qubit_name, amplitude in self._readout_amplitudes.items():
            readout_pulse = profile.qubits[qubit_name].pulses[
                SUPPORTED_PULSE_TYPES.readout
            ][SUPPORTED_PULSE_SHAPES.const]
            readout_pulse.readout_amplitude = amplitude

        return super()._execute(
            session,
            profile,
            update_params_method_override,
            sequence_data,
            qubit_names_override,
        )


qubit_names = ["q3"]
readout_amplitudes = [0.02, 0.03, 0.04, 0.05]

kernel_settings = ExperimentSettings(
    num_shots=48_000,
    acquisition_type=AcquisitionType.RAW,
    averaging_mode=AveragingMode.CYCLIC,
    update_params_method=UpdateParamsMethod.UPDATE,
    exportation_method=ExportationMethod.FULL,
    do_emulation=False,
)

iq_settings = ExperimentSettings(
    num_shots=20_000,
    acquisition_type=AcquisitionType.INTEGRATION,
    averaging_mode=AveragingMode.SINGLE_SHOT,
    update_params_method=UpdateParamsMethod.UPDATE,
    exportation_method=ExportationMethod.FULL,
    do_emulation=False,
    reset=ResetSettings(reset_type=ResetType.ACTIVE, reset_num=7),
)

dag = nx.DiGraph()
previous = None

for amp in readout_amplitudes:
    kernel_node = ReadoutAmplitudeExperimentNode(
        name=f"kernel_traces_readout_amp_{amp:g}",
        handler_class=KernelTracesCalculationHandler,
        handler_kwargs={
            "qubit_names": qubit_names,
            "settings": kernel_settings,
            "states": ["g", "e"],
        },
        readout_amplitudes={q: amp for q in qubit_names},
    )
    iq_node = ReadoutAmplitudeExperimentNode(
        name=f"iq_blobs_readout_amp_{amp:g}",
        handler_class=IQBlobsHandler,
        handler_kwargs={
            "qubit_names": qubit_names,
            "settings": iq_settings,
            "states": ["g", "e"],
        },
        readout_amplitudes={q: amp for q in qubit_names},
    )

    if previous is not None:
        dag.add_edge(previous, kernel_node)
    dag.add_edge(kernel_node, iq_node)
    previous = iq_node

sequence = ExperimentSequence(
    dag=dag,
    policy=SequencePolicy(
        max_retries=0,
        on_failure=OnFailure.CONTINUE,
        update_params_method=UpdateParamsMethod.UPDATE,
    ),
    session=session,
    profile=profile,
)

result = sequence.run()


print("completed:", result.completed)
print("failed nodes:", result.failed_nodes)

data = result.sequence_data.all_data

for node_name, node_data in data.items():
    print(f"\n{node_name}")
    for qubit_name, qubit_data in (node_data or {}).items():
        print(qubit_name, qubit_data)
