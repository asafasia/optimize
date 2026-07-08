from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
import sys


WORKBENCH_ROOT = Path(__file__).resolve().parents[1]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from workbench_bootstrap import setup_workbench_environment


setup_workbench_environment()

import networkx as nx
from laboneq.dsl.session import Session
from laboneq.simple import AcquisitionType, AveragingMode

from qratena.experiments.base_experiment import ExperimentSettings, ResetSettings
from qratena.experiments.iq_blobs import IQBlobsHandler
from qratena.experiments.kernel_traces_calculation import KernelTracesCalculationHandler
from qratena.experiments.sequence import (
    ExperimentNode,
    ExperimentSequence,
    NodeResult,
    NodeRunRecord,
    NodeStatus,
    OnFailure,
    SequencePolicy,
)
from qratena.system.components_params.profile import Profile
from qratena.system.qratena_platform import create_platform
from qratena.util.enums import (
    ExportationMethod,
    ResetType,
    SUPPORTED_PULSE_SHAPES,
    UpdateParamsMethod,
)
from qratena.util.qratena_logging import qratena_logger

from resources.load_profile import load_profile


PROFILE_NAME = "main_asaf"
QUBIT_NAMES = ["q3"]
STATES = ["g", "e"]
DO_EMULATION = True


@dataclass
class IQCalibrationKernelConfig:
    settings: ExperimentSettings = field(
        default_factory=lambda: ExperimentSettings(
            num_shots=48_000,
            acquisition_type=AcquisitionType.RAW,
            averaging_mode=AveragingMode.CYCLIC,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            update_params_method=UpdateParamsMethod.UPDATE,
            exportation_method=ExportationMethod.FULL,
            do_emulation=DO_EMULATION,
        )
    )


@dataclass
class IQCalibrationBlobsConfig:
    num_active_resets: int = 7
    settings: ExperimentSettings = field(
        default_factory=lambda: ExperimentSettings(
            num_shots=20_000,
            acquisition_type=AcquisitionType.INTEGRATION,
            averaging_mode=AveragingMode.SINGLE_SHOT,
            pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            update_params_method=UpdateParamsMethod.UPDATE,
            exportation_method=ExportationMethod.FULL,
            do_emulation=DO_EMULATION,
            reset=ResetSettings(
                reset_type=ResetType.ACTIVE,
                reset_num=7,
                reset_pulse_shape=SUPPORTED_PULSE_SHAPES.const,
            ),
        )
    )


class KernelTracesSequentialNode(ExperimentNode):
    """Runs kernel traces one qubit at a time before the IQ blob calibration."""

    def __init__(
        self,
        name: str,
        qubit_names: list[str],
        kernel_config: IQCalibrationKernelConfig,
        states: list[str] | None = None,
        on_failure: OnFailure | None = None,
        max_retries: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            handler_class=KernelTracesCalculationHandler,
            handler_kwargs={
                "settings": kernel_config.settings,
                "states": list(states or ["g", "e"]),
            },
            on_failure=on_failure,
            max_retries=max_retries,
        )
        self._qubit_names = list(qubit_names)

    def _execute(
        self,
        session: Session,
        profile: Profile,
        update_params_method_override: UpdateParamsMethod | None,
        sequence_data=None,
        qubit_names_override: list[str] | None = None,
    ) -> bool:
        if self._result is None:
            self._result = NodeResult(node_name=self._name, final_status=NodeStatus.PENDING)

        attempt_number = len(self._result.attempts) + 1
        started_at = time.time()
        qubit_names = list(qubit_names_override or self._qubit_names)

        try:
            self._run_sequentially(session, profile, update_params_method_override, qubit_names)
            if sequence_data is not None:
                sequence_data.add(
                    self._name,
                    {q: {"qubit_status": "succeeded"} for q in qubit_names},
                )
            self._result.attempts.append(
                NodeRunRecord(
                    attempt=attempt_number,
                    started_at=started_at,
                    ended_at=time.time(),
                    succeeded=True,
                    exception=None,
                )
            )
            return True
        except Exception as exc:
            if sequence_data is not None:
                sequence_data.add(
                    self._name,
                    {q: {"qubit_status": "failed"} for q in qubit_names},
                )
            self._result.attempts.append(
                NodeRunRecord(
                    attempt=attempt_number,
                    started_at=started_at,
                    ended_at=time.time(),
                    succeeded=False,
                    exception=exc,
                )
            )
            raise

    def _run_sequentially(
        self,
        session: Session,
        profile: Profile,
        update_params_method_override: UpdateParamsMethod | None,
        qubit_names: list[str],
    ) -> None:
        settings = self._handler_kwargs["settings"]
        if update_params_method_override is not None:
            settings = settings.model_copy(
                update={"update_params_method": update_params_method_override}
            )

        for qubit_name in qubit_names:
            qratena_logger.info("Running kernel traces for %s", qubit_name)
            handler = KernelTracesCalculationHandler(
                qubit_names=[qubit_name],
                settings=settings,
                session=session,
                profile=profile,
                states=self._handler_kwargs["states"],
            )
            handler.run()


class IQCalibrationSequence(ExperimentSequence):
    """IQ calibration sequence: kernel traces -> IQ blobs."""

    def __init__(
        self,
        qubit_names: list[str],
        kernel_config: IQCalibrationKernelConfig,
        iq_blobs_config: IQCalibrationBlobsConfig,
        policy: SequencePolicy,
        session: Session,
        profile: Profile,
        states: list[str] | None = None,
    ) -> None:
        dag = self._build_dag(
            qubit_names=qubit_names,
            kernel_config=kernel_config,
            iq_blobs_config=iq_blobs_config,
            states=list(states or ["g", "e"]),
        )
        super().__init__(dag=dag, policy=policy, session=session, profile=profile)

    @staticmethod
    def _build_dag(
        qubit_names: list[str],
        kernel_config: IQCalibrationKernelConfig,
        iq_blobs_config: IQCalibrationBlobsConfig,
        states: list[str],
    ) -> nx.DiGraph:
        kernel_node = KernelTracesSequentialNode(
            name="kernel",
            qubit_names=qubit_names,
            kernel_config=kernel_config,
            states=states,
        )
        iq_blobs_node = ExperimentNode(
            name="iq_blobs",
            handler_class=IQBlobsHandler,
            handler_kwargs={
                "qubit_names": qubit_names,
                "settings": iq_blobs_config.settings.model_copy(
                    update={
                        "reset": ResetSettings(
                            reset_type=ResetType.ACTIVE,
                            reset_num=iq_blobs_config.num_active_resets,
                            reset_pulse_shape=iq_blobs_config.settings.pulse_shape,
                        )
                    }
                ),
                "states": states,
            },
        )

        dag = nx.DiGraph()
        dag.add_edge(kernel_node, iq_blobs_node)
        return dag


def create_iq_calibration_sequence(
    qubit_names: list[str],
    profile: Profile,
    session: Session,
    states: list[str] | None = None,
    kernel_config: IQCalibrationKernelConfig | None = None,
    iq_blobs_config: IQCalibrationBlobsConfig | None = None,
) -> IQCalibrationSequence:
    return IQCalibrationSequence(
        qubit_names=qubit_names,
        kernel_config=kernel_config or IQCalibrationKernelConfig(),
        iq_blobs_config=iq_blobs_config or IQCalibrationBlobsConfig(),
        policy=SequencePolicy(
            max_retries=0,
            on_failure=OnFailure.ABORT_SEQUENCE,
            update_params_method=UpdateParamsMethod.UPDATE,
        ),
        session=session,
        profile=profile,
        states=states or STATES,
    )


def main() -> None:
    profile = load_profile(PROFILE_NAME)

    for qubit_name in QUBIT_NAMES:
        pulse = profile.qubits[qubit_name].pulses["readout"]["const"]
        print(
            f"Readout pulse for {qubit_name}: "
            f"{pulse.readout_amplitude}, {pulse.readout_duration}"
        )

    platform = create_platform(profile)
    session = Session(platform.setup)
    session.connect(do_emulation=DO_EMULATION)

    sequence = create_iq_calibration_sequence(
        qubit_names=QUBIT_NAMES,
        profile=profile,
        session=session,
        states=STATES,
    )
    result = sequence.run()

    print("completed:", result.completed)
    print("failed nodes:", result.failed_nodes)
    print("sequence data:", result.sequence_data.all_data)


if __name__ == "__main__":
    main()
