from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from qratena.experiments.iq_blobs import IQBlobsHandler, IQBlobsSettings
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ExportationMethod, ResetType, SUPPORTED_PULSE_SHAPES

from workbench.optimize.readout.run_all_qubit_iq_blobs import (
    count_measured,
    extract_fidelity_rows,
    make_run_dir,
    plot_readout_fidelities,
    qubit_sort_key,
    save_fidelity_csv,
)
from workbench.resources.load_profile import load_profile, load_task_manager


PROFILE_NAME = "main"
OUTPUT_ROOT = Path("data/readout_iq_blobs_all_qubits_experiment")

NUM_SHOTS = 10_000
ACTIVE_RESET_NUM = 5
DO_EMULATION = False
STATES = ["g", "e"]


def main() -> None:
    args = parse_args()
    profile = load_profile()
    qubit_names = args.qubits or sorted(profile.qubits.keys(), key=qubit_sort_key)

    settings = IQBlobsSettings(
        num_shots=args.num_shots,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.SINGLE_SHOT,
        exportation_method=ExportationMethod.NONE,
        pulse_shape=SUPPORTED_PULSE_SHAPES.const,
        reset=ResetSettings(ResetType.ACTIVE, reset_num=args.active_reset_num),
        do_emulation=True,
        iq_plane_analysis="kde",
    )
    handler = IQBlobsHandler(
        qubit_names=qubit_names,
        settings=settings,
        profile=profile,
        states=args.states,
    )

    start = perf_counter()
    if args.do_emulation:
        handler.run()
    else:
        task_manager = load_task_manager()
        compiled_experiment = handler.get_compiled_experiment()
        task = task_manager.run_compiled_experiment(
            experiment_name=handler.experiment_name,
            profile_name=args.profile_name,
            qubit_names=handler.qubit_names,
            compiled_experiment=compiled_experiment,
            do_emulation=False,
        )
        task_result = task_manager.wait(task)
        handler.load_result(task_result)
        handler.analyze()
        handler.update_system_params()
        handler.export_data()

    elapsed = perf_counter() - start
    fidelity_rows = extract_fidelity_rows(qubit_names, handler.data)
    run_dir = make_run_dir(args.output_root, qubit_names)
    save_fidelity_csv(run_dir / "readout_fidelities.csv", fidelity_rows)

    figure = plot_readout_fidelities(fidelity_rows)
    figure.savefig(run_dir / "readout_fidelities.png", dpi=200, bbox_inches="tight")

    print(f"Measured {count_measured(fidelity_rows)} of {len(qubit_names)} qubits")
    print(f"IQ blobs experiment finished in {elapsed:.1f}s")
    print(f"Saved summary to {run_dir}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IQ blobs directly for all selected qubits, without the readout "
            "workflow, and plot their measured readout fidelities."
        )
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        help="Qubits to measure. Defaults to every qubit in the loaded profile.",
    )
    parser.add_argument("--profile-name", default=PROFILE_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory where the run folder, plot, and CSV are saved.",
    )
    parser.add_argument("--num-shots", type=int, default=NUM_SHOTS)
    parser.add_argument("--active-reset-num", type=int, default=ACTIVE_RESET_NUM)
    parser.add_argument("--states", nargs="+", default=STATES)
    parser.add_argument("--do-emulation", action="store_true", default=DO_EMULATION)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the summary plot after saving it.",
    )
    args, _unknown_args = parser.parse_known_args()
    return args


if __name__ == "__main__":
    main()
