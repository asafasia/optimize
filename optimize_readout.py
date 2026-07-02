from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt

from optimize.readout.readout_amplitude_optimizer import (
    ReadoutAmplitudeSweepSettings,
    ReadoutAmplitudeSweepWorkflow,
)
from optimize.readout.readout_workflow import ReadoutFidelityWorkflowSettings
from optimize.readout.utils.readout_scan_types import ReadoutScanMethod
from qratena.system.components_params.reset_settings import ResetSettings
from qratena.util.enums import ResetType
from resources.load_profile import load_profile, load_task_manager


def main() -> None:
    args = parse_args()

    profile = load_profile(args.profile_branch)
    if args.states == ["g", "e", "f"]:
        profile.ensure_pi_ef_pulse_for_all_qubits(overwrite=False)

    task_manager = load_task_manager()
    workflow_settings = ReadoutFidelityWorkflowSettings(
        profile_name=args.profile_name,
        do_emulation=args.do_emulation,
        run_resonator=args.run_resonator,
        run_kernels=not args.skip_kernels,
        run_iq_blobs=True,
        do_plotting=False,
        show_handler_output=args.show_handler_output,
        reset=ResetSettings(ResetType.ACTIVE, reset_num=args.active_reset_num),
        states=args.states,
    )
    optimizer_settings = ReadoutAmplitudeSweepSettings(
        amplitudes=args.amplitudes,
        workflow_settings=workflow_settings,
        method=ReadoutScanMethod(args.method),
        use_live_html_plotter=not args.no_live_html,
    )

    optimizer = ReadoutAmplitudeSweepWorkflow(
        qubit_names=args.qubits,
        profile=profile,
        task_manager=task_manager,
        settings=optimizer_settings,
    )
    optimizer.run()
    figure = optimizer.plot()
    run_dir = optimizer.save_results(output_dir=args.output_dir, figure=figure)
    plt.close(figure)
    print(f"Saved readout optimization results to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run readout amplitude optimization.")
    parser.add_argument("--qubits", nargs="+", required=True)
    parser.add_argument("--amplitudes", nargs="+", type=float, required=True)
    parser.add_argument("--profile-branch", default="main")
    parser.add_argument("--profile-name", default="main")
    parser.add_argument("--output-dir", default="data/readout_optimize")
    parser.add_argument(
        "--method",
        default=ReadoutScanMethod.SWEEP.value,
        choices=[method.value for method in ReadoutScanMethod],
    )
    parser.add_argument("--states", nargs="+", default=["g", "e"])
    parser.add_argument("--active-reset-num", type=int, default=5)
    parser.add_argument("--do-emulation", action="store_true")
    parser.add_argument("--run-resonator", action="store_true")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument("--show-handler-output", action="store_true")
    parser.add_argument("--no-live-html", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
