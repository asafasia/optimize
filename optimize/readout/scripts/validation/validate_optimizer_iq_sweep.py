from __future__ import annotations

import argparse
import sys
from pathlib import Path

WORKBENCH_ROOT = Path(__file__).resolve().parents[4]
if str(WORKBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKBENCH_ROOT))

from optimize.readout.validation import (
    ReadoutOptimizerValidation,
    ReadoutOptimizerValidationSettings,
)
from resources.load_profile import load_profile, load_task_manager


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile_name)
    task_manager = object() if args.do_emulation else load_task_manager()
    validator = ReadoutOptimizerValidation(
        optimizer_run_dir=args.optimizer_run_dir,
        profile=profile,
        task_manager=task_manager,
        settings=ReadoutOptimizerValidationSettings(
            profile_name=args.profile_name,
            active_reset_num=args.active_reset_num,
            task_status_poll_interval=args.task_status_poll_interval,
            do_emulation=args.do_emulation,
            show_handler_output=args.show_handler_output,
            states=args.states,
        ),
    )
    result = validator.run()
    print(f"Saved validation artifacts to {result['validation_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an optimizer run by applying its amplitudes to the main "
            "profile and running readout kernels plus IQ blobs."
        )
    )
    parser.add_argument("optimizer_run_dir", type=Path)
    parser.add_argument("--profile-name", default="main")
    parser.add_argument("--active-reset-num", type=int, default=5)
    parser.add_argument("--states", nargs="+", default=["g", "e"])
    parser.add_argument("--task-status-poll-interval", type=float, default=10.0)
    parser.add_argument("--do-emulation", action="store_true")
    parser.add_argument("--show-handler-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
