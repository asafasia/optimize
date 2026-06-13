import os
from pathlib import Path
import subprocess
import sys


def test_readout_optimizer_forces_headless_matplotlib_backend():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["MPLBACKEND"] = "TkAgg"
    python_path = [str(root), str(root / "qigeon")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from optimize.readout.readout_amplitude_optimizer import "
                "ReadoutAmplitudeSweepSettings; "
                "import matplotlib; "
                "print(matplotlib.get_backend()); "
                "print(ReadoutAmplitudeSweepSettings(amplitudes=[0.1]).live_html_open_browser)"
            ),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["Agg", "False"]


def test_optimizer_runner_selects_agg_before_importing_pyplot():
    root = Path(__file__).resolve().parents[2]
    source = (root / "optimize_readout.py").read_text(encoding="utf-8")

    assert source.index('matplotlib.use("Agg", force=True)') < source.index(
        "from matplotlib import pyplot as plt"
    )
