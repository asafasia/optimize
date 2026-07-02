import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
TEST_DATA_ROOT = ROOT / "outputs" / "qratena_test"
DEPENDENCY_ROOTS = (
    PROJECT_ROOT / "qigeon",
    PROJECT_ROOT / "qratena",
    PROJECT_ROOT / "qhipu-lab",
    PROJECT_ROOT / "q-b2c",
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QRATENA_DATA_DIR", str(TEST_DATA_ROOT))

for dependency_root in DEPENDENCY_ROOTS:
    if str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))
