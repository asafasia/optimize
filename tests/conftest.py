import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
QIGEON_ROOT = ROOT / "qigeon"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(QIGEON_ROOT) not in sys.path:
    sys.path.insert(0, str(QIGEON_ROOT))
