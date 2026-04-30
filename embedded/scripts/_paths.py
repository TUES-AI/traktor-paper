"""Make scripts runnable from the `scripts/` directory or embedded root."""

import sys
from pathlib import Path

EMBEDDED_ROOT = Path(__file__).resolve().parents[1]
if str(EMBEDDED_ROOT) not in sys.path:
    sys.path.insert(0, str(EMBEDDED_ROOT))
