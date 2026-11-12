"""Entry point for `python -I /abs/path/scripts/raimon_cli.py ...`.

`python -I` ignores PYTHONPATH and does not add the script's directory to sys.path, so the pinned code's `src`
directory is added explicitly from this file's location. The code that runs is always the checkout this script
belongs to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raimon.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
