"""Run the pinned OVTrack test entry point with the V10 pre-ID adapter.

The pinned checkout is loaded from ``V10_OVTRACK_SOURCE`` and is never edited
by this wrapper.  The only runtime additions are the behavior-preserving
NumPy compatibility alias required by the pinned 2021 code and the
``work_dir`` attribute expected by its ``main`` function.  All model,
dataloader, detector, embedding and native-affinity work remains in the
official ``tools/test.py`` path; the patched tracker is the only association
insertion point.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to an audited absolute path")
    path = Path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def main() -> None:
    source = _required_path("V10_OVTRACK_SOURCE")
    test_script = source / "tools" / "test.py"
    if not test_script.is_file():
        raise FileNotFoundError(f"pinned OVTrack tools/test.py missing: {test_script}")
    work_dir_value = os.environ.get("V10_WORK_DIR")
    if not work_dir_value:
        raise RuntimeError("V10_WORK_DIR must be an audited absolute output path")
    work_dir = Path(work_dir_value).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    if not hasattr(np, "int"):
        np.int = int

    spec = importlib.util.spec_from_file_location("v10_pinned_ovtrack_test", test_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned test entry point: {test_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_parse_args = module.parse_args

    def parse_args_with_work_dir():
        args = original_parse_args()
        args.work_dir = str(work_dir)
        return args

    module.parse_args = parse_args_with_work_dir
    module.main()


if __name__ == "__main__":
    main()
