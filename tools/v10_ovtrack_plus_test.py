"""Run the pinned OVT-B OVTrack+ test entry point with audited shims.

The upstream checkout is never edited.  This wrapper supplies only the
pin-local compatibility symbols already required by the completed Val native
run and materializes its configured Kalman motion object before inference.
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


def _require_final_checkpoint_argument() -> Path:
    value = os.environ.get("V10_OVTRACK_PLUS_FINAL_CHECKPOINT")
    if not value:
        raise RuntimeError(
            "V10_OVTRACK_PLUS_FINAL_CHECKPOINT is required for OVTrack+ inference"
        )
    checkpoint = Path(value).expanduser().resolve()
    if checkpoint.name == "ovtrack_clip_distillation.pth":
        raise RuntimeError(
            "OVTRACK_PLUS_INVALID_PRETRAIN_CHECKPOINT: "
            "ovtrack_clip_distillation.pth is not a final OVTrack+ checkpoint"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if len(sys.argv) >= 3:
        requested = Path(sys.argv[2]).expanduser().resolve()
        if requested != checkpoint:
            raise RuntimeError(
                "OVTRACK_PLUS_CHECKPOINT_MISMATCH: positional checkpoint must "
                "equal V10_OVTRACK_PLUS_FINAL_CHECKPOINT"
            )
        sys.argv[2] = str(checkpoint)
    return checkpoint


def main() -> None:
    _require_final_checkpoint_argument()
    source = _required_path("V10_OVTRACK_SOURCE")
    work_dir = _required_path("V10_WORK_DIR")
    test_script = source / "tools" / "test.py"
    if not test_script.is_file():
        raise FileNotFoundError(f"pinned tools/test.py missing: {test_script}")

    import numpy as np

    if not hasattr(np, "int"):
        np.int = int

    # Importing this helper installs the exact pin-local compatibility path
    # used by the Val native run and the official finalizer.
    tools_root = Path(__file__).resolve().parent
    if str(tools_root) not in sys.path:
        sys.path.insert(0, str(tools_root))
    from v10_ovtrack_finalize_native import install_ovtrack_plus_runtime_compat

    install_ovtrack_plus_runtime_compat()
    from ovtrack.models.mot.ovtrack import OVTrack

    if not getattr(OVTrack, "_v10_motion_compat", False):
        original_init = OVTrack.__init__

        def init_with_motion(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # The pin stores motion_cfg but accesses self.motion in
            # simple_test.  This is the same one-time materialization used by
            # the completed Val run; it does not alter detector or tracker
            # configuration.
            self.init_motion()

        OVTrack.__init__ = init_with_motion
        OVTrack._v10_motion_compat = True

    spec = importlib.util.spec_from_file_location("v10_pinned_ovtrack_plus_test", test_script)
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
