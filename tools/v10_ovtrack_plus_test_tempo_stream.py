"""Run the pinned OVT-B test entry point with the shared V10 overlay.

This is the OVT+ counterpart of ``v10_ovtrack_test_tempo_stream.py``.  The
only additional setup is the already-audited pin-local import/motion shim;
detector, embeddings, native affinity, and official result transport remain
the upstream execution path.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

from v10_ovtrack_test_tempo_stream import _required_path, streaming_single_gpu_test


def main() -> None:
    source = _required_path("V10_OVTRACK_SOURCE")
    test_script = source / "tools" / "test.py"
    work_dir = _required_path("V10_WORK_DIR")
    if not test_script.is_file():
        raise FileNotFoundError(f"pinned OVT-B tools/test.py missing: {test_script}")

    import numpy as np

    if not hasattr(np, "int"):
        np.int = int

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
            self.init_motion()

        OVTrack.__init__ = init_with_motion
        OVTrack._v10_motion_compat = True

    import ovtrack.apis as apis
    import ovtrack.apis.test as api_test

    apis.single_gpu_test = streaming_single_gpu_test
    api_test.single_gpu_test = streaming_single_gpu_test
    try:
        import mmdet.apis as mmdet_apis
        import mmdet.apis.test as mmdet_api_test
    except Exception:
        mmdet_apis = None
        mmdet_api_test = None
    if mmdet_apis is not None:
        mmdet_apis.single_gpu_test = streaming_single_gpu_test
    if mmdet_api_test is not None:
        mmdet_api_test.single_gpu_test = streaming_single_gpu_test

    spec = importlib.util.spec_from_file_location("v10_pinned_ovtrack_plus_test_stream", test_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned OVT-B test entry point: {test_script}")
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
