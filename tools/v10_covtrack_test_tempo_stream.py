"""Run the pinned COVTrack frontend with the shared V10 overlay.

The COV checkout is not edited.  ``covtrack_runtime`` inserts the adapter at
the verified post-MCF/pre-ID boundary in memory, while the generic stream
transport writes only the final official tracking rows per complete video.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import Any

from tempotrack_v10.covtrack_runtime import install_covtrack_runtime
from tempotrack_v10.overlay import TempoTrackConfig
from v10_ovtrack_test_tempo_stream import streaming_single_gpu_test


def _config() -> TempoTrackConfig:
    config_path = Path(
        os.environ.get(
            "V10_COV_TEMPO_CONFIG",
            "/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v10_unified/"
            "configs/research/v10/covtrack_full_tempo.yaml",
        )
    ).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise RuntimeError("COV V10 config requires PyYAML") from exc
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tempo = raw.get("tempo", raw) if isinstance(raw, dict) else None
    if not isinstance(tempo, dict):
        raise ValueError(f"COV V10 tempo config must be a mapping: {config_path}")
    fields = set(TempoTrackConfig.__dataclass_fields__)
    values = {key: value for key, value in tempo.items() if key in fields}
    checkpoint = values.get("reranker_checkpoint")
    values["reranker_checkpoint"] = (
        str(Path(checkpoint).resolve()) if checkpoint else None
    )
    return TempoTrackConfig(**values)


def main() -> None:
    source_value = os.environ.get("V10_COV_SOURCE")
    work_value = os.environ.get("V10_WORK_DIR")
    stream_value = os.environ.get("V10_STREAM_RESULTS_DIR")
    if not source_value or not work_value or not stream_value:
        raise RuntimeError(
            "V10_COV_SOURCE, V10_WORK_DIR and V10_STREAM_RESULTS_DIR are required"
        )
    source = Path(source_value).resolve()
    work_dir = Path(work_value).resolve()
    stream_dir = Path(stream_value).resolve()
    if not source.is_dir() or not work_dir.is_absolute() or not stream_dir.is_absolute():
        raise RuntimeError("COV V10 paths must be audited absolute paths")
    test_script = source / "tools" / "test.py"
    if not test_script.is_file():
        raise FileNotFoundError(test_script)
    work_dir.mkdir(parents=True, exist_ok=True)
    stream_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    if not hasattr(np, "int"):
        np.int = int

    install_covtrack_runtime(_config())

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

    spec = importlib.util.spec_from_file_location("v10_pinned_covtrack_test_stream", test_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned COVTrack test entry point: {test_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_parse_args = module.parse_args

    def parse_args_with_work_dir() -> Any:
        args = original_parse_args()
        args.work_dir = str(work_dir)
        return args

    module.parse_args = parse_args_with_work_dir
    module.main()


if __name__ == "__main__":
    main()
