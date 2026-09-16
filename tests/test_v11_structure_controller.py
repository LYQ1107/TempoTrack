import importlib.util
from pathlib import Path
from types import SimpleNamespace


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "v11_search_structure_full_test.py"
_SPEC = importlib.util.spec_from_file_location("tempotrack_v11_structure_controller", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_worker_command_keeps_event_diagnostics_and_unique_port(tmp_path):
    args = SimpleNamespace(
        worker_python="python",
        repo=str(tmp_path),
        root=str(tmp_path),
        img_prefix=str(tmp_path / "images"),
        stream_python="stream-python",
        teta_source_root=str(tmp_path / "teta"),
    )
    preflight = {
        "cov": {"source": "/cov", "config": "/cov.py", "checkpoint": "/cov.pth"},
        "qdic": {"checkpoint": "/qdic.pt"},
        "runtime_environment": {
            "ld_preload": "/sqlite.so",
            "pythonpath": "/runtime/path",
            "scalabel_root": "/scalabel",
            "reference_receipt": "/reference.json",
            "reference_stream_script": "/reference_stream.py",
        },
    }
    trial = {
        "trial_id": "ST1",
        "score_threshold": 0.0,
        "margin_threshold": 0.3,
        "candidate_top_k": 16,
        "max_gap": 360,
        "collect_event_diagnostics": True,
        "master_port": 29501,
    }
    shard = {"path": "/annotation.json", "index": 0, "frame_count": 10, "video_count": 1}

    command = _MODULE.base._worker_command(
        args=args,
        preflight=preflight,
        trial=trial,
        shard=shard,
        shard_dir=tmp_path / "shard",
        attempt=1,
        gpu="1",
    )

    assert "--event-diagnostics" in command
    assert str(tmp_path / "shard" / "event_diagnostics.jsonl") in command
    assert command[command.index("--master-port") + 1] == "29501"
