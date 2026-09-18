import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "v11_post_b0_calibrated_d_wave.py"
SPEC = importlib.util.spec_from_file_location("v11_post_b0_calibrated_d_wave", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_formal_pass_requires_causal_guards(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"status":"PASS","trial_id":"s00_m03","detector_forward_calls":0,"gt_loaded_during_replay":false}\n',
        encoding="utf-8",
    )
    assert MODULE.formal_pass(path, "s00_m03")
    path.write_text(
        '{"status":"PASS","trial_id":"s00_m03","detector_forward_calls":1,"gt_loaded_during_replay":false}\n',
        encoding="utf-8",
    )
    assert not MODULE.formal_pass(path, "s00_m03")


def test_merge_and_evaluation_commands_are_single_frozen_trial(tmp_path):
    args = type(
        "Args",
        (),
        {
            "python": Path("/usr/bin/python"),
            "merge_script": Path("merge.py"),
            "evaluator": Path("eval.py"),
            "annotation": tmp_path / "val.json",
            "shard_annotation_root": tmp_path / "annotations",
            "repo": tmp_path,
            "evaluation_cores": 8,
        },
    )()
    merge = MODULE.merge_command(args, tmp_path / "D1", tmp_path / "D1" / "aggregated", "s00_m03")
    evaluate = MODULE.evaluate_command(args, tmp_path / "D1" / "aggregated", "s00_m03")
    assert merge[merge.index("--trial-id") + 1] == "s00_m03"
    assert "--wait-for-complete" not in merge
    assert evaluate[evaluate.index("--trial-id") + 1] == "s00_m03"
    assert "--wait-for-aggregate" not in evaluate
