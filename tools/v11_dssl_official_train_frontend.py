"""Run the audited COVTrack frontend on the exact Official Train split.

The launcher only constructs experiment-owned annotation shards and runtime
receipts.  It consumes the existing live-proc audited environment capture and
fails closed if its provenance does not match the reference receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from tempotrack_v10.cov_category_ontology import build_category_mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ANNOTATION = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/"
    "annotations/train.json"
)
FRAMES_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames"
)
COV_SOURCE = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
COV_CONFIG = COV_SOURCE / "configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
COV_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/"
    "saved_models/ctao_public_res/ctao_public.pth"
)
CAPTURE = Path(
    "/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/"
    "v10_runtime_env_capture.json"
)
REFERENCE_RECEIPT = Path(
    "/data2/usr_for_deadline/tempotrack_v10_unified/search/"
    "covtrack_v104_best20h_20260914/full/s03_m01/trials/shard_00/receipt.json"
)
TEMPO_CONFIG = REPO_ROOT / "configs/research/v11/covtrack_dssl_official_frontend_disabled.yaml"
STREAM_PYTHON = Path("/home/lwr/anaconda3/envs/ovtr/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path = REPO_ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def git_branch(path: Path = REPO_ROOT) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_inputs() -> dict[str, Any]:
    required = (TRAIN_ANNOTATION, FRAMES_ROOT, COV_SOURCE, COV_CONFIG, COV_CHECKPOINT, CAPTURE, REFERENCE_RECEIPT, TEMPO_CONFIG, STREAM_PYTHON)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Official Train frontend inputs missing: " + ", ".join(missing))
    capture = read_json(CAPTURE)
    if capture.get("capture_source") != "observed_live_proc_environ":
        raise RuntimeError("FAIL_CLOSED_CAPTURE_PROVENANCE: capture was not observed live /proc")
    if str(capture.get("reference_receipt", "")) != str(REFERENCE_RECEIPT):
        raise RuntimeError("FAIL_CLOSED_CAPTURE_RECEIPT_MISMATCH")
    receipt = read_json(REFERENCE_RECEIPT)
    external = receipt.get("external_source", {})
    inputs = receipt.get("inputs", {})
    if str(external.get("path")) != str(COV_SOURCE):
        raise RuntimeError("FAIL_CLOSED_EXTERNAL_COV_SOURCE_MISMATCH")
    if str(inputs.get("external_config")) != str(COV_CONFIG):
        raise RuntimeError("FAIL_CLOSED_EXTERNAL_COV_CONFIG_MISMATCH")
    if str(inputs.get("external_checkpoint")) != str(COV_CHECKPOINT):
        raise RuntimeError("FAIL_CLOSED_EXTERNAL_COV_CHECKPOINT_MISMATCH")
    if inputs.get("external_config_sha256") != sha256(COV_CONFIG):
        raise RuntimeError("FAIL_CLOSED_EXTERNAL_COV_CONFIG_HASH_MISMATCH")
    if inputs.get("external_checkpoint_sha256") != sha256(COV_CHECKPOINT):
        raise RuntimeError("FAIL_CLOSED_EXTERNAL_COV_CHECKPOINT_HASH_MISMATCH")
    if external.get("commit") != git_head(COV_SOURCE):
        raise RuntimeError("FAIL_CLOSED_EXTERNAL_COV_COMMIT_MISMATCH")
    return {
        "capture": capture,
        "reference_receipt": receipt,
        "external_cov_commit": external["commit"],
        "external_config_sha256": inputs["external_config_sha256"],
        "external_checkpoint_sha256": inputs["external_checkpoint_sha256"],
    }


def load_train() -> dict[str, Any]:
    data = read_json(TRAIN_ANNOTATION)
    if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
        raise ValueError("Official Train annotation is not a COCO/TAO mapping")
    return data


def subset_annotation(data: Mapping[str, Any], video_ids: set[int], output: Path) -> dict[str, Any]:
    selected = copy.deepcopy(dict(data))
    selected["videos"] = [item for item in data.get("videos", []) if int(item["id"]) in video_ids]
    selected_images = []
    next_frame_by_video: dict[int, int] = {}
    for item in data.get("images", []):
        if int(item["video_id"]) not in video_ids:
            continue
        image = dict(item)
        # Official TAO Train uses frame_index while the audited COV TAO parser
        # requires its normalized frame_id field.  This is an experiment-owned
        # annotation adapter, not a GT/model-input substitution.
        video_id = int(image["video_id"])
        image["source_frame_index"] = int(image["frame_index"])
        image["frame_id"] = int(next_frame_by_video.get(video_id, 0))
        next_frame_by_video[video_id] = int(image["frame_id"]) + 1
        selected_images.append(image)
    selected["images"] = selected_images
    image_ids = {int(item["id"]) for item in selected_images}
    selected["annotations"] = [item for item in data.get("annotations", []) if int(item.get("image_id", -1)) in image_ids]
    selected["tracks"] = [item for item in data.get("tracks", []) if int(item.get("video_id", -1)) in video_ids]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, selected)
    return {
        "path": str(output),
        "sha256": sha256(output),
        "video_ids": sorted(video_ids),
        "video_count": len(selected["videos"]),
        "image_count": len(selected_images),
        "annotation_count": len(selected["annotations"]),
        "annotation_adapter": "added contiguous per-video image.frame_id in source image order; retained source_frame_index",
    }


def make_shards(data: Mapping[str, Any], root: Path, shard_count: int) -> list[dict[str, Any]]:
    video_ids = [int(item["id"]) for item in data.get("videos", [])]
    if shard_count < 1 or shard_count > len(video_ids):
        raise ValueError(f"invalid shard_count={shard_count} for {len(video_ids)} videos")
    result: list[dict[str, Any]] = []
    for index, ids in enumerate(
        video_ids[start::shard_count] for start in range(shard_count)
    ):
        shard_root = root / f"shard_{index:02d}"
        result.append(subset_annotation(data, set(ids), shard_root / "annotation.json"))
        result[-1]["index"] = index
        result[-1]["root"] = str(shard_root)
    return result


def provenance(
    *,
    audit: Mapping[str, Any],
    annotation: Mapping[str, Any],
    annotation_path: Path,
    shard_index: int,
) -> dict[str, Any]:
    capture = audit["capture"]
    category_ids, category_metadata = build_category_mapping(
        annotation=annotation,
        cov_source=COV_SOURCE,
    )
    return {
        "artifact": "v11_dssl_official_train_frontend_provenance",
        "input_source": "COVTRACK_FRONTEND",
        "supervision_source": "OFFICIAL_TRAIN_GT",
        "oracle_features_used": False,
        "gt_boxes_used_as_model_input": False,
        "gt_tracks_used_as_memory": False,
        "gt_used_only_for_supervision": True,
        "source_role": "OFFICIAL_TRAIN",
        "exact_split_name": "train",
        "source_annotation": str(TRAIN_ANNOTATION),
        "source_annotation_sha256": sha256(TRAIN_ANNOTATION),
        "shard_annotation": str(annotation_path),
        "shard_annotation_sha256": sha256(annotation_path),
        "shard_index": int(shard_index),
        "annotation_adapter": "added contiguous per-video image.frame_id in source image order; retained source_frame_index",
        "frontend": "COVTrack_native_official_stream",
        "cache_boundary": "before_pinned_covtrack_tracker_match",
        "repo_branch": git_branch(),
        "repo_head": git_head(),
        "external_cov_commit": audit["external_cov_commit"],
        "external_config_sha256": audit["external_config_sha256"],
        "external_checkpoint_sha256": audit["external_checkpoint_sha256"],
        "audited_runtime_capture": str(CAPTURE),
        "audited_runtime_capture_sha256": sha256(CAPTURE),
        "capture_source": capture.get("capture_source"),
        "capture_pid": capture.get("captured_pid"),
        "reference_receipt": str(REFERENCE_RECEIPT),
        "reference_receipt_sha256": sha256(REFERENCE_RECEIPT),
        # COV emits global detector labels in this exact order.  This mapping
        # is ontology metadata only; it does not contain GT boxes/tracks and
        # never changes the causal frontend arrays.
        "category_ids": category_ids,
        "runtime_contract": {
            "max_gap": 360,
            "candidate_top_k": 8,
            "memory_capacity": 64,
            "recent_k": 8,
            "context_candidate_top_k": 64,
            "query_observations": 1,
            "overlay_enabled": False,
            "train_gt_loaded_by_frontend": False,
        },
        "category_provenance": {
            "train_annotation_category_count": len(annotation.get("categories", [])),
            **category_metadata,
        },
    }


def command_for(
    *,
    audit: Mapping[str, Any],
    run_root: Path,
    shard: Mapping[str, Any],
    gpu: str,
) -> tuple[list[str], dict[str, str], Path]:
    shard_root = Path(str(shard["root"])).resolve()
    stream_root = shard_root / "stream"
    work_root = shard_root / "work"
    cache_root = shard_root / "frontend_cache"
    stream_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    provenance_path = shard_root / "provenance.json"
    write_json(provenance_path, provenance(
        audit=audit,
        annotation=read_json(Path(str(shard["path"]))),
        annotation_path=Path(str(shard["path"])),
        shard_index=int(shard["index"]),
    ))
    captured_env = dict(audit["capture"].get("environment", {}))
    env = dict(os.environ)
    if captured_env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = str(captured_env["LD_PRELOAD"])
    captured_pythonpath = str(captured_env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(REPO_ROOT), str(COV_SOURCE), captured_pythonpath) if value
    )
    env.update(
        {
            "V10_COV_SOURCE": str(COV_SOURCE),
            "V10_WORK_DIR": str(work_root),
            "V10_STREAM_RESULTS_DIR": str(stream_root),
            "V10_COV_TEMPO_CONFIG": str(TEMPO_CONFIG),
            "V10_COV_TEMPO_DIAGNOSTICS": str(shard_root / "diagnostics.json"),
            "V11_COV_REPLAY_CACHE_ROOT": str(cache_root),
            "V11_COV_REPLAY_PROVENANCE_JSON": str(provenance_path),
            "V11_COV_REPLAY_REPO_BRANCH": git_branch(),
            "V11_COV_REPLAY_REPO_HEAD": git_head(),
            "V11_COV_REPLAY_COV_COMMIT": str(audit["external_cov_commit"]),
            "V11_COV_REPLAY_COV_CONFIG_SHA256": str(audit["external_config_sha256"]),
            "V11_COV_REPLAY_COV_CHECKPOINT_SHA256": str(audit["external_checkpoint_sha256"]),
            "V10_TAO_FRAMES_ROOT": str(FRAMES_ROOT) + "/",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    command = [
        str(STREAM_PYTHON),
        str(REPO_ROOT / "tools/v10_covtrack_test_tempo_stream.py"),
        str(COV_CONFIG),
        str(COV_CHECKPOINT),
        "--out", str(shard_root / "native_results.pkl"),
        "--eval-options", f"resfile_path={shard_root / 'internal_results.pth'}",
        "--cfg-options",
        f"data.test.ann_file={shard['path']}",
        f"data.test.img_prefix={FRAMES_ROOT}/",
        "data.workers_per_gpu=1",
        "model.roi_head.only_validation_categories=False",
        "model.roi_head.only_test_categories=False",
        "model.tracker.match_score_thr=0.37",
        "model.tracker.memo_frames=50",
        "model.tracker.momentum_embed=0.4",
        "model.tracker.confused_features=True",
        "model.tracker.vis=False",
        "model.test_cfg.rcnn.max_per_img=80",
        "model.roi_head.feature_fusion_head.max_fusion_ratio=2.0",
    ]
    write_json(shard_root / "command.json", {"argv": command, "gpu": str(gpu), "environment": {key: env.get(key) for key in ("LD_PRELOAD", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "V10_COV_SOURCE", "V11_COV_REPLAY_PROVENANCE_JSON")}})
    return command, env, shard_root


def validate_shard(shard_root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = shard_root / "frontend_cache/manifest.json"
    stream_manifest_path = shard_root / "stream/stream_manifest.json"
    if not manifest_path.is_file() or not stream_manifest_path.is_file():
        raise RuntimeError(f"frontend shard did not produce complete artifacts: {shard_root}")
    manifest = read_json(manifest_path)
    stream = read_json(stream_manifest_path)
    if manifest.get("status") not in {"PASS", "COMPLETED"}:
        raise RuntimeError(f"frontend replay cache is not complete: {manifest_path}")
    if int(manifest.get("frame_count", -1)) != int(expected["image_count"]):
        raise RuntimeError(f"frontend frame count mismatch in {shard_root}")
    if int(manifest.get("video_count", -1)) != int(expected["video_count"]):
        raise RuntimeError(f"frontend video count mismatch in {shard_root}")
    if int(stream.get("frames", -1)) != int(expected["image_count"]):
        raise RuntimeError(f"stream frame count mismatch in {shard_root}")
    return {
        "root": str(shard_root),
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": sha256(manifest_path),
        "stream_manifest": str(stream_manifest_path),
        "frames": int(manifest["frame_count"]),
        "videos": int(manifest["video_count"]),
        "status": "PASS",
    }


def run_one(command: list[str], env: Mapping[str, str], root: Path) -> int:
    log_path = root / "frontend.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=str(COV_SOURCE),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def run_parallel(jobs: list[tuple[list[str], Mapping[str, str], Path]]) -> list[int]:
    """Launch disjoint complete-video shards concurrently, one per GPU."""

    processes: list[tuple[subprocess.Popen[Any], Any, Path, str | None]] = []
    try:
        for command, env, root in jobs:
            log_path = root / "frontend.log"
            log = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(COV_SOURCE),
                env=dict(env),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write_json(root / "process.json", {
                "pid": int(process.pid),
                "gpu": env.get("CUDA_VISIBLE_DEVICES"),
                "argv": command,
                "status": "RUNNING",
            })
            processes.append((process, log, root, env.get("CUDA_VISIBLE_DEVICES")))
    except Exception:
        for process, log, _root, _gpu in processes:
            if process.poll() is None:
                process.terminate()
            log.close()
        raise

    returncodes: list[int] = []
    for process, log, root, gpu in processes:
        returncode = int(process.wait())
        log.flush()
        log.close()
        returncodes.append(returncode)
        write_json(root / "process.json", {
            "pid": int(process.pid),
            "gpu": gpu,
            "returncode": returncode,
            "status": "PASS" if returncode == 0 else "FAILED",
        })
    return returncodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--gpus", default="1")
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    audit = validate_inputs()
    data = load_train()
    if args.mode == "smoke":
        first = int(data["videos"][0]["id"])
        shard_root = run_root / "00_frontend_smoke"
        shard = subset_annotation(data, {first}, shard_root / "annotation.json")
        shard.update({"index": 0, "root": str(shard_root)})
        shards = [shard]
        gpus = [str(args.gpus).split(",")[0]]
    else:
        gpus = [value.strip() for value in str(args.gpus).split(",") if value.strip()]
        shard_count = int(args.shards or len(gpus))
        if shard_count != len(gpus):
            raise ValueError("full mode requires one listed GPU per shard")
        shards = make_shards(data, run_root / "annotations", shard_count)
    write_json(run_root / f"{args.mode}_plan.json", {
        "mode": args.mode,
        "created_at_unix": time.time(),
        "train_annotation": str(TRAIN_ANNOTATION),
        "train_annotation_sha256": sha256(TRAIN_ANNOTATION),
        "audit_capture": str(CAPTURE),
        "audit_capture_sha256": sha256(CAPTURE),
        "external_cov_commit": audit["external_cov_commit"],
        "repo_head": git_head(),
        "shards": shards,
        "gpus": gpus,
        "structural_runtime": {"candidate_top_k": 8, "max_gap": 360, "memory_capacity": 64, "recent_k": 8, "context_candidate_top_k": 64},
    })
    if args.no_launch:
        return 0
    jobs = [
        command_for(audit=audit, run_root=run_root, shard=shard, gpu=gpu)
        for shard, gpu in zip(shards, gpus)
    ]
    if args.mode == "full":
        returncodes = run_parallel(jobs)
    else:
        returncodes = [run_one(*job) for job in jobs]
    results: list[dict[str, Any]] = []
    for (command, env, root), shard, gpu, returncode in zip(jobs, shards, gpus, returncodes):
        if returncode != 0:
            raise RuntimeError(f"Official Train frontend failed on GPU {gpu}, shard {root}; see {root / 'frontend.log'}")
        results.append(validate_shard(root, shard))
        write_json(run_root / f"{args.mode}_results.json", {"status": "PASS", "shards": results})
    print(json.dumps({"status": "PASS", "mode": args.mode, "shards": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
