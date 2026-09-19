#!/usr/bin/env python3
"""Finalize and publish the V12 Test-tuned closeout after Official-Val audit.

The Test-tuned supervisor may finish before the independent Official-Val
replay is available.  This fail-closed post-closeout step attaches the Val
metrics to the same final leaderboard, verifies both selected Val candidates,
and publishes only compact receipts/metrics to the target repository.  Raw
predictions remain in the external artifact store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _require_test_tuned(value: dict[str, Any], label: str) -> None:
    if value.get("status") not in {"PASS", "COMPLETED"}:
        raise RuntimeError(f"{label} is not complete: {value.get('status')}")
    if value.get("paper_status") != "TEST_TUNED_EXPLORATION" or value.get("paper_valid") is not False:
        raise RuntimeError(f"{label} paper-status guard failed")


def _verify_val(plan_path: Path, runtime_path: Path, val_root: Path) -> dict[str, Any]:
    plan = read_json(plan_path)
    runtime = read_json(runtime_path)
    _require_test_tuned(plan, "Official-Val plan")
    _require_test_tuned(runtime, "Official-Val runtime")
    if plan.get("split") != "Official-Val" or plan.get("exact_split_name") != "validation_ours_v1":
        raise RuntimeError("Official-Val split provenance is invalid")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Official-Val plan has no candidates")
    verified: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        trial_id = str(candidate.get("trial_id"))
        root = val_root / candidate_id / trial_id
        manifests = sorted(root.glob("shard_*/manifest.json"))
        if len(manifests) != 10:
            raise RuntimeError(f"Official-Val candidate is not 10/10: {candidate_id} ({len(manifests)})")
        for manifest_path in manifests:
            manifest = read_json(manifest_path)
            if manifest.get("status") != "PASS":
                raise RuntimeError(f"Official-Val shard failed: {manifest_path}")
            if int(manifest.get("detector_forward_calls", -1)) != 0:
                raise RuntimeError(f"Official-Val detector was called: {manifest_path}")
            if manifest.get("gt_loaded_during_replay") is not False:
                raise RuntimeError(f"Official-Val replay loaded GT: {manifest_path}")
            if manifest.get("paper_status") != "TEST_TUNED_EXPLORATION" or manifest.get("paper_valid") is not False:
                raise RuntimeError(f"Official-Val shard paper-status guard failed: {manifest_path}")
        metrics = root / "merged" / "val_metrics.json"
        merged_manifest = root / "merged" / "manifest.json"
        if not metrics.is_file() or not merged_manifest.is_file():
            raise RuntimeError(f"Official-Val merged artifacts missing: {candidate_id}")
        merged = read_json(merged_manifest)
        if merged.get("status") != "PASS" or int(merged.get("detector_forward_calls", -1)) != 0:
            raise RuntimeError(f"Official-Val merged contract failed: {merged_manifest}")
        verified.append(
            {
                "candidate_id": candidate_id,
                "trial_id": trial_id,
                "metrics": str(metrics.resolve()),
                "metrics_sha256": sha256_file(metrics),
                "merged_manifest": str(merged_manifest.resolve()),
                "merged_manifest_sha256": sha256_file(merged_manifest),
                "shard_count": len(manifests),
            }
        )
    return {
        "status": "PASS",
        "artifact": "v12_mgf_test_tuned_official_val_verification",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "split": "Official-Val",
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "runtime": str(runtime_path.resolve()),
        "runtime_sha256": sha256_file(runtime_path),
        "candidates": verified,
        "verified_at_unix": time.time(),
    }


def _run_finalizer(args: argparse.Namespace) -> None:
    command = [
        str(args.python_post.resolve()),
        str(args.repo.resolve() / "tools" / "v12_finalize_test_tuned_leaderboard.py"),
        "--ranking-json", str(args.ranking.resolve()),
        "--replay-root", str(args.test_replay_root.resolve()),
        "--output-root", str(args.final_root.resolve()),
        "--val-metrics-root", str(args.val_replay_root.resolve()),
    ]
    log = args.final_root.resolve() / "finalize_with_official_val.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=str(args.repo.resolve()), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"finalizer failed; see {log}")


def publish(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("target repository must be clean before Official-Val publish")
    verification = _verify_val(args.val_plan, args.val_runtime, args.val_replay_root)
    _run_finalizer(args)
    final_json = args.final_root.resolve() / "final_20h_test_tuned_leaderboard.json"
    final_csv = args.final_root.resolve() / "final_20h_test_tuned_leaderboard.csv"
    final_md = args.final_root.resolve() / "final_20h_test_tuned_report.md"
    final = read_json(final_json)
    _require_test_tuned(final, "final leaderboard")
    if final.get("val_metrics_root") != str(args.val_replay_root.resolve()):
        raise RuntimeError("final leaderboard is not bound to Official-Val metrics root")
    receipt = {
        "status": "PASS",
        "artifact": "v12_mgf_test_tuned_official_val_complete_closeout",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_selection": True,
        "val_used_for_selection": False,
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "ranking": {"path": str(args.ranking.resolve()), "sha256": sha256_file(args.ranking)},
        "official_val_verification": verification,
        "final_json": {"path": str(final_json.resolve()), "sha256": sha256_file(final_json)},
        "final_csv": {"path": str(final_csv.resolve()), "sha256": sha256_file(final_csv)},
        "final_report": {"path": str(final_md.resolve()), "sha256": sha256_file(final_md)},
        "created_at_unix": time.time(),
    }
    receipt_path = args.final_root.resolve() / "official_val_complete_closeout_receipt.json"
    write_json(receipt_path, receipt)

    destination = repo / "artifacts" / "v12_exact_log_mgf_test_tuned"
    destination.mkdir(parents=True, exist_ok=True)
    files: list[tuple[Path, str]] = [
        (args.ranking.resolve(), "ranking_leaderboard_test_tuned.json"),
        (args.ranking.resolve().with_suffix(".csv"), "ranking_leaderboard_test_tuned.csv"),
        (args.test_replay_root.resolve().parent / "throughput_probe" / "20260919" / "throughput_probe_receipt.json", "throughput_probe_receipt.json"),
        (args.final_root.resolve() / "final_20h_test_tuned_leaderboard.csv", "final_20h_test_tuned_leaderboard.csv"),
        (final_json, "final_20h_test_tuned_leaderboard.json"),
        (final_md, "final_20h_test_tuned_report.md"),
        (args.val_plan.resolve(), "plans/official_val_tuned_plan.json"),
        (args.val_runtime.resolve(), "official_val_tuned_runtime.json"),
        (receipt_path, "official_val_complete_closeout_receipt.json"),
    ]
    controller = args.test_replay_root.resolve().parent / "controller_runtime.json"
    if controller.is_file():
        files.append((controller, "controller_runtime.json"))
    for candidate in verification["candidates"]:
        candidate_id = candidate["candidate_id"]
        root = args.val_replay_root.resolve() / candidate_id / candidate["trial_id"] / "merged"
        files.extend(
            [
                (root / "val_metrics.json", f"official_val/{candidate_id}/val_metrics.json"),
                (root / "val_metrics.csv", f"official_val/{candidate_id}/val_metrics.csv"),
                (root / "manifest.json", f"official_val/{candidate_id}/merged_manifest.json"),
            ]
        )
    sync_receipt_path = args.final_root.resolve() / "github_sync_receipt_official_val.json"
    for source, relative in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run(["git", "-C", str(repo), "add", str(destination)], check=True)
    if subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet"], check=False).returncode == 0:
        raise RuntimeError("Official-Val publish has no staged changes")
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Attach Official-Val audit to V12 Test-tuned closeout"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "origin", "codex/v12-exact-log-mgf"], check=True)
    local = _git(repo, "rev-parse", "HEAD")
    remote = _git(repo, "ls-remote", "origin", "refs/heads/codex/v12-exact-log-mgf").split()[0]
    if local != remote or _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"GitHub synchronization failed: local={local}, remote={remote}")
    sync = {"status": "PASS", "local_head": local, "remote_head": remote, "worktree_clean": True}
    write_json(sync_receipt_path, sync)
    sync_target = destination / "github_sync_receipt_official_val.json"
    shutil.copy2(sync_receipt_path, sync_target)
    subprocess.run(["git", "-C", str(repo), "add", str(sync_target)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Record Official-Val GitHub synchronization"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "origin", "codex/v12-exact-log-mgf"], check=True)
    local = _git(repo, "rev-parse", "HEAD")
    remote = _git(repo, "ls-remote", "origin", "refs/heads/codex/v12-exact-log-mgf").split()[0]
    if local != remote or _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"final GitHub synchronization failed: local={local}, remote={remote}")
    return {"status": "PASS", "verification": verification, "github": sync}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_mgf"))
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--test-replay-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--val-plan", type=Path, required=True)
    parser.add_argument("--val-runtime", type=Path, required=True)
    parser.add_argument("--val-replay-root", type=Path, required=True)
    parser.add_argument("--python-post", type=Path, default=Path("/home/lwr/anaconda3/envs/masaenv/bin/python"))
    args = parser.parse_args()
    print(json.dumps(publish(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
