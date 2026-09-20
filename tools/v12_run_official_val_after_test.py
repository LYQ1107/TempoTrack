#!/usr/bin/env python3
"""Continue the V12 closeout after the existing Test-tuned supervisor ends.

This process is deliberately fail-closed.  It does not start any Val replay
until the existing Test closeout supervisor is gone, its final Test-tuned
leaderboard is complete, and the refinement receipt is PASS.  A nonterminal
Val runtime is never restarted automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _run_logged(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _require_pass(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} missing: {path}")
    value = read_json(path)
    if value.get("status") != "PASS":
        raise RuntimeError(f"{label} is not PASS: {value.get('status')}")
    if value.get("paper_status") != "TEST_TUNED_EXPLORATION" or value.get("paper_valid") is not False:
        raise RuntimeError(f"{label} paper-status guard failed")
    return value


def _wait_for_test_closeout(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    final_json = args.final_root.resolve() / "final_20h_test_tuned_leaderboard.json"
    refinement = args.refinement_runtime.resolve()
    while True:
        final_ready = final_json.is_file()
        refinement_ready = refinement.is_file()
        closeout_alive = bool(proc_cmdline(args.closeout_pid)) if args.closeout_pid else False
        if final_ready and refinement_ready and not closeout_alive:
            final = _require_pass(final_json, "Test-tuned final leaderboard")
            _require_pass(refinement, "Test refinement runtime")
            runtime.update(
                {
                    "status": "TEST_CLOSEOUT_PASS",
                    "test_final_json": str(final_json),
                    "test_final_json_sha256": sha256_file(final_json),
                    "test_refinement_runtime": str(refinement),
                    "test_refinement_runtime_sha256": sha256_file(refinement),
                    "test_closeout_observed_at_unix": time.time(),
                }
            )
            write_json(args.runtime, runtime)
            return final
        runtime.update(
            {
                "status": "WAITING_FOR_TEST_CLOSEOUT",
                "final_json_exists": final_ready,
                "refinement_runtime_exists": refinement_ready,
                "closeout_pid_alive": closeout_alive,
                "updated_at_unix": time.time(),
            }
        )
        write_json(args.runtime, runtime)
        time.sleep(max(10.0, float(args.poll_seconds)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_mgf"))
    parser.add_argument("--python-replay", type=Path, default=Path("/home/lwr/anaconda3/envs/ovtr/bin/python"))
    parser.add_argument("--python-post", type=Path, default=Path("/home/lwr/anaconda3/envs/masaenv/bin/python"))
    parser.add_argument("--closeout-pid", type=int, default=7924)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--ranking", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/04_ranking/ranking_leaderboard_test_tuned.json"))
    parser.add_argument("--test-replay-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/full_test"))
    parser.add_argument("--final-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/06_final"))
    parser.add_argument("--refinement-runtime", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/refinement_runtime.json"))
    parser.add_argument("--val-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/07_val_tuned"))
    parser.add_argument("--val-plan", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/07_val_tuned/official_val_plan.json"))
    parser.add_argument("--val-replay-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/07_val_tuned/full_val"))
    parser.add_argument("--val-runtime", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/07_val_tuned/official_val_runtime.json"))
    parser.add_argument("--val-cache", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v11_dssl_official_val_20260917_full/official_val_cov_frontend"))
    parser.add_argument("--val-annotation", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/validation_ours_v1.json"))
    parser.add_argument("--capture", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/v10_runtime_env_capture.json"))
    parser.add_argument("--cov-source", type=Path, default=Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean"))
    parser.add_argument("--cov-config", type=Path, default=Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"))
    parser.add_argument("--cov-checkpoint", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth"))
    parser.add_argument("--card-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/03_train/exploration_cards"))
    parser.add_argument("--b0-checkpoint", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v11_dssl_official_20260917_full/03_train_rerun_c4f1bab/B0_OFFICIAL_V11/best.pt"))
    parser.add_argument("--runtime", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/07_val_tuned/after_test_supervisor_runtime.json"))
    args = parser.parse_args()

    for path, label in ((args.ranking, "ranking"), (args.val_cache, "Official-Val cache"), (args.val_annotation, "Official-Val annotation"), (args.capture, "runtime capture")):
        if not path.exists():
            raise FileNotFoundError(f"{label} missing: {path}")
    args.runtime.parent.mkdir(parents=True, exist_ok=True)
    if args.runtime.exists():
        previous = read_json(args.runtime)
        if previous.get("status") == "PASS":
            print(json.dumps(previous, ensure_ascii=False, indent=2))
            return 0
        if previous.get("status") != "WAITING_FOR_TEST_CLOSEOUT":
            raise RuntimeError(f"refusing to restart existing after-Test supervisor: {args.runtime}")
        # A previous waiter may have been attached to an obsolete closeout PID.
        # Preserve its evidence and safely reattach only from this explicit
        # nonterminal waiting state; no replay is restarted here.
        runtime = dict(previous)
        runtime.update(
            {
                "attached_at_unix": time.time(),
                "attached_closeout_pid": int(args.closeout_pid) if args.closeout_pid else None,
                "attached_source_commit": subprocess.check_output(
                    ["git", "-C", str(args.repo.resolve()), "rev-parse", "HEAD"], text=True
                ).strip(),
            }
        )
    else:
        runtime = {
            "status": "STARTING",
            "artifact": "v12_mgf_test_tuned_official_val_after_test_supervisor",
            "paper_status": "TEST_TUNED_EXPLORATION",
            "paper_valid": False,
            "diagnostic_only": True,
            "test_used_for_selection": True,
            "val_used_for_selection": False,
            "repo": str(args.repo.resolve()),
            "source_commit": subprocess.check_output(["git", "-C", str(args.repo.resolve()), "rev-parse", "HEAD"], text=True).strip(),
            "started_at_unix": time.time(),
        }
    write_json(args.runtime, runtime)
    try:
        final = _wait_for_test_closeout(args, runtime)
        if args.val_plan.exists():
            plan = _require_pass(args.val_plan, "Official-Val plan")
        else:
            plan_command = [
                str(args.python_post.resolve()),
                str(args.repo.resolve() / "tools" / "v12_prepare_official_val_tuned_plan.py"),
                "--final-json", str(args.final_root.resolve() / "final_20h_test_tuned_leaderboard.json"),
                "--output-root", str(args.val_plan.parent.resolve() / "plan"),
                "--plan-path", str(args.val_plan.resolve()),
                "--card-root", str(args.card_root.resolve()),
                "--b0-checkpoint", str(args.b0_checkpoint.resolve()),
            ]
            _run_logged(plan_command, cwd=args.repo.resolve(), log=args.val_root / "prepare_official_val_plan.log")
            plan = _require_pass(args.val_plan, "Official-Val plan")
        runtime.update({"status": "VAL_PLAN_PASS", "val_plan": str(args.val_plan.resolve()), "val_plan_sha256": sha256_file(args.val_plan), "val_candidate_ids": plan.get("candidate_ids") or [item.get("candidate_id") for item in plan.get("candidates", [])]})
        write_json(args.runtime, runtime)

        if args.val_runtime.exists():
            val_runtime = _require_pass(args.val_runtime, "Official-Val runtime")
        else:
            launcher_command = [
                str(args.python_post.resolve()),
                str(args.repo.resolve() / "tools" / "v12_launch_official_val_tuned_replays.py"),
                "--plan-json", str(args.val_plan.resolve()),
                "--full-cache", str(args.val_cache.resolve()),
                "--annotation", str(args.val_annotation.resolve()),
                "--capture", str(args.capture.resolve()),
                "--repo", str(args.repo.resolve()),
                "--python-replay", str(args.python_replay.resolve()),
                "--python-post", str(args.python_post.resolve()),
                "--cov-source", str(args.cov_source.resolve()),
                "--cov-config", str(args.cov_config.resolve()),
                "--cov-checkpoint", str(args.cov_checkpoint.resolve()),
                "--replay-root", str(args.val_replay_root.resolve()),
                "--gpus", "0,1,2,3,4,5,6,7,8,9",
                "--shard-count", "10",
                "--batch-size", "2",
                "--poll-seconds", "60",
                "--evaluation-cores", "8",
                "--runtime", str(args.val_runtime.resolve()),
            ]
            _run_logged(launcher_command, cwd=args.repo.resolve(), log=args.val_root / "official_val_launcher.log")
            val_runtime = _require_pass(args.val_runtime, "Official-Val runtime")
        runtime.update({"status": "VAL_REPLAY_PASS", "val_runtime": str(args.val_runtime.resolve()), "val_runtime_sha256": sha256_file(args.val_runtime)})
        write_json(args.runtime, runtime)

        publish_command = [
            str(args.python_post.resolve()),
            str(args.repo.resolve() / "tools" / "v12_publish_official_val_tuned_closeout.py"),
            "--repo", str(args.repo.resolve()),
            "--ranking", str(args.ranking.resolve()),
            "--test-replay-root", str(args.test_replay_root.resolve()),
            "--final-root", str(args.final_root.resolve()),
            "--val-plan", str(args.val_plan.resolve()),
            "--val-runtime", str(args.val_runtime.resolve()),
            "--val-replay-root", str(args.val_replay_root.resolve()),
            "--python-post", str(args.python_post.resolve()),
        ]
        _run_logged(publish_command, cwd=args.repo.resolve(), log=args.val_root / "publish_official_val_closeout.log")
        runtime.update({"status": "PASS", "completed_at_unix": time.time(), "final_json": str((args.final_root / "final_20h_test_tuned_leaderboard.json").resolve()), "final_json_sha256": sha256_file(args.final_root / "final_20h_test_tuned_leaderboard.json"), "test_final_snapshot_sha256": final.get("ranking_sha256")})
        write_json(args.runtime, runtime)
        print(json.dumps(runtime, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        runtime.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}", "ended_at_unix": time.time()})
        write_json(args.runtime, runtime)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
