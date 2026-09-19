#!/usr/bin/env python3
"""Fail-closed supervisor for the V12 Test-tuned Exact Log-MGF closeout.

The initial four-card replay controller is started separately.  This process
only waits for its PASS receipt, then runs the already-approved B0 comparator,
selects the Top-2 by the frozen Test-tuned rule, runs the planned margin
refinement, aggregates the final report, and publishes small receipts/results
to the target repository.  It never restarts an existing replay and never
changes the scientific protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


TOP4 = ("E05", "E10", "E07", "E06")
SELECTION_ORDER = (
    "Test Novel AssocA",
    "Test Overall AssocA",
    "Test TETA",
    "Test Base AssocA",
)


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


def _status(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = read_json(path)
    return value.get("status") if isinstance(value, dict) else None


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def wait_for_initial_controller(runtime: Path, pid: int | None, poll_seconds: float) -> dict[str, Any]:
    while True:
        if not runtime.is_file():
            raise RuntimeError(f"initial controller receipt missing: {runtime}")
        value = read_json(runtime)
        status = value.get("status")
        if status == "PASS":
            return value
        if status == "FAILED":
            raise RuntimeError(f"initial controller failed: {value.get('error')}")
        if status != "RUNNING":
            raise RuntimeError(f"unexpected initial controller status: {status!r}")
        if pid is not None:
            command = _proc_cmdline(pid)
            if not command or "v12_launch_mgf_exploration_replays.py" not in command:
                raise RuntimeError("initial controller receipt is RUNNING but controller PID is absent")
        time.sleep(max(5.0, float(poll_seconds)))


def _run_logged(command: list[str], *, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with {result.returncode}: {command}; see {log}")


def _require_initial_outputs(replay_root: Path) -> None:
    for card in TOP4:
        root = replay_root / card / "s00_m00"
        manifests = sorted(root.glob("shard_*/manifest.json"))
        metrics = root / "merged" / "test_tuned_metrics.json"
        if len(manifests) != 10 or not metrics.is_file():
            raise RuntimeError(f"initial Full-Test is incomplete for {card}: {len(manifests)}/10 manifests, metrics={metrics.is_file()}")
        for path in manifests:
            value = read_json(path)
            if value.get("status") != "PASS" or int(value.get("detector_forward_calls", -1)) != 0:
                raise RuntimeError(f"initial replay contract failed: {path}")
            if value.get("gt_loaded_during_replay") is not False:
                raise RuntimeError(f"GT replay guard failed: {path}")


def _metric(metrics: dict[str, Any], split: str, field: str) -> float:
    value = metrics.get(split, {}).get(field)
    if value is None:
        raise RuntimeError(f"missing {split}.{field}")
    return float(value)


def _select_top2(replay_root: Path) -> list[str]:
    rows: list[tuple[str, tuple[float, float, float, float, str]]] = []
    for card in TOP4:
        metrics = read_json(replay_root / card / "s00_m00" / "merged" / "test_tuned_metrics.json")
        rows.append(
            (
                card,
                (
                    _metric(metrics, "novel", "AssocA"),
                    _metric(metrics, "overall", "AssocA"),
                    _metric(metrics, "overall", "TETA"),
                    _metric(metrics, "base", "AssocA"),
                    card,
                ),
            )
        )
    rows.sort(key=lambda item: tuple([-item[1][i] for i in range(4)] + [item[1][4]]))
    return [item[0] for item in rows[:2]]


def _margin_points(diagnostic: Path, top2: list[str]) -> list[float]:
    value = read_json(diagnostic)
    if value.get("status") != "PASS" or value.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("margin diagnostic is not a valid Test-tuned receipt")
    distributions = value.get("distributions", {})
    result: list[float] = []
    for card in top2:
        item = distributions.get(card)
        if not isinstance(item, dict) or item.get("p25") is None or item.get("p50") is None:
            raise RuntimeError(f"missing card-specific margin distribution for {card}")
        result.extend([float(item["p25"]), float(item["p50"])])
    return result


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _file_receipt(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.is_file(), "sha256": sha256_file(path) if path.is_file() else None}


def _build_provenance(args: argparse.Namespace, controller: dict[str, Any], final_result: dict[str, Any]) -> dict[str, Any]:
    replay_root = args.replay_root.resolve()
    merged = sorted(replay_root.glob("*/*/merged/manifest.json"))
    failed_guards = []
    for path in merged:
        value = read_json(path)
        if value.get("status") != "PASS" or int(value.get("detector_forward_calls", -1)) != 0 or value.get("gt_loaded_during_replay") is not False:
            failed_guards.append(str(path))
    if failed_guards:
        raise RuntimeError(f"merged replay contract failed: {failed_guards}")
    capture = read_json(args.capture.resolve())
    if capture.get("capture_source") != "observed_live_proc_environ":
        raise RuntimeError("FAIL_CLOSED_CAPTURE_PROVENANCE")
    source_commit = _git(args.repo.resolve(), "rev-parse", "HEAD")
    return {
        "status": "PASS",
        "artifact": "v12_mgf_test_tuned_complete_provenance_receipt",
        "paper_status": "TEST_TUNED_EXPLORATION",
        "paper_valid": False,
        "diagnostic_only": True,
        "test_used_for_selection": True,
        "selection_order": list(SELECTION_ORDER),
        "source_commit": source_commit,
        "ranking": _file_receipt(args.ranking),
        "margin_diagnostic": _file_receipt(args.diagnostic),
        "capture": {**_file_receipt(args.capture), "capture_source": capture.get("capture_source")},
        "frontend_cache_manifest": _file_receipt(args.full_cache / "manifest.json"),
        "throughput_receipt": _file_receipt(args.throughput),
        "controller_runtime": _file_receipt(args.controller_runtime),
        "b0_runtime": _file_receipt(args.b0_runtime),
        "refinement_plan": _file_receipt(args.refinement_plan),
        "refinement_runtime": _file_receipt(args.refinement_runtime),
        "merged_manifest_count": len(merged),
        "merged_manifest_paths": [str(path.resolve()) for path in merged],
        "merged_replay_contract_failures": failed_guards,
        "final_result": {
            "path": str(args.final_json.resolve()),
            "sha256": sha256_file(args.final_json),
            "label": final_result.get("final_label"),
            "best_mgf": final_result.get("best_mgf", {}).get("candidate_id"),
            "b0_reference": final_result.get("b0_reference", {}).get("candidate_id"),
        },
        "created_at_unix": time.time(),
    }


def _write_budget(args: argparse.Namespace, controller: dict[str, Any], output: Path) -> None:
    started = float(controller.get("started_at_unix", time.time()))
    completed = time.time()
    write_json(
        output,
        {
            "status": "PASS",
            "artifact": "v12_mgf_test_tuned_budget_receipt",
            "paper_status": "TEST_TUNED_EXPLORATION",
            "controller_started_at_unix": started,
            "closeout_completed_at_unix": completed,
            "wall_seconds_controller_to_closeout": completed - started,
            "gpu_count": 10,
            "initial_replay_batch_size": 2,
            "throughput_policy": "retain double only after combined FPS >= 1.5x single FPS",
            "source": "observed controller and closeout wall clock",
        },
    )


def _publish(args: argparse.Namespace, files: list[tuple[Path, str]]) -> dict[str, Any]:
    repo = args.repo.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("refusing to publish with a dirty target worktree")
    destination = repo / "artifacts" / "v12_exact_log_mgf_test_tuned"
    destination.mkdir(parents=True, exist_ok=True)
    for source, relative in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    subprocess.run(["git", "-C", str(repo), "add", str(destination)], check=True)
    staged_empty = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet"],
        check=False,
    ).returncode == 0
    if staged_empty:
        raise RuntimeError("no result artifacts were staged")
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "Add V12 Exact Log-MGF Test-tuned closeout artifacts"], check=True)
    subprocess.run(["git", "-C", str(repo), "push", "origin", "codex/v12-exact-log-mgf"], check=True)
    head = _git(repo, "rev-parse", "HEAD")
    remote = _git(repo, "ls-remote", "origin", "refs/heads/codex/v12-exact-log-mgf").split()[0]
    if head != remote or _git(repo, "status", "--porcelain"):
        raise RuntimeError(f"GitHub closeout synchronization failed: local={head}, remote={remote}")
    return {"status": "PASS", "local_head": head, "remote_head": remote, "worktree_clean": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa_mgf"))
    parser.add_argument("--controller-runtime", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/controller_runtime.json"))
    parser.add_argument("--controller-pid", type=int, default=26725)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--replay-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/full_test"))
    parser.add_argument("--full-cache", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/00_provenance/test_cov_frontend"))
    parser.add_argument("--frontend-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/00_provenance/test_frontend/annotations"))
    parser.add_argument("--annotation", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/ovtr/tao_test_burst_v1.json"))
    parser.add_argument("--capture", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/v10_runtime_env_capture.json"))
    parser.add_argument("--cov-source", type=Path, default=Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean"))
    parser.add_argument("--cov-config", type=Path, default=Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"))
    parser.add_argument("--cov-checkpoint", type=Path, default=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/saved_models/ctao_public_res/ctao_public.pth"))
    parser.add_argument("--python-replay", type=Path, default=Path("/home/lwr/anaconda3/envs/ovtr/bin/python"))
    parser.add_argument("--python-post", type=Path, default=Path("/home/lwr/anaconda3/envs/masaenv/bin/python"))
    parser.add_argument("--b0-plan", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/plans/b0_initial/plan.json"))
    parser.add_argument("--b0-runtime", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/b0_initial_runtime.json"))
    parser.add_argument("--b0-checkpoint", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v11_dssl_official_20260917_full/03_train_rerun_c4f1bab/B0_OFFICIAL_V11/best.pt"))
    parser.add_argument("--card-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/03_train/exploration_cards"))
    parser.add_argument("--diagnostic", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/plans/test_margin_diagnostic.json"))
    parser.add_argument("--refinement-plan-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/plans/refinement"))
    parser.add_argument("--refinement-plan", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/plans/refinement/plan.json"))
    parser.add_argument("--refinement-runtime", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/refinement_runtime.json"))
    parser.add_argument("--ranking", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/04_ranking/ranking_leaderboard_test_tuned.json"))
    parser.add_argument("--throughput", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/05_replay/throughput_probe/20260919/throughput_probe_receipt.json"))
    parser.add_argument("--final-root", type=Path, default=Path("/data2/usr_for_deadline/tempotrack_v12_mgf_explore/06_final"))
    args = parser.parse_args()

    controller = wait_for_initial_controller(args.controller_runtime.resolve(), args.controller_pid, args.poll_seconds)
    _require_initial_outputs(args.replay_root.resolve())

    b0_status = _status(args.b0_runtime.resolve())
    if b0_status is None:
        command = [
            str(args.python_post.resolve()), str(args.repo.resolve() / "tools" / "v12_launch_mgf_refinement_replays.py"),
            "--plan-json", str(args.b0_plan.resolve()), "--frontend-root", str(args.frontend_root.resolve()),
            "--full-cache", str(args.full_cache.resolve()), "--replay-root", str(args.replay_root.resolve()),
            "--annotation", str(args.annotation.resolve()), "--capture", str(args.capture.resolve()),
            "--repo", str(args.repo.resolve()), "--python-replay", str(args.python_replay.resolve()),
            "--python-post", str(args.python_post.resolve()), "--cov-source", str(args.cov_source.resolve()),
            "--cov-config", str(args.cov_config.resolve()), "--cov-checkpoint", str(args.cov_checkpoint.resolve()),
            "--batch-size", "1", "--shard-count", "10", "--gpus", "0,1,2,3,4,5,6,7,8,9",
            "--poll-seconds", "60", "--evaluation-cores", "8", "--runtime", str(args.b0_runtime.resolve()),
        ]
        _run_logged(command, cwd=args.repo.resolve(), log=args.replay_root.parent / "b0_initial_controller.log")
    elif b0_status != "PASS":
        raise RuntimeError(f"refusing to attach or duplicate nonterminal B0 controller: {args.b0_runtime}")

    top2 = _select_top2(args.replay_root.resolve())
    margins = _margin_points(args.diagnostic.resolve(), top2)
    if not args.refinement_plan.resolve().is_file():
        command = [
            str(args.python_post.resolve()), str(args.repo.resolve() / "tools" / "v12_prepare_test_tuned_plan.py"),
            "--mode", "refinement", "--output-root", str(args.refinement_plan_root.resolve()),
            "--plan-path", str(args.refinement_plan.resolve()), "--b0-checkpoint", str(args.b0_checkpoint.resolve()),
            "--initial-replay-root", str(args.replay_root.resolve()), "--initial-cards", *TOP4,
            "--card-root", str(args.card_root.resolve()), "--margins", *(str(value) for value in margins),
        ]
        _run_logged(command, cwd=args.repo.resolve(), log=args.refinement_plan_root / "prepare.log")
    plan = read_json(args.refinement_plan.resolve())
    if plan.get("status") != "PASS" or plan.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("refinement plan failed Test-tuned contract")

    refinement_status = _status(args.refinement_runtime.resolve())
    if refinement_status is None:
        command = [
            str(args.python_post.resolve()), str(args.repo.resolve() / "tools" / "v12_launch_mgf_refinement_replays.py"),
            "--plan-json", str(args.refinement_plan.resolve()), "--frontend-root", str(args.frontend_root.resolve()),
            "--full-cache", str(args.full_cache.resolve()), "--replay-root", str(args.replay_root.resolve()),
            "--annotation", str(args.annotation.resolve()), "--capture", str(args.capture.resolve()),
            "--repo", str(args.repo.resolve()), "--python-replay", str(args.python_replay.resolve()),
            "--python-post", str(args.python_post.resolve()), "--cov-source", str(args.cov_source.resolve()),
            "--cov-config", str(args.cov_config.resolve()), "--cov-checkpoint", str(args.cov_checkpoint.resolve()),
            "--batch-size", "2", "--shard-count", "10", "--gpus", "0,1,2,3,4,5,6,7,8,9",
            "--poll-seconds", "60", "--evaluation-cores", "8", "--runtime", str(args.refinement_runtime.resolve()),
        ]
        _run_logged(command, cwd=args.repo.resolve(), log=args.replay_root.parent / "refinement_controller.log")
    elif refinement_status != "PASS":
        raise RuntimeError(f"refusing to attach or duplicate nonterminal refinement controller: {args.refinement_runtime}")

    args.final_root.resolve().mkdir(parents=True, exist_ok=True)
    final_json = args.final_root.resolve() / "final_20h_test_tuned_leaderboard.json"
    if not final_json.is_file():
        command = [
            str(args.python_post.resolve()), str(args.repo.resolve() / "tools" / "v12_finalize_test_tuned_leaderboard.py"),
            "--ranking-json", str(args.ranking.resolve()), "--replay-root", str(args.replay_root.resolve()),
            "--output-root", str(args.final_root.resolve()),
        ]
        _run_logged(command, cwd=args.repo.resolve(), log=args.final_root / "finalize.log")
    final_result = read_json(final_json)
    if final_result.get("status") != "PASS" or final_result.get("paper_status") != "TEST_TUNED_EXPLORATION":
        raise RuntimeError("final leaderboard failed Test-tuned contract")

    args.final_json = final_json
    budget = args.final_root.resolve() / "budget_receipt.json"
    _write_budget(args, controller, budget)
    provenance = args.final_root.resolve() / "provenance_receipt.json"
    write_json(provenance, _build_provenance(args, controller, final_result))
    files = [
        (args.ranking.resolve(), "ranking_leaderboard_test_tuned.json"),
        (args.ranking.resolve().with_suffix(".csv"), "ranking_leaderboard_test_tuned.csv"),
        (args.throughput.resolve(), "throughput_probe_receipt.json"),
        (args.controller_runtime.resolve(), "controller_runtime.json"),
        (args.b0_plan.resolve(), "plans/b0_initial_plan.json"),
        (args.b0_runtime.resolve(), "b0_initial_runtime.json"),
        (args.diagnostic.resolve(), "plans/test_margin_diagnostic.json"),
        (args.refinement_plan.resolve(), "plans/refinement_plan.json"),
        (args.refinement_runtime.resolve(), "refinement_runtime.json"),
        (args.final_root.resolve() / "final_20h_test_tuned_leaderboard.csv", "final_20h_test_tuned_leaderboard.csv"),
        (final_json, "final_20h_test_tuned_leaderboard.json"),
        (args.final_root.resolve() / "final_20h_test_tuned_report.md", "final_20h_test_tuned_report.md"),
        (budget, "budget_receipt.json"),
        (provenance, "provenance_receipt.json"),
    ]
    sync = _publish(args, files)
    write_json(args.final_root.resolve() / "github_sync_receipt.json", sync)
    print(json.dumps({"status": "PASS", "top2": top2, "margins": margins, "final_label": final_result.get("final_label"), "github": sync}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
