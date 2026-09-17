"""Audit the exact data/provenance contract for the V11 DSSL experiment.

This audit is intentionally read-only with respect to external data.  It
records the official Train annotation, the current V11 Val/Test annotation
contract, video disjointness, category metadata, and the feature/frontend
sources that are already available.  A missing Official-Train frontend is
reported explicitly; the audit never substitutes a Val/Test or pilot cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/"
    "annotations/train.json"
)
DEFAULT_VAL = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/"
    "ovtr/validation_ours_v1.json"
)
DEFAULT_TEST = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/data/external_annotations/"
    "ovtr/tao_test_burst_v1.json"
)
HISTORICAL_PILOT_FEATURES = Path(
    "/data2/usr_for_deadline/tempotrack_v11_qdic_pilot_20260913_224455/"
    "val_base/qdic_features"
)
HISTORICAL_VAL_EVENT_CACHE = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/masa_psmr_v9/outputs/"
    "tempotrack_v9/covtrack/val/event_cache"
)
VAL_FRONTEND_MANIFEST = Path(
    "/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/"
    "crossbaseline_v8_final/cov_val_native/manifest.json"
)
VAL_FRONTEND_PREDICTION = Path(
    "/data2/usr_for_deadline/tempotrack_v8_relocated_20260910/"
    "crossbaseline_v8_final/cov_val_native_frontend_aligned/prediction.json"
)
COVTRACK_CONFIG = Path(
    "/data2/usr_for_deadline/COVTrack_9b0ced_final_clean/configs/"
    "uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py"
)
COVTRACK_CHECKPOINT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/external_ovmot/COVTrack/"
    "saved_models/ctao_public_res/ctao_public.pth"
)
COVTRACK_SOURCE = Path("/data2/usr_for_deadline/COVTrack_9b0ced_final_clean")
AUDITED_RUNTIME_CAPTURE = Path(
    "/data2/usr_for_deadline/tempotrack_v11_qdic_fulltest_20h_20260914/"
    "v10_runtime_env_capture.json"
)
AUDITED_REFERENCE_RECEIPT = Path(
    "/data2/usr_for_deadline/tempotrack_v10_unified/search/"
    "covtrack_v104_best20h_20260914/full/s03_m01/trials/shard_00/receipt.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_if_file(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"annotation root must be an object: {path}")
    return value


def category_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    categories = [dict(item) for item in data.get("categories", [])]
    frequencies = Counter(str(item.get("frequency", "<missing>")) for item in categories)
    base_ids = sorted(
        int(item["id"]) for item in categories if str(item.get("frequency")) in {"f", "c"}
    )
    novel_ids = sorted(
        int(item["id"]) for item in categories if str(item.get("frequency")) == "r"
    )
    used_ids = sorted(
        {
            int(item["category_id"])
            for item in data.get("annotations", [])
            if "category_id" in item
        }
    )
    return {
        "category_count": len(categories),
        "frequency_counts": dict(sorted(frequencies.items())),
        "base_definition": "category.frequency in {'f','c'}",
        "novel_definition": "category.frequency == 'r'",
        "base_category_count": len(base_ids),
        "novel_category_count": len(novel_ids),
        "base_category_ids_sha256": hashlib.sha256(
            json.dumps(base_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "novel_category_ids_sha256": hashlib.sha256(
            json.dumps(novel_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "annotation_category_count": len(used_ids),
        "annotation_category_ids_sha256": hashlib.sha256(
            json.dumps(used_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def annotation_record(role: str, split_name: str, path: Path) -> dict[str, Any]:
    data = load_json(path)
    videos = list(data.get("videos", []))
    video_ids = {int(item["id"]) for item in videos}
    names = [str(item.get("name", item.get("file_name", ""))) for item in videos]
    prefixes = Counter(name.split("/", 1)[0] for name in names if "/" in name)
    info = data.get("info")
    return {
        "role": role,
        "exact_split_name": split_name,
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "schema_keys": sorted(str(key) for key in data),
        "info": info if isinstance(info, dict) else None,
        "video_count": len(videos),
        "video_ids": sorted(video_ids),
        "video_ids_sha256": hashlib.sha256(
            json.dumps(sorted(video_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "video_name_prefix_counts": dict(sorted(prefixes.items())),
        "image_count": len(data.get("images", [])),
        "annotation_count": len(data.get("annotations", [])),
        "track_count": len(data.get("tracks", [])),
        "category": category_summary(data),
    }


def source_record(kind: str, path: Path, *, hash_large: bool = False) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "kind": kind,
        "path": str(path),
        "exists": bool(exists),
        "is_file": bool(path.is_file()),
        "is_dir": bool(path.is_dir()),
    }
    if exists:
        record["size_bytes"] = path.stat().st_size if path.is_file() else None
        if path.is_file() and (hash_large or path.stat().st_size <= 128 * 1024 * 1024):
            record["sha256"] = sha256_file(path)
        elif path.is_file():
            record["sha256"] = None
            record["sha256_note"] = "omitted for large external asset"
    return record


def intersection_record(left: set[int], right: set[int]) -> dict[str, Any]:
    values = sorted(left & right)
    return {"count": len(values), "video_ids": values}


def build_audit(train: Path, val: Path, test: Path) -> dict[str, Any]:
    records = {
        "OFFICIAL_TRAIN": annotation_record("OFFICIAL_TRAIN", "train", train),
        "OFFICIAL_VAL": annotation_record(
            "OFFICIAL_VAL", "validation_ours_v1", val
        ),
        "CURRENT_TEST": annotation_record(
            "CURRENT_TEST", "tao_test_burst_v1", test
        ),
    }
    ids = {
        role: set(value["video_ids"])
        for role, value in records.items()
    }
    intersections = {
        "OFFICIAL_TRAIN__OFFICIAL_VAL": intersection_record(
            ids["OFFICIAL_TRAIN"], ids["OFFICIAL_VAL"]
        ),
        "OFFICIAL_TRAIN__CURRENT_TEST": intersection_record(
            ids["OFFICIAL_TRAIN"], ids["CURRENT_TEST"]
        ),
        "OFFICIAL_VAL__CURRENT_TEST": intersection_record(
            ids["OFFICIAL_VAL"], ids["CURRENT_TEST"]
        ),
    }
    disjoint = all(item["count"] == 0 for item in intersections.values())

    feature_sources = {
        "qdic_event_cache_builder": source_record(
            "qdic_event_cache_builder",
            REPO_ROOT / "tempotrack_research/orchestration/v9_parameter_search.py",
        ),
        "qdic_feature_builder": source_record(
            "qdic_feature_builder", REPO_ROOT / "tempotrack_v10/qdic_features.py"
        ),
        "partial_support_source": source_record(
            "partial_support_source",
            REPO_ROOT / "tempotrack_research/streaming/partial_support.py",
        ),
        "fixed_dual_memory_source": source_record(
            "fixed_dual_memory_source",
            REPO_ROOT / "tempotrack_research/memory/fixed_dual.py",
        ),
    }
    frontend_sources = {
        "official_train_covtrack_native": {
            "status": "MISSING_AUDITED_OUTPUT",
            "source": "COVTrack-native frontend output bound to OFFICIAL_TRAIN",
            "required_before_train_cache": True,
            "substitutions_forbidden": [
                str(HISTORICAL_VAL_EVENT_CACHE),
                str(HISTORICAL_PILOT_FEATURES),
                "OVTR/Val",
                "COVTrack/Val",
            ],
        },
        "official_val_covtrack_native": {
            "status": "AVAILABLE_HISTORICAL_VALIDATION_SOURCE",
            "manifest": source_record("Val native manifest", VAL_FRONTEND_MANIFEST),
            "prediction": source_record("Val frontend prediction", VAL_FRONTEND_PREDICTION),
            "event_cache": source_record("Val event cache", HISTORICAL_VAL_EVENT_CACHE),
            "optimizer_source_allowed": False,
        },
        "current_test": {
            "status": "HISTORICALLY_EXPOSED_CURRENT_V11_CONTRACT",
            "annotation_only": True,
            "optimizer_source_allowed": False,
            "selection_allowed": False,
            "final_label": "FROZEN_AFTER_VAL / HISTORICALLY_TEST_EXPOSED",
        },
        "covtrack_runtime_assets": {
            "source_root": source_record(
                "audited COVTrack source root", COVTRACK_SOURCE
            ),
            "config": source_record("COVTrack config", COVTRACK_CONFIG),
            "checkpoint": source_record(
                "COVTrack checkpoint", COVTRACK_CHECKPOINT, hash_large=False
            ),
            "checkpoint_hash_policy": "large checkpoint hash is deferred to the audited frontend receipt",
            "audited_runtime_capture": source_record(
                "live proc environ capture", AUDITED_RUNTIME_CAPTURE
            ),
            "audited_reference_receipt": source_record(
                "reference trial receipt", AUDITED_REFERENCE_RECEIPT
            ),
        },
    }
    capture_status = "MISSING"
    capture_details: dict[str, Any] = {
        "path": str(AUDITED_RUNTIME_CAPTURE),
        "reference_receipt": str(AUDITED_REFERENCE_RECEIPT),
    }
    if AUDITED_RUNTIME_CAPTURE.is_file():
        capture = load_json(AUDITED_RUNTIME_CAPTURE)
        capture_status = "PASS_OBSERVED_LIVE_PROC_ENVIRON"
        capture_details.update(
            {
                "capture_source": capture.get("capture_source"),
                "captured_pid": capture.get("captured_pid"),
                "capture_reference_receipt": capture.get("reference_receipt"),
                "capture_reference_stream_script": capture.get("reference_stream_script"),
                "capture_matches_expected_receipt": str(
                    capture.get("reference_receipt", "")
                ).rstrip("/")
                == str(AUDITED_REFERENCE_RECEIPT),
            }
        )
        if capture.get("capture_source") != "observed_live_proc_environ":
            capture_status = "FAIL_CLOSED_CAPTURE_PROVENANCE"
        if capture_details["capture_reference_receipt"] != str(AUDITED_REFERENCE_RECEIPT):
            capture_status = "FAIL_CLOSED_CAPTURE_RECEIPT_MISMATCH"
    else:
        capture_status = "FAIL_CLOSED_CAPTURE_MISSING"
    frontend_sources["covtrack_runtime_assets"]["audited_capture_contract"] = {
        "status": capture_status,
        **capture_details,
        "do_not_rewrite_or_handcraft": True,
    }
    category_note = {
        "status": "SPLIT_SPECIFIC_METADATA_RECORDED",
        "official_train_category_space": "raw TAO 1230-category annotation metadata",
        "v11_val_test_category_space": "current V11 1203-category annotation metadata",
        "warning": (
            "Train and current V11 Val/Test category IDs/frequency tables are not silently "
            "declared identical. Any later Train-to-V11 category mapping must be explicit "
            "and auditable; the split audit does not invent aliases."
        ),
    }
    pilot_guard = {
        "forbidden_optimizer_source": str(HISTORICAL_PILOT_FEATURES),
        "exists": HISTORICAL_PILOT_FEATURES.exists(),
        "allowed_role": "historical comparison only",
        "optimizer_source_allowed": False,
    }
    return {
        "schema_version": 1,
        "artifact": "v11_dssl_official_split_audit",
        "generated_by": "tools/v11_dssl_split_audit.py",
        "repository": str(REPO_ROOT),
        "repository_commit": subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "roles": records,
        "intersections": intersections,
        "split_disjoint": disjoint,
        "exact_split_allowlist": {
            "OFFICIAL_TRAIN": {"exact_split_name": "train", "source": "official_train"},
            "OFFICIAL_VAL": {
                "exact_split_name": "validation_ours_v1",
                "source": "current_v11_val_contract",
            },
            "CURRENT_TEST": {
                "exact_split_name": "tao_test_burst_v1",
                "source": "current_v11_test_contract",
            },
        },
        "category_protocol": category_note,
        "frontend_sources": frontend_sources,
        "feature_sources": feature_sources,
        "pilot_guard": pilot_guard,
        "optimizer_ready": bool(disjoint and frontend_sources["official_train_covtrack_native"]["status"] == "AVAILABLE"),
        "status": "PASS_SPLIT_DISJOINT_FRONTEND_TRAIN_PENDING" if disjoint else "FAIL_CLOSED_VIDEO_OVERLAP",
    }


def markdown(audit: Mapping[str, Any]) -> str:
    roles = audit["roles"]
    lines = [
        "# V11 DSSL Official-Train split audit",
        "",
        f"- Status: **{audit['status']}**",
        f"- Split disjointness: **{audit['split_disjoint']}**",
        f"- Optimizer source ready: **{audit['optimizer_ready']}**",
        f"- Repository commit: `{audit['repository_commit']}`",
        "",
        "## Exact annotation contract",
        "",
        "| role | exact split name | videos | images | annotations | tracks | categories | sha256 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for role in ("OFFICIAL_TRAIN", "OFFICIAL_VAL", "CURRENT_TEST"):
        item = roles[role]
        lines.append(
            f"| {role} | `{item['exact_split_name']}` | {item['video_count']} | "
            f"{item['image_count']} | {item['annotation_count']} | {item['track_count']} | "
            f"{item['category']['category_count']} | `{item['sha256']}` |"
        )
        lines.append(f"| path | `{item['path']}` | | | | | | |")
    lines += ["", "## Video intersections", "", "| pair | count |", "|---|---:|"]
    for name, value in audit["intersections"].items():
        lines.append(f"| `{name}` | {value['count']} |")
    lines += ["", "All three intersections must remain zero; otherwise the training run must fail closed.", ""]
    lines += [
        "## Category metadata",
        "",
        "Train uses raw TAO category metadata; current V11 Val/Test use the existing 1203-category V11 annotation contract.",
        "The audit records these spaces separately and does not invent a Train-to-Val/Test alias mapping.",
        "",
        "| role | Base categories | Novel categories | annotation categories used |",
        "|---|---:|---:|---:|",
    ]
    for role in ("OFFICIAL_TRAIN", "OFFICIAL_VAL", "CURRENT_TEST"):
        cat = roles[role]["category"]
        lines.append(
            f"| {role} | {cat['base_category_count']} | {cat['novel_category_count']} | {cat['annotation_category_count']} |"
        )
    lines += ["", "## Frontend and feature provenance", ""]
    train_frontend = audit["frontend_sources"]["official_train_covtrack_native"]
    lines.append(
        f"- Official Train frontend: **{train_frontend['status']}**. A Val cache, Test output, OVTR output, or the historical pilot cache cannot substitute for it."
    )
    val_frontend = audit["frontend_sources"]["official_val_covtrack_native"]
    lines.append(
        f"- Existing Val event cache: `{val_frontend['event_cache']['path']}`; optimizer source allowed: **{val_frontend['optimizer_source_allowed']}**."
    )
    lines.append(
        f"- Forbidden historical optimizer source: `{audit['pilot_guard']['forbidden_optimizer_source']}`."
    )
    lines.append(
        "- QDIC feature construction remains bound to the existing `qdic_features.py` and V9.1 event-cache builder; no second feature definition is introduced by this audit."
    )
    lines += ["", "## Gate", "", "Do not build the Official-Train cache or start DSSL training until an audited Train frontend output/manifest is bound to the exact `train` annotation and recorded in a new receipt.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports/tempotrack_v11/dssl_official",
    )
    args = parser.parse_args()
    audit = build_audit(args.train.resolve(), args.val.resolve(), args.test.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "split_audit.json"
    md_path = output_dir / "split_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0 if audit["split_disjoint"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
