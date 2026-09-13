#!/usr/bin/env python3
"""Run the MASA-R50 downstream lane as an early, isolated sidecar.

The canonical MASA supervisor conservatively waits for OVTrack, although the
actual MASA input contract is the audited, frozen COV public-detection tree.
This sidecar uses a distinct output/state root and therefore cannot overwrite
the canonical supervisor's outputs.  It deliberately reuses the canonical
MASA command construction and result parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v10_v104_masa_downstream as canonical  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument(
        "--only-split",
        choices=("val", "test"),
        default=None,
        help="Retry only one split; by default run both splits.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    state_root = args.state_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    # Rebind only this process's module global.  The canonical supervisor and
    # its outputs remain untouched in their original root.
    canonical.DOWNSTREAM_ROOT = output_root
    canonical.RUN_CWD = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa")
    runner = canonical.MasaRunner(state_root)
    runner.state["execution_mode"] = "EARLY_PARALLEL_ISOLATED_ROOT"
    runner.state["wait_ov_bypassed_reason"] = (
        "audited COV public detections are the direct MASA input; "
        "canonical supervisor retains its OV wait"
    )
    runner.save(current_stage="MASA_PREFLIGHT", next_action="launch four isolated MASA jobs")

    splits = (args.only_split,) if args.only_split else ("val", "test")
    jobs = [(split, method) for split in splits for method in ("native", "tempo")]
    if not runner.run_parallel(jobs):
        runner.state["status"] = "BLOCKED"
        runner.save(current_stage="MASA_RUN", next_action="inspect early sidecar logs")
        return 2

    runner.write_results(jobs)
    runner.state["status"] = "COMPLETED"
    runner.state["current_stage"] = "COMPLETED"
    runner.state["next_action"] = "compare with canonical MASA lane when available"
    runner.state["completed_at"] = canonical.iso()
    runner.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
