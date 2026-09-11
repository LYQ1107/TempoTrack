# TRACT source audit

**Audit status: `TRACT_ASSOCIATION_CODE_MISSING`**

Audit time: 2026-09-12T00:52:50+08:00
Agent lane: V10.3 Agent D
Worktree: `/data1/LWR/vranlee/SERVER_ONLY/avis/v10_tract`
Branch: `codex/v10-tract`
TempoTrack audit HEAD: `aa30fba4ebc4739e6a5936cb4fa3454bb13805f`
TempoTrack tree: `e1d1569c516584299a6c6e952c28d5a6658f4b1b`

## Scope and rule applied

This is the V10.3 D0 source-completeness audit. The allowed integration path requires locating all three real TRACT runtime stages:

1. trajectory-aware association feature path;
2. final native affinity/similarity computation;
3. final ID/new-ID assignment before state/memo update.

The V10.3 task explicitly forbids reconstructing TCR/TFA from the paper when these paths are absent. No shared TempoTrack core was copied, and no TRACT adapter or guessed tracker was implemented.

## Repository and worktree audit

The requested worktree exists and was clean before this report was added:

```text
git rev-parse --show-toplevel
/data1/LWR/vranlee/SERVER_ONLY/avis/v10_tract

git rev-parse HEAD
aa30fba4ebc4739e6a5936cb4f4fa3454bb13805f

git branch --show-current
codex/v10-tract
```

The local branch currently tracks `origin/codex/tempotrack-v9-8gpu-r50-audit`, not a dedicated V10 remote branch. The requested read-only refresh was attempted with `git fetch --all --prune`, but the shared worktree Git administration area is mounted read-only:

```text
error: cannot open
/data1/LWR/vranlee/SERVER_ONLY/avis/masa/.git/worktrees/v10_tract/FETCH_HEAD:
Read-only file system
```

This did not change the source tree. Local refs show the V10 agent branches at the same `aa30fba` snapshot. The target worktree had no source modifications before this report.

Submodule/LFS/tree checks:

```text
git submodule status --recursive
NO_GITMODULES

git lfs ls-files
git: 'lfs' is not a git command
```

The TempoTrack worktree contains 1,703 tracked files at this HEAD. The only files whose names contain `tract`, `traclip`, `tcr`, `tfa`, or `trajectory` are generic TempoTrack trajectory/training utilities and reports; there is no TRACT upstream implementation. The MASA tracker files in this worktree are MASA/TempoTrack code, not evidence of TRACT's TCR/TFA implementation.

The V10.3 instruction file used for this audit was read in full. SHA-256:

```text
de14b58df6ce3a64397076626dbb2cbf904f47bcfe992f987334d57bbb7aa2c5
```

## Audited upstream TRACT clones

Two local clones were checked read-only:

| Path | HEAD | refs / status |
|---|---|---|
| `/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/third_party/TRACT` | `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` | clean `main`, `origin/main`, no local tags, no submodules |
| `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_reference_repos/tract` | `19f01d72f9f6c212c28fd9cb0171a5432cd41a6a` | same tree/commit, separately materialized clone |

The first clone's Git tree is `0dff9aa022bd647b9a1ffd49c72ec861dbc293b5`, with 1,273 tracked files (1,000 Python files). Its top-level tree is only:

```text
.gitignore
README.md
TraCLIP/
assets/
masa/
```

The two local clones contain no `.pth`, `.pt`, `.ckpt`, `.bin`, `.pkl`, or `.pickle` checkpoint/artifact file. A direct remote ref query was also attempted, but outbound access is unavailable in this environment:

```text
git ls-remote --heads --tags https://github.com/Nathan-Li123/TRACT.git
fatal: unable to access ... Couldn't connect to server
```

Therefore no unobserved remote branch/tag/release was assumed.

## What is actually present in the public tree

### `masa/`

The checked-out `masa/masa` package contains only:

```text
__init__.py
apis/__init__.py
apis/masa_inference.py
version.py
visualization/__init__.py
visualization/visualizer.py
```

There is no `masa/models/`, no `masa/models/tracker/`, no `MasaTaoTracker`, no tracker `track()` implementation, and no association module in this TRACT clone. `masa/tools/convert_ovtrack_to_masa.py` is an observation-format conversion utility; it does not implement tracking, affinity, or ID assignment.

### `TraCLIP/`

`TraCLIP/main.py` contains the trajectory classification training/evaluation loop. The supporting tools match/convert/extract tracklets and transfer results to TETA. They consume tracklets or already-produced tracking results; they do not provide the online TRACT association decision path.

The actual searches were:

```text
rg -n -i 'TCR|TFA|TSE|trajectory-aware' TraCLIP masa --glob '*.py' --glob '*.sh' --glob '*.md'
```

No TCR/TFA/TSE implementation was returned. The few `tcr` matches are output/path strings such as `masa-ovtrack-ovtb_tcr` or `outfile_prefix=...tcr`, not definitions.

```text
rg -n -i 'class .*Tracker|def track\s*\(|def match\s*\(|affinity|memo_ids|new_id|d2t_scores|t2d_scores' TraCLIP masa --glob '*.py'
```

No tracker class, online `track()`/`match()` function, native affinity matrix, memo-ID update, or new-ID allocator was found in the TRACT clone.

## Installed package and server clone checks

In `masaenv`, `pip show`/package inspection found no installed `tract`, `traclip`, `ovtrack`, or pip-installed `masa` package. There is no separate TRACT package directory under the audited environment site-packages.

Importing the external clone with its own `masa` path fails because that clone is incomplete:

```text
PYTHONPATH=.../third_party/TRACT/masa:.../third_party/TRACT/TraCLIP python -c ...
masa -> /data1/.../third_party/TRACT/masa/masa/__init__.py
masa.models -> ModuleNotFoundError("No module named 'masa.datasets'")
masa.models.tracker -> ModuleNotFoundError("No module named 'masa.datasets'")
masa.models.tracker.masa_tao_tracker -> ModuleNotFoundError("No module named 'masa.datasets'")
```

The normal `masaenv` import from the requested TempoTrack worktree resolves the local TempoTrack MASA package, not TRACT; it also stops on the environment's missing `nltk` dependency before importing the model registry. This is not evidence of a hidden TRACT tracker.

An additional exact-name search under the server's existing clones found the two TRACT copies above and no separate `tcr.py`, `tfa.py`, or named TRACT checkpoint. Existing VOV/COV/MASA source trees were not modified.

## Required runtime locator results

| Required TRACT runtime component | Result | Evidence |
|---|---|---|
| trajectory-aware association feature (TCR/TFA feature path) | **NOT LOCATED** | no tracker/model association source in either `19f01d72` clone |
| final native affinity/similarity matrix | **NOT LOCATED** | no `track()`/`match()`/affinity implementation in TRACT clone |
| final track-ID / new-ID assignment | **NOT LOCATED** | no tracker state/memo/ID allocator in TRACT clone |
| TraCLIP classification | **PRESENT, separate** | `TraCLIP/main.py` and tools; classification consumes tracklets and is not the association core |
| detector configuration | **PARTIAL** | README says detector is replaceable; no complete TRACT association runtime is present |
| TRACT checkpoint/artifact | **NOT FOUND locally** | no checkpoint files in either audited TRACT clone |

## Gate decision

`TRACT_SOURCE_COMPLETE` is not supportable from the available source. `TRACT_SOURCE_PARTIAL` would be misleading for the required association integration because all three required runtime points are absent. The correct allowed status is:

```text
TRACT_ASSOCIATION_CODE_MISSING
```

Consequences required by V10.3:

- no TRACT native reproduction was run;
- no TRACT Val/Test metrics are claimed;
- no insertion-point report was produced, because there is no valid pre-association hook to locate;
- no TCR/TFA approximation was written;
- no shared `TempoTrackOverlay` was copied or integrated;
- no detector stream was fabricated to compensate for missing association code.

The only valid next step would be to obtain the missing official association source/checkpoint package (or a user-authorized, verifiable artifact containing the actual tracker) and rerun this audit. Until then this lane is blocked specifically by missing upstream association code, not by an algorithmic result.

## Audit commands and reproducibility notes

The audit used read-only checks equivalent to the V10.3 D0 requirements:

```text
git status --short --branch
git rev-parse HEAD
git branch -a -vv
git tag -l
git submodule status --recursive
git lfs ls-files
find . -maxdepth 6 -type f | sort
rg -n -i 'TCR|TFA|trajectory|MasaTaoTracker|def track|track_id|association|affinity|similarity|match_score|memo|TraCLIP'
python package/import inspection in masaenv
existing-clone and local-artifact searches
```

No user code, data, security configuration, healthy V9 task, or other worktree was changed during the audit.
