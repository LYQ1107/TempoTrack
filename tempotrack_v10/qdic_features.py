"""Feature construction for TempoTrack's QDIC-MO candidate model.

The module deliberately keeps the statistical part separate from the learned
model.  A QDIC row is the existing 24-dimensional Q1 evidence vector followed
by nine causal distributional identity features.  The projected
Gaussian-Moment/MGF-inspired moments are evidence features, not a Gaussian
likelihood or fixed positive identity reward; no high-dimensional covariance is
materialized.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .query_conditioned_reranker import (
    FEATURE_NAMES,
    add_event_context,
    candidate_features,
)


QDIC_DISTRIBUTIONAL_FEATURE_NAMES = (
    "query_fast_cosine",
    "query_slow_cosine",
    "fast_slow_cosine",
    "projected_fast_mean",
    "projected_fast_variance",
    "projected_slow_mean",
    "projected_slow_variance",
    "projected_fast_mo",
    "projected_slow_mo",
)
QDIC_FEATURE_NAMES = tuple(FEATURE_NAMES) + QDIC_DISTRIBUTIONAL_FEATURE_NAMES
QDIC_RAW_DIM = len(QDIC_FEATURE_NAMES)
QDIC_INDEPENDENT_DIM = 19 + len(QDIC_DISTRIBUTIONAL_FEATURE_NAMES)
QDIC_FEATURE_SCHEMA_VERSION = 11
QDIC_MEMORY_CAPACITY = 64
QDIC_MEMORY_DEDUP_COS = 0.95
QDIC_RECENT_K = 8
QDIC_CONTEXT_CANDIDATE_TOP_K = 64
QDIC_DECISION_CANDIDATE_TOP_K = 8
QDIC_QUERY_OBSERVATIONS = 1


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _normalise_rows(value: Any, *, name: str) -> np.ndarray:
    array = _finite_array(value, name=name, ndim=2)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-8)


def projected_distribution_moments(
    cosine: np.ndarray,
    *,
    recent_k: int = 8,
) -> tuple[float, float, float, float, float, float]:
    """Return projected ``mu/variance/MO`` evidence for recent and full history.

    ``cosine`` is ``q @ Z.T``.  QDIC V11 uses one query observation, but this
    helper accepts ``[Q,L]`` and deterministically uses its first query row so
    callers cannot accidentally average padded query rows.  Variance is the
    population variance (``ddof=0``), and a singleton history has variance
    exactly zero.
    """
    values = _finite_array(cosine, name="cosine")
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("cosine must be a nonempty [Q,L] array")
    if int(recent_k) < 1:
        raise ValueError("recent_k must be positive")
    projected = values[0]
    fast = projected[-min(int(recent_k), len(projected)) :]
    slow = projected

    def stats(sample: np.ndarray) -> tuple[float, float, float]:
        mean = float(np.mean(sample, dtype=np.float64))
        variance = 0.0 if len(sample) <= 1 else float(np.var(sample, ddof=0, dtype=np.float64))
        return mean, variance, mean + 0.5 * variance

    fast_mean, fast_variance, fast_mo = stats(fast)
    slow_mean, slow_variance, slow_mo = stats(slow)
    result = (fast_mean, fast_variance, fast_mo, slow_mean, slow_variance, slow_mo)
    if not np.isfinite(np.asarray(result, dtype=np.float32)).all():
        raise FloatingPointError("projected distribution moments are non-finite")
    return result


def projected_mo(cosine: np.ndarray, *, recent_k: int = 8) -> dict[str, float]:
    """Named form of :func:`projected_distribution_moments` for diagnostics."""
    values = projected_distribution_moments(cosine, recent_k=recent_k)
    return {
        "projected_fast_mean": values[0],
        "projected_fast_variance": values[1],
        "projected_fast_mo": values[2],
        "projected_slow_mean": values[3],
        "projected_slow_variance": values[4],
        "projected_slow_mo": values[5],
    }


def build_qdic_candidate_features(
    cosine: np.ndarray,
    evidence: np.ndarray,
    gap: int,
    rank: int,
    *,
    query_fast_cosine: float,
    query_slow_cosine: float,
    fast_slow_cosine: float,
    recent_k: int = QDIC_RECENT_K,
    top_r: int = 3,
    max_gap: int = 360,
) -> np.ndarray:
    """Build one independent 28-D QDIC candidate row.

    The first 19 values and their ordering are exactly the controlled Q1
    ``candidate_features`` contract.  Event context (the final five Q1
    values) is added by :func:`build_qdic_event_features`, because it must see
    all context candidates at once.
    """
    direct = np.asarray(
        [query_fast_cosine, query_slow_cosine, fast_slow_cosine], dtype=np.float32
    )
    if not np.isfinite(direct).all():
        raise ValueError("QDIC direct fast/slow features must be finite")
    base = candidate_features(
        cosine,
        evidence,
        int(gap),
        int(rank),
        top_r=int(top_r),
        max_gap=max(int(max_gap), 1),
    )
    moments = projected_distribution_moments(cosine, recent_k=int(recent_k))
    row = np.concatenate(
        (
            np.asarray(base, dtype=np.float32),
            direct,
            np.asarray(
                [moments[0], moments[1], moments[3], moments[4], moments[2], moments[5]],
                dtype=np.float32,
            ),
        )
    ).astype(np.float32, copy=False)
    # Event competition adds the five legacy context columns later.  Keeping
    # this independent row at 19+9 makes it impossible for one candidate to
    # observe the other candidates before ``add_event_context`` is called.
    if row.shape != (QDIC_INDEPENDENT_DIM,) or not np.isfinite(row).all():
        raise FloatingPointError(
            f"QDIC independent candidate row must be finite [{QDIC_INDEPENDENT_DIM}]"
        )
    return row


def build_qdic_event_features(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Add Q1 event competition context to independent QDIC candidate rows.

    Each mapping must contain ``base_features`` (19-D) and the three direct
    prototype features.  This small public helper is also used by the online
    overlay, making the offline/online numerical path explicit.
    """
    if not rows:
        raise ValueError("QDIC event must contain at least one candidate")
    independent = np.stack(
        [np.asarray(item["base_features"], dtype=np.float32) for item in rows], axis=0
    )
    if independent.ndim != 2 or independent.shape[1] != 19:
        raise ValueError("QDIC event base_features must be [N,19]")
    context = add_event_context(independent)
    output = np.empty((len(rows), QDIC_RAW_DIM), dtype=np.float32)
    for index, item in enumerate(rows):
        direct_and_moments = np.asarray(item["distributional_features"], dtype=np.float32)
        if direct_and_moments.shape != (len(QDIC_DISTRIBUTIONAL_FEATURE_NAMES),):
            raise ValueError("distributional_features must be [9]")
        output[index] = np.concatenate((context[index], direct_and_moments))
    if not np.isfinite(output).all():
        raise FloatingPointError("QDIC event features are non-finite")
    return output


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_sidecar_producer_hashes() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[1]
    paths = (
        repo_root / "tempotrack_research/orchestration/v9_parameter_search.py",
        repo_root / "tempotrack_research/streaming/partial_support.py",
        repo_root / "tempotrack_research/memory/fixed_dual.py",
    )
    return {str(path): _sha256(path) for path in paths}


def _hash_for_basename(hashes: Mapping[str, Any], basename: str) -> str | None:
    return next(
        (
            str(value)
            for key, value in hashes.items()
            if Path(str(key)).name == basename
        ),
        None,
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _object_sha256(value: Any) -> str:
    """Hash a JSON object canonically so metadata exposes a real digest."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(
    event_cache: str | Path | Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    list[dict[str, Any]] | None,
    Path | None,
]:
    """Load an existing V9.1 cache without accepting legacy dense caches."""
    if isinstance(event_cache, Mapping):
        metadata = dict(event_cache.get("metadata", {}))
        arrays = {str(key): np.asarray(value) for key, value in dict(event_cache.get("arrays", {})).items()}
        rows = [dict(item) for item in event_cache.get("rows", [])]
        if not metadata or not arrays:
            raise ValueError("event_cache mapping must contain metadata, arrays and rows")
        return metadata, arrays, rows, None
    from tempotrack_research.orchestration.v9_parameter_search import _load_event_cache

    metadata, arrays, _ = _load_event_cache(event_cache)
    root = Path(event_cache)
    metadata_path = root if root.is_file() else root / "metadata.json"
    rows_path = Path(str(metadata.get("rows_path", metadata_path.parent / "events.jsonl")))
    if not rows_path.is_absolute():
        rows_path = metadata_path.parent / rows_path
    return metadata, arrays, None, rows_path.resolve()


def _load_sidecar(
    sidecar: str | Path | Mapping[str, Any],
    *,
    expected_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if isinstance(sidecar, Mapping):
        metadata = dict(sidecar.get("metadata", {}))
        raw_arrays = sidecar.get("arrays")
        if raw_arrays is None:
            raw_arrays = {
                key: sidecar[key]
                for key in (
                    "query_fast_cosine",
                    "query_slow_cosine",
                    "fast_slow_cosine",
                )
                if key in sidecar
            }
        arrays = {str(key): np.asarray(value) for key, value in dict(raw_arrays).items()}
        metadata_path = None
    else:
        root = Path(sidecar)
        metadata_path = root if root.is_file() else root / "metadata.json"
        if not metadata_path.exists() and not root.is_file() and (root / "qdic_sidecar.json").exists():
            metadata_path = root / "qdic_sidecar.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        array_paths = dict(metadata.get("arrays", {}))
        arrays = {}
        for name in (
            "query_fast_cosine",
            "query_slow_cosine",
            "fast_slow_cosine",
        ):
            array_path = Path(str(array_paths.get(name, metadata_path.parent / f"{name}.npy")))
            if not array_path.is_absolute():
                array_path = metadata_path.parent / array_path
            if not array_path.is_file():
                raise FileNotFoundError(f"QDIC sidecar array missing: {array_path}")
            expected_hash = dict(metadata.get("array_hashes", {})).get(name)
            if expected_hash is not None and _sha256(array_path) != str(expected_hash):
                raise ValueError(f"QDIC sidecar array hash mismatch: {array_path}")
            arrays[name] = np.load(array_path, mmap_mode="r", allow_pickle=False)
        metadata["metadata_hash"] = _sha256(metadata_path)
        producer_hashes = metadata.get("producer_source_hashes")
        if not isinstance(producer_hashes, Mapping):
            raise ValueError("QDIC sidecar producer source hashes missing")
        for name, current in _current_sidecar_producer_hashes().items():
            basename = Path(name).name
            expected = _hash_for_basename(producer_hashes, basename)
            if expected is None or expected != current:
                raise ValueError(
                    f"QDIC sidecar producer source hash mismatch: {basename}"
                )
    if metadata:
        artifact = metadata.get("artifact")
        if artifact is not None and artifact != "qdic_v11_projected_prototype_sidecar":
            raise ValueError("unsupported QDIC sidecar artifact")
        if "rows" in metadata and int(metadata["rows"]) != int(expected_count):
            raise ValueError("QDIC sidecar row count does not match event rows")
        for key, expected in (
            ("query_observations", 1),
            ("memory_capacity", QDIC_MEMORY_CAPACITY),
            ("recent_k", QDIC_RECENT_K),
        ):
            if key in metadata and int(metadata[key]) != expected:
                raise ValueError(f"QDIC sidecar {key} mismatch")
        for key, expected in (
            ("alpha_fast", 0.70),
            ("alpha_slow", 0.15),
            ("memory_dedup_cos", QDIC_MEMORY_DEDUP_COS),
        ):
            if key in metadata and not np.isclose(
                float(metadata[key]), expected, rtol=0.0, atol=1e-8
            ):
                raise ValueError(f"QDIC sidecar {key} mismatch")
    required = {"query_fast_cosine", "query_slow_cosine", "fast_slow_cosine"}
    if not required.issubset(arrays):
        raise ValueError(f"QDIC sidecar missing arrays: {sorted(required - set(arrays))}")
    for name in required:
        value = np.asarray(arrays[name], dtype=np.float32).reshape(-1)
        if len(value) != int(expected_count) or not np.isfinite(value).all():
            raise ValueError(f"QDIC sidecar {name} must be finite [{expected_count}]")
        arrays[name] = value
    return metadata, arrays


def _candidate_base(row: Mapping[str, Any]) -> bool:
    if "candidate_base" not in row:
        raise ValueError("event row is missing candidate_base; cannot prove Base-only supervision")
    return bool(row["candidate_base"])


def build_qdic_features(
    event_cache: str | Path | Mapping[str, Any],
    output: str | Path,
    *,
    sidecar: str | Path | Mapping[str, Any] | None = None,
    recent_k: int = QDIC_RECENT_K,
    alpha_fast: float = 0.70,
    alpha_slow: float = 0.15,
    memory_capacity: int = QDIC_MEMORY_CAPACITY,
    memory_dedup_cos: float = QDIC_MEMORY_DEDUP_COS,
    context_candidate_top_k: int = QDIC_CONTEXT_CANDIDATE_TOP_K,
    decision_candidate_top_k: int = QDIC_DECISION_CANDIDATE_TOP_K,
) -> dict[str, Any]:
    """Materialize a new 33-D feature cache from V9 event arrays + sidecar."""
    if int(recent_k) != QDIC_RECENT_K:
        raise ValueError("QDIC V11 requires recent_k=8")
    if not (0.0 <= float(alpha_slow) <= float(alpha_fast) < 1.0):
        raise ValueError("alpha_fast/alpha_slow must satisfy 0 <= slow <= fast < 1")
    if int(memory_capacity) != QDIC_MEMORY_CAPACITY or not np.isclose(
        float(memory_dedup_cos), QDIC_MEMORY_DEDUP_COS, rtol=0.0, atol=1e-8
    ):
        raise ValueError("QDIC V11 memory contract requires capacity=64 and dedup cosine=0.95")
    if int(context_candidate_top_k) != QDIC_CONTEXT_CANDIDATE_TOP_K:
        raise ValueError("QDIC V11 context candidate top-K must be 64")
    if int(decision_candidate_top_k) != QDIC_DECISION_CANDIDATE_TOP_K:
        raise ValueError("QDIC V11 decision candidate top-K must be 8")

    metadata, arrays, inline_rows, rows_path = _load_cache(event_cache)
    count = int(np.asarray(arrays["mem_len"]).shape[0])
    if inline_rows is not None and len(inline_rows) != count:
        raise ValueError("event cache rows and arrays have different lengths")
    if sidecar is None:
        value = metadata.get("qdic_sidecar")
        if value is None and not isinstance(event_cache, Mapping):
            root = Path(event_cache)
            value = root / "qdic_sidecar"
        if value is None:
            raise ValueError("QDIC feature building requires precomputed qdic sidecar")
        if not isinstance(event_cache, Mapping) and not Path(str(value)).is_absolute():
            root = Path(event_cache).resolve()
            root = root.parent if root.is_file() else root
            value = root / Path(str(value))
        sidecar = value
    sidecar_metadata, sidecar_arrays = _load_sidecar(sidecar, expected_count=count)

    cosine = np.asarray(arrays["cosine"])
    evidence = np.asarray(arrays["evidence"])
    mem_len = np.asarray(arrays["mem_len"], dtype=np.int64).reshape(-1)
    gap = np.asarray(arrays["gap"], dtype=np.int64).reshape(-1)
    rank = np.asarray(
        arrays.get("prefilter_rank_b1", arrays.get("prefilter_rank")), dtype=np.int64
    ).reshape(-1)
    group_ids = np.asarray(arrays["group_id"], dtype=np.int64).reshape(-1)
    label_array = None
    if "label" in arrays:
        label_array = np.asarray(arrays["label"], dtype=np.int8).reshape(-1)
    if "target_base" not in arrays:
        raise ValueError("event cache is missing target_base; cannot prove Base-only supervision")
    target_base_array = np.asarray(arrays["target_base"], dtype=bool).reshape(-1)
    if cosine.ndim != 3 or evidence.ndim != 3 or evidence.shape[-1] != 7:
        raise ValueError("event cache cosine/evidence arrays have invalid shapes")
    if not all(
        len(value) == count
        for value in (cosine, evidence, mem_len, gap, rank, group_ids, target_base_array)
    ):
        raise ValueError("event cache arrays are not row aligned")
    if label_array is not None and len(label_array) != count:
        raise ValueError("event cache label array is not row aligned")
    if (
        sidecar_metadata.get("event_rows_hash") is not None
        and not isinstance(event_cache, Mapping)
        and rows_path is not None
    ):
        if rows_path.is_file() and _sha256(rows_path) != str(sidecar_metadata["event_rows_hash"]):
            raise ValueError("QDIC sidecar event rows hash does not match event cache")
    event_metadata_hash = None if isinstance(event_cache, Mapping) else _sha256(
        Path(event_cache) if Path(event_cache).is_file() else Path(event_cache) / "metadata.json"
    )
    if (
        sidecar_metadata.get("event_cache_metadata_hash") is not None
        and event_metadata_hash is not None
        and str(sidecar_metadata["event_cache_metadata_hash"]) != event_metadata_hash
    ):
        raise ValueError("QDIC sidecar event-cache metadata hash does not match event cache")
    if cosine.shape[1] < 1:
        raise ValueError("QDIC requires a Q=1 cosine row")

    # V9 retains a union of B1/B2/B4 rows.  QDIC V11 is explicitly Q=1, so
    # only the B1 Top64 view is allowed to contribute to event context.
    output_row_count = int(np.count_nonzero(rank <= int(context_candidate_top_k)))
    if output_row_count < 1:
        raise ValueError("QDIC feature cache has no Q=1 context rows")

    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=False)
    array_specs = {
        "features": (np.float32, (output_row_count, QDIC_RAW_DIM)),
        "labels": (np.int8, (output_row_count,)),
        "supervision_allowed": (np.bool_, (output_row_count,)),
        "candidate_base": (np.bool_, (output_row_count,)),
        "target_base": (np.bool_, (output_row_count,)),
        "group_ids": (np.int64, (output_row_count,)),
    }
    array_paths: dict[str, str] = {
        name: str((output_path / f"{name}.npy").resolve())
        for name in (
            "features",
            "labels",
            "supervision_allowed",
            "offsets",
            "videos",
            "candidate_base",
            "target_base",
            "group_ids",
        )
    }
    maps = {
        name: np.lib.format.open_memmap(
            array_paths[name],
            mode="w+",
            dtype=dtype,
            shape=shape,
        )
        for name, (dtype, shape) in array_specs.items()
    }

    def iter_event_rows():
        if inline_rows is not None:
            for index, row in enumerate(inline_rows):
                yield index, row
            return
        if rows_path is None:
            raise ValueError("path-based event cache is missing events.jsonl")
        processed = 0
        with rows_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if processed >= count:
                    raise ValueError("QDIC event rows exceed event cache array count")
                row = json.loads(line)
                index = processed
                processed += 1
                yield index, row
        if processed != count:
            raise ValueError("QDIC event rows and event cache arrays have different lengths")

    written = 0
    event_videos: list[int] = []
    offsets = [0]
    current_group_id: int | None = None
    current_event_key: tuple[int, int] | None = None
    current_items: list[tuple[int, Mapping[str, Any]]] = []
    seen_groups: set[int] = set()

    def flush_current_group() -> None:
        nonlocal written
        if current_group_id is None or not current_items:
            return
        independent: list[np.ndarray] = []
        distributional: list[np.ndarray] = []
        for index, row in current_items:
            length = int(mem_len[index])
            if length < 1 or length > int(cosine.shape[2]) or length > int(evidence.shape[1]):
                raise ValueError(f"invalid memory length at event row {index}: {length}")
            qcos = np.asarray(cosine[index, :1, :length], dtype=np.float32)
            ev = np.asarray(evidence[index, :length], dtype=np.float32)
            moments = projected_distribution_moments(qcos, recent_k=int(recent_k))
            independent_row = candidate_features(
                qcos,
                ev,
                int(gap[index]),
                int(rank[index]),
                top_r=3,
                max_gap=max(int(metadata.get("max_gap", 360)), 1),
            )
            distributional_row = np.asarray(
                [
                    sidecar_arrays["query_fast_cosine"][index],
                    sidecar_arrays["query_slow_cosine"][index],
                    sidecar_arrays["fast_slow_cosine"][index],
                    moments[0],
                    moments[1],
                    moments[3],
                    moments[4],
                    moments[2],
                    moments[5],
                ],
                dtype=np.float32,
            )
            independent.append(independent_row)
            distributional.append(distributional_row)

        context = add_event_context(np.stack(independent, axis=0))
        for local_index, (index, row) in enumerate(current_items):
            complete = np.concatenate((context[local_index], distributional[local_index])).astype(
                np.float32, copy=False
            )
            if complete.shape != (QDIC_RAW_DIM,) or not np.isfinite(complete).all():
                raise FloatingPointError("non-finite QDIC feature row")
            label = int(row["label"])
            cand_base = _candidate_base(row)
            targ_base = bool(row["target_base"])
            if targ_base != bool(target_base_array[index]):
                raise ValueError("event row/target_base array mismatch")
            maps["features"][written] = complete
            maps["labels"][written] = label
            maps["candidate_base"][written] = cand_base
            maps["target_base"][written] = targ_base
            maps["supervision_allowed"][written] = bool(targ_base and cand_base and label in (0, 1))
            maps["group_ids"][written] = int(current_group_id)
            written += 1
        if current_event_key is None:
            raise ValueError("QDIC event group is missing its video/target key")
        event_videos.append(int(current_event_key[0]))
        offsets.append(written)

    for index, row in iter_event_rows():
        group_id = int(group_ids[index])
        row_key = (int(row["video_id"]), int(row["target_serial"]))
        if "label" not in row or "target_base" not in row:
            raise ValueError(f"QDIC event row {index} is missing label/target_base")
        if label_array is not None and int(row["label"]) != int(label_array[index]):
            raise ValueError("event row/label array mismatch")
        if bool(row["target_base"]) != bool(target_base_array[index]):
            raise ValueError("event row/target_base array mismatch")
        if int(row.get("prefilter_rank_b1", rank[index])) != int(rank[index]):
            raise ValueError("event row/rank array mismatch")
        if current_group_id is None:
            current_group_id = group_id
            current_event_key = row_key
        elif group_id != current_group_id:
            flush_current_group()
            seen_groups.add(current_group_id)
            if group_id in seen_groups:
                raise ValueError("QDIC event groups are noncontiguous")
            current_group_id = group_id
            current_event_key = row_key
            current_items = []
        elif row_key != current_event_key:
            raise ValueError("QDIC event group video_id/target_serial mismatch")
        if int(rank[index]) <= int(context_candidate_top_k):
            if len(current_items) >= int(context_candidate_top_k):
                raise ValueError("QDIC event group exceeds context candidate top-K")
            current_items.append((index, row))

    flush_current_group()
    if written != output_row_count:
        raise ValueError(
            f"QDIC feature row count mismatch: wrote {written}, expected {output_row_count}"
        )
    if len(offsets) != len(event_videos) + 1 or offsets[-1] != written:
        raise ValueError("QDIC feature event offsets are inconsistent")
    for value in maps.values():
        value.flush()
    del maps
    np.save(array_paths["offsets"], np.asarray(offsets, dtype=np.int64), allow_pickle=False)
    np.save(array_paths["videos"], np.asarray(event_videos, dtype=np.int64), allow_pickle=False)

    metadata_path = output_path / "features.json"
    source_metadata_path = None if isinstance(event_cache, Mapping) else (
        Path(event_cache) if Path(event_cache).is_file() else Path(event_cache) / "metadata.json"
    )
    event_cache_hash = None if source_metadata_path is None else _sha256(source_metadata_path)
    manifest_hash = metadata.get("manifest_hash")
    rows_hash = None if rows_path is None or not rows_path.is_file() else _sha256(rows_path)
    source_hashes = {
        "event_cache_metadata": event_cache_hash,
        "event_cache_manifest": manifest_hash,
        "event_cache_rows": rows_hash,
        "qdic_sidecar_metadata": sidecar_metadata.get("metadata_hash"),
        "qdic_sidecar_event_rows_hash": sidecar_metadata.get("event_rows_hash"),
    }
    feature_config = {
        "query_observations": QDIC_QUERY_OBSERVATIONS,
        "recent_k": int(recent_k),
        "alpha_fast": float(alpha_fast),
        "alpha_slow": float(alpha_slow),
        "memory_capacity": int(memory_capacity),
        "memory_dedup_cos": float(memory_dedup_cos),
        "context_candidate_top_k": int(context_candidate_top_k),
        "decision_candidate_top_k": int(decision_candidate_top_k),
        "top_r": 3,
        "max_gap": int(metadata.get("max_gap", 360)),
        "min_gap": int(metadata.get("min_gap", 0)),
    }
    result_metadata: dict[str, Any] = {
        "schema_version": QDIC_FEATURE_SCHEMA_VERSION,
        "artifact": "qdic_v11_feature_cache",
        "protocol": "QDIC_V11_BASE_ONLY_TRAINING",
        "split": str(metadata.get("split", "unknown")),
        "feature_names": list(QDIC_FEATURE_NAMES),
        "feature_dim": QDIC_RAW_DIM,
        "query_observations": QDIC_QUERY_OBSERVATIONS,
        "recent_k": int(recent_k),
        "alpha_fast": float(alpha_fast),
        "alpha_slow": float(alpha_slow),
        "memory_capacity": int(memory_capacity),
        "memory_dedup_cos": float(memory_dedup_cos),
        "context_candidate_top_k": int(context_candidate_top_k),
        "decision_candidate_top_k": int(decision_candidate_top_k),
        "base_only_supervision": True,
        "novel_gt_used_for_optimizer": False,
        "test_weights_used": False,
        "test_gt_used_for_optimizer": bool(metadata.get("test_gt_used_for_optimizer", False)),
        "source_role": metadata.get("source_role"),
        "exact_split_name": metadata.get("exact_split_name"),
        "optimizer_source_allowed": metadata.get("optimizer_source_allowed"),
        "video_disjoint_split": metadata.get("video_disjoint_split"),
        "normalization_fit": metadata.get("normalization_fit"),
        "official_train_annotation_sha256": metadata.get("official_train_annotation_sha256"),
        "source_annotation": metadata.get("source_annotation", metadata.get("annotation")),
        "source_annotation_sha256": metadata.get(
            "source_annotation_sha256", metadata.get("annotation_hash")
        ),
        "source_event_cache": metadata.get("source_event_cache"),
        "source_frontend_cache": metadata.get("source_frontend_cache"),
        "supervision_source": metadata.get("supervision_source"),
        "input_source": metadata.get("input_source"),
        "oracle_features_used": metadata.get("oracle_features_used"),
        "gt_boxes_used_as_model_input": metadata.get("gt_boxes_used_as_model_input"),
        "gt_tracks_used_as_memory": metadata.get("gt_tracks_used_as_memory"),
        "gt_used_only_for_supervision": metadata.get("gt_used_only_for_supervision"),
        "feature_config": feature_config,
        "source_hashes": source_hashes,
        "event_cache": None if isinstance(event_cache, Mapping) else str(Path(event_cache).resolve()),
        "event_cache_manifest": metadata.get("manifest"),
        "event_cache_hash": event_cache_hash,
        "manifest_hash": manifest_hash,
        "rows_hash": rows_hash,
        "event_cache_manifest_hash": manifest_hash,
        "qdic_sidecar": None if isinstance(sidecar, Mapping) else str(Path(sidecar).resolve()),
        "events": len(offsets) - 1,
        "rows": written,
        "arrays": array_paths,
    }
    result_metadata["array_hashes"] = {
        name: _sha256(path) for name, path in array_paths.items()
    }
    result_metadata["arrays_hash"] = _object_sha256(result_metadata["array_hashes"])
    metadata_path.write_text(
        json.dumps(result_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"status": "COMPLETED", "output": str(output_path), **result_metadata}


__all__ = [
    "QDIC_CONTEXT_CANDIDATE_TOP_K",
    "QDIC_DECISION_CANDIDATE_TOP_K",
    "QDIC_DISTRIBUTIONAL_FEATURE_NAMES",
    "QDIC_FEATURE_NAMES",
    "QDIC_RAW_DIM",
    "QDIC_FEATURE_SCHEMA_VERSION",
    "QDIC_INDEPENDENT_DIM",
    "QDIC_MEMORY_CAPACITY",
    "QDIC_MEMORY_DEDUP_COS",
    "QDIC_QUERY_OBSERVATIONS",
    "QDIC_RECENT_K",
    "build_qdic_candidate_features",
    "build_qdic_event_features",
    "build_qdic_features",
    "projected_distribution_moments",
    "projected_mo",
]
