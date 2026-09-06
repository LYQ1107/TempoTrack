"""Small, dependency-light configuration and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .schemas import RunSpec


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def file_hash(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML when available, with a useful JSON fallback.

    PyYAML is present in the research environment but is not made a hard
    dependency of the import-safe inventory path.  JSON is a valid YAML
    subset, so minimal configs remain readable in a bare Python environment.
    """

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ImportError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Cannot read {path}: install PyYAML or provide JSON-compatible YAML"
            ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def dump_yaml(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False)
    except ImportError:
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def deep_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def resolve_path(repo: str | Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(repo) / path


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; lists/scalars are explicit replacements."""

    result: dict[str, Any] = {str(key): value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], *, path: str = "config") -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {path} keys: {sorted(unknown)}")


def resolved_config(
    method_config: Mapping[str, Any],
    suite_override: Mapping[str, Any] | None = None,
    local_override: Mapping[str, Any] | None = None,
    cli_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the documented precedence and return a JSON-safe snapshot."""

    value = deep_merge(method_config, suite_override or {})
    value = deep_merge(value, local_override or {})
    value = deep_merge(value, cli_override or {})
    value.setdefault("schema_version", 2)
    return _jsonable(value)


def build_run_spec(
    *,
    method: str,
    frontend: str,
    phase: str | None,
    config: Mapping[str, Any],
    run_root: str | Path,
    seed: int,
    provenance: Mapping[str, Any] | None = None,
) -> RunSpec:
    """Normalize a resolved mapping into the one runtime contract."""

    for name in ("model", "data", "optimizer", "schedule", "train", "infer", "evaluation"):
        if name not in config:
            config = {**config, name: {}}
    return RunSpec(
        method=method,
        frontend=frontend,
        phase=phase,
        model=dict(config.get("model", {})),
        data=dict(config.get("data", {})),
        optimizer=dict(config.get("optimizer", {})),
        schedule=dict(config.get("schedule", {})),
        train=dict(config.get("train", {})),
        infer=dict(config.get("infer", {})),
        evaluation=dict(config.get("evaluation", {})),
        seed=int(seed),
        run_root=Path(run_root).resolve(),
        provenance=dict(provenance or {}),
    )
