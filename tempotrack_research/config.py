"""Small, dependency-light configuration and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .schemas import RunSpec


@dataclass(frozen=True)
class ArtifactSignature:
    """Semantic identity shared by data, training and deployment artifacts."""

    data_semantics: str
    training_semantics: str
    deployment_semantics: str
    runtime_provenance: str
    schema_version: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "data_semantics": str(self.data_semantics),
            "training_semantics": str(self.training_semantics),
            "deployment_semantics": str(self.deployment_semantics),
            "runtime_provenance": str(self.runtime_provenance),
        }

    def content_hash(self) -> str:
        return object_hash(self.to_dict())

    def compatible_with(self, other: "ArtifactSignature") -> tuple[bool, list[str]]:
        return self.compatible_for(other, purpose="prediction_reuse")

    def compatible_for(self, other: "ArtifactSignature", *, purpose: str) -> tuple[bool, list[str]]:
        """Compare only the semantic layers relevant to an artifact use.

        Scheduler/report provenance is intentionally retained in metadata but
        never invalidates a training resume.  The old implementation compared
        the whole mapping at every call site, which made a harmless executor
        change look like a data or model change.
        """
        if purpose not in {"train_resume", "prediction_reuse", "report_reuse"}:
            raise ValueError(f"unknown artifact compatibility purpose: {purpose}")
        fields = ["data_semantics", "training_semantics"]
        if purpose in {"prediction_reuse", "report_reuse"}:
            fields.append("deployment_semantics")
        if purpose == "report_reuse":
            fields.append("schema_version")
        reasons: list[str] = []
        for field in fields:
            if getattr(self, field) != getattr(other, field):
                reasons.append(field)
        return not reasons, reasons

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactSignature":
        required = ("data_semantics", "training_semantics", "deployment_semantics", "runtime_provenance")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"artifact signature missing fields: {missing}")
        return cls(*(str(value[key]) for key in required), schema_version=int(value.get("schema_version", 4)))


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


def resolve_training_run_dir(
    run_root: str | Path,
    *,
    frontend: str,
    method: str,
    train_phase: str,
    profile: str,
    seed: int,
    variant: str = "default",
) -> Path:
    """Return the one canonical directory used by binders and runtime.

    ``run_root`` may be either the experiment root or its ``runs`` child.
    Variant names are explicit so an ablation cannot overwrite the default
    lineage.  No global checkpoint search is involved.
    """
    root = Path(run_root).expanduser().resolve()
    runs = root if root.name == "runs" else root / "runs"
    suffix = "" if profile == "trial" else f"_{profile}"
    variant_suffix = "" if variant in {"", "default", None} else f"_{variant}"
    return runs / f"{frontend}_{method}_{train_phase}_seed{int(seed)}{variant_suffix}{suffix}"


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
        loss=dict(config.get("loss", {})),
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
