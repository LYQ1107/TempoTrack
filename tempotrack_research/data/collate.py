"""Small deterministic collators for reference-based episode samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def _collate_action_tables(values: list[Mapping[str, Any]]) -> dict[str, Tensor]:
    """Pad complete per-state action tables without manufacturing valid actions."""
    fields = ("kind", "edge_index", "replacement_edge_index", "valid")
    lengths: list[int] = []
    converted: list[dict[str, Tensor]] = []
    for value in values:
        if any(field not in value for field in fields):
            raise ValueError("action table is missing a required field")
        item: dict[str, Tensor] = {}
        for field in fields:
            tensor = torch.as_tensor(value[field])
            if tensor.ndim != 1:
                raise ValueError(f"action table field {field} must be one-dimensional before collation")
            item[field] = tensor
        if len({int(item[field].shape[0]) for field in fields}) != 1:
            raise ValueError("action table fields have inconsistent lengths")
        lengths.append(int(item["kind"].shape[0]))
        converted.append(item)
    width = max(lengths, default=0)
    if width < 1:
        raise ValueError("action table must contain STOP")
    output = {
        "kind": torch.full((len(values), width), 3, dtype=torch.long),
        "edge_index": torch.full((len(values), width), -1, dtype=torch.long),
        "replacement_edge_index": torch.full((len(values), width), -1, dtype=torch.long),
        "valid": torch.zeros((len(values), width), dtype=torch.bool),
    }
    for row, (item, length) in enumerate(zip(converted, lengths)):
        output["kind"][row, :length] = item["kind"].long()
        output["edge_index"][row, :length] = item["edge_index"].long()
        output["replacement_edge_index"][row, :length] = item["replacement_edge_index"].long()
        output["valid"][row, :length] = item["valid"].bool()
    return output


def _pad_tensor(values: list[Tensor]) -> Tensor:
    if not values:
        raise ValueError("cannot collate an empty tensor list")
    if all(value.shape == values[0].shape for value in values):
        return torch.stack(values)
    ndim = max(value.ndim for value in values)
    shapes = [tuple(value.shape) + (1,) * (ndim - value.ndim) for value in values]
    target = tuple(max(shape[axis] for shape in shapes) for axis in range(ndim))
    padded: list[Tensor] = []
    for value in values:
        current = value.reshape(tuple(value.shape) + (1,) * (ndim - value.ndim))
        output = value.new_zeros(target)
        slices = tuple(slice(0, size) for size in current.shape)
        output[slices] = current
        padded.append(output)
    return torch.stack(padded)


def collate_training_batches(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("empty training batch")
    result: dict[str, Any] = {}
    keys = set().union(*(item.keys() for item in batch))
    for key in sorted(keys):
        values = [item.get(key) for item in batch]
        if key == "action_table" and all(isinstance(value, Mapping) for value in values):
            result[key] = _collate_action_tables(values)  # type: ignore[arg-type]
        elif all(torch.is_tensor(value) for value in values):
            result[key] = _pad_tensor(values)  # type: ignore[arg-type]
        elif all(isinstance(value, Mapping) for value in values):
            result[key] = collate_training_batches(values)  # type: ignore[arg-type]
        else:
            result[key] = values
    return result
