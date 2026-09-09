from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

from src.qwen.input import move_inputs


VISION_END_ID = 248054


def _torch() -> Any:
    import torch

    return torch


def trim_inputs(inputs: dict[str, Any], start: int | None = None, end: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "shape"):
            out[key] = value
        elif key in {"input_ids", "attention_mask", "mm_token_type_ids", "position_ids"}:
            out[key] = value[:, start:end]
        else:
            out[key] = value
    return out


def visual_split(inputs: dict[str, Any]) -> int:
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    ends = [idx for idx, token_id in enumerate(ids) if token_id == VISION_END_ID]
    if not ends:
        raise RuntimeError("no <|vision_end|> token found")
    return max(ends) + 1


def shares_prefix(inputs: dict[str, Any], ref_tokens: list[int], split: int) -> bool:
    ids = inputs["input_ids"][0, :split].detach().cpu().tolist()
    return ids == ref_tokens[:split]


def reset_rope_deltas(model: Any) -> None:
    model_model = getattr(model, "model", None)
    if model_model is not None and hasattr(model_model, "rope_deltas"):
        model_model.rope_deltas = None


def full_position_ids(model: Any, inputs: dict[str, Any], device: Any) -> Any | None:
    model_model = getattr(model, "model", None)
    if model_model is None or not hasattr(model_model, "compute_3d_position_ids"):
        return None
    moved = move_inputs(inputs, device)
    input_embeds = model_model.get_input_embeddings()(moved["input_ids"]) if hasattr(model_model, "get_input_embeddings") else None
    return model_model.compute_3d_position_ids(
        input_ids=moved.get("input_ids"),
        image_grid_thw=moved.get("image_grid_thw"),
        video_grid_thw=moved.get("video_grid_thw"),
        inputs_embeds=input_embeds,
        attention_mask=moved.get("attention_mask"),
        past_key_values=None,
        mm_token_type_ids=moved.get("mm_token_type_ids"),
    )


def fork_cache(cache: Any) -> Any:
    return copy.deepcopy(cache)


class ExactPrefixCache:
    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self.cache_by_key: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()

    def clear(self) -> None:
        self.cache_by_key.clear()

    def get(self, key: tuple[Any, ...], inputs: dict[str, Any]) -> dict[str, Any] | None:
        split = visual_split(inputs)
        item = self.cache_by_key.get(key)
        if item is None or item["split"] != split or not shares_prefix(inputs, item["tokens"], split):
            return None
        self.cache_by_key.move_to_end(key)
        return item

    def remember(self, key: tuple[Any, ...], inputs: dict[str, Any], cache: Any) -> dict[str, Any]:
        split = visual_split(inputs)
        item = {
            "split": split,
            "tokens": inputs["input_ids"][0, :split].detach().cpu().tolist(),
            "cache": cache,
        }
        self.cache_by_key[key] = item
        self.cache_by_key.move_to_end(key)
        while len(self.cache_by_key) > self.max_entries:
            self.cache_by_key.popitem(last=False)
        return item
