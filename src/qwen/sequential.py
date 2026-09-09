from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.cache.prefix_cache import VISION_END_ID, fork_cache, full_position_ids, reset_rope_deltas
from src.qwen.input import apply_messages, image_content, move_inputs


def _torch() -> Any:
    import torch
    return torch


def _seq_slice(value: Any, start: int, end: int) -> Any:
    if value is None or not hasattr(value, "shape"):
        return value
    # input_ids / attention_mask / mm_token_type_ids: [B, S]
    # multimodal position_ids may be [3, B, S].
    return value[..., start:end]


def _vision_ends(inputs: dict[str, Any]) -> list[int]:
    ids = inputs["input_ids"][0].detach().cpu().tolist()
    return [idx + 1 for idx, token_id in enumerate(ids) if int(token_id) == VISION_END_ID]


def _pixel_axis_and_ranges(pixel_values: Any, image_grid_thw: Any) -> tuple[int, list[tuple[int, int]]]:
    """
    Qwen VL processors flatten visual patches across images. image_grid_thw gives
    one [t,h,w] row per image. This derives per-image slices and verifies that one
    tensor axis matches the summed raw patch count.

    The function deliberately fails loudly if the installed processor uses a
    different layout; silently feeding mismatched pixel tensors would invalidate
    the sequential-cache experiment.
    """
    if pixel_values is None or image_grid_thw is None:
        raise RuntimeError("sequential visual prefill requires pixel_values and image_grid_thw")

    counts: list[int] = []
    for row in image_grid_thw.detach().cpu().tolist():
        t, h, w = (int(x) for x in row)
        counts.append(t * h * w)
    total = sum(counts)

    shape = tuple(int(x) for x in pixel_values.shape)
    candidate_axes = [axis for axis, size in enumerate(shape) if size == total]
    if not candidate_axes:
        raise RuntimeError(
            f"cannot map pixel_values shape={shape} to image_grid_thw patch counts={counts} total={total}"
        )

    # Qwen-VL normally uses axis 0 for flattened patches. Prefer it when possible.
    axis = 0 if 0 in candidate_axes else candidate_axes[0]
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for count in counts:
        ranges.append((cursor, cursor + count))
        cursor += count
    return axis, ranges


def _slice_axis(value: Any, axis: int, start: int, end: int) -> Any:
    index = [slice(None)] * value.ndim
    index[axis] = slice(start, end)
    return value[tuple(index)]


def _encode(processor: Any, images: list[Any], prompt: str, *, thinking: bool) -> dict[str, Any]:
    messages = [{
        "role": "user",
        "content": image_content(images, prompt, no_think=not thinking),
    }]
    return apply_messages(processor, messages, thinking=thinking)


def _full_positions(model: Any, inputs: dict[str, Any], device: Any) -> Any:
    reset_rope_deltas(model)
    positions = full_position_ids(model, inputs, device)
    if positions is None:
        positions = inputs.get("position_ids")
    if positions is None:
        raise RuntimeError(
            "Qwen runtime did not expose multimodal position_ids; refusing unsafe sequential prefill"
        )
    return positions


def _common_kwargs(
    inputs: dict[str, Any],
    positions: Any,
    device: Any,
    *,
    start: int,
    end: int,
    cache: Any,
) -> dict[str, Any]:
    torch = _torch()
    kwargs: dict[str, Any] = {
        "input_ids": inputs["input_ids"][..., start:end],
        # With PKV, a 2-D attention mask should describe past + current tokens.
        "attention_mask": inputs["attention_mask"][..., :end],
        "position_ids": positions[..., start:end],
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
        "output_attentions": False,
        "cache_position": torch.arange(start, end, dtype=torch.long),
        "logits_to_keep": 1,
    }
    if "mm_token_type_ids" in inputs:
        kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"][..., start:end]
    return move_inputs(kwargs, device)


def _prefill_visual_prefix(
    processor: Any,
    model: Any,
    device: Any,
    images: list[Any],
    prompt: str,
    *,
    thinking: bool,
) -> tuple[Any, int, dict[str, Any], Any, dict[str, Any]]:
    """
    Build one canonical interleaved multimodal sequence, but execute its visual
    prefix incrementally: image block 1 -> cache -> image block 2 -> cache -> ...

    This avoids independently-computed KV concatenation. Each later image is
    forwarded with the accumulated PKV from all preceding visual blocks.
    """
    if not images:
        raise ValueError("sequential prefill needs at least one image")

    inputs = _encode(processor, images, prompt, thinking=thinking)
    ends = _vision_ends(inputs)
    if len(ends) != len(images):
        raise RuntimeError(
            f"expected {len(images)} <|vision_end|> tokens, found {len(ends)}; "
            "chat template / processor layout changed"
        )

    positions = _full_positions(model, inputs, device)
    pixel_values = inputs.get("pixel_values")
    grid = inputs.get("image_grid_thw")
    pixel_axis, pixel_ranges = _pixel_axis_and_ranges(pixel_values, grid)

    cache = None
    start = 0
    t0 = time.perf_counter()
    chunk_audit: list[dict[str, Any]] = []

    torch = _torch()
    for image_idx, end in enumerate(ends):
        kwargs = _common_kwargs(
            inputs, positions, device, start=start, end=end, cache=cache
        )

        p0, p1 = pixel_ranges[image_idx]
        kwargs["pixel_values"] = move_inputs(
            {"x": _slice_axis(pixel_values, pixel_axis, p0, p1)}, device
        )["x"]
        kwargs["image_grid_thw"] = move_inputs(
            {"x": grid[image_idx:image_idx + 1]}, device
        )["x"]

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            outputs = model(**kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        cache = outputs.past_key_values
        chunk_audit.append({
            "image_index": image_idx,
            "token_start": start,
            "token_end": end,
            "token_count": end - start,
            "pixel_patch_start": p0,
            "pixel_patch_end": p1,
            "pixel_patch_count": p1 - p0,
        })
        start = end

    audit = {
        "visual_images": len(images),
        "visual_token_end": start,
        "total_tokens": int(inputs["input_ids"].shape[-1]),
        "chunks": chunk_audit,
        "visual_prefill_sec": time.perf_counter() - t0,
    }
    return cache, start, inputs, positions, audit


def _tail_logits(
    model: Any,
    device: Any,
    inputs: dict[str, Any],
    positions: Any,
    cache: Any,
    start: int,
) -> tuple[Any, float]:
    torch = _torch()
    end = int(inputs["input_ids"].shape[-1])
    if end <= start:
        raise RuntimeError(f"empty query tail: split={start} total={end}")

    kwargs = _common_kwargs(
        inputs, positions, device, start=start, end=end, cache=cache
    )
    # No visual tensors belong in the text/query tail.
    kwargs.pop("pixel_values", None)
    kwargs.pop("image_grid_thw", None)

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        outputs = model(**kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return outputs.logits[0, -1, :].detach().float().cpu(), time.perf_counter() - t0


def _verify_visual_prefix(a: dict[str, Any], b: dict[str, Any], split: int) -> None:
    aa = a["input_ids"][0, :split].detach().cpu().tolist()
    bb = b["input_ids"][0, :split].detach().cpu().tolist()
    if aa != bb:
        raise RuntimeError(
            "forward/reverse prompts changed tokens inside the visual prefix; "
            "cannot safely fork one visual cache"
        )


def _score_from_logits(
    f_logits: Any,
    r_logits: Any,
    *,
    a_id: int,
    b_id: int,
) -> dict[str, Any]:
    f_a = float(f_logits[a_id])
    f_b = float(f_logits[b_id])
    r_a = float(r_logits[a_id])
    r_b = float(r_logits[b_id])

    forward_margin = f_b - f_a       # forward: A=No, B=Yes
    reverse_margin = r_a - r_b       # reverse: A=Yes, B=No
    score = 0.5 * (forward_margin + reverse_margin)

    return {
        "score": score,
        "pred": int(score > 0),
        "forward_A_logit": f_a,
        "forward_B_logit": f_b,
        "reverse_A_logit": r_a,
        "reverse_B_logit": r_b,
        "forward_margin": forward_margin,
        "reverse_margin": reverse_margin,
        "forward_support": int(forward_margin > 0),
        "reverse_support": int(reverse_margin > 0),
        "support_status": "SUPPORT" if forward_margin > 0 and reverse_margin > 0 else "REJECT",
    }


@dataclass
class SequentialABScorer:
    processor: Any
    model: Any
    device: Any
    a_id: int
    b_id: int
    thinking: bool = False

    def score(
        self,
        images: list[Any],
        forward: str,
        reverse: str,
    ) -> dict[str, Any]:
        """
        Sequentially prefill every image block into one accumulated cache, then
        fork that exact visual cache for forward/reverse text tails.
        """
        base_cache, split, f_inputs, f_positions, audit = _prefill_visual_prefix(
            self.processor,
            self.model,
            self.device,
            images,
            forward,
            thinking=self.thinking,
        )

        r_inputs = _encode(self.processor, images, reverse, thinking=self.thinking)
        r_positions = _full_positions(self.model, r_inputs, self.device)
        r_ends = _vision_ends(r_inputs)
        if not r_ends or r_ends[-1] != split:
            raise RuntimeError(
                f"forward/reverse visual split mismatch: forward={split} reverse={r_ends[-1] if r_ends else None}"
            )
        _verify_visual_prefix(f_inputs, r_inputs, split)

        f_logits, f_sec = _tail_logits(
            self.model, self.device, f_inputs, f_positions, fork_cache(base_cache), split
        )
        r_logits, r_sec = _tail_logits(
            self.model, self.device, r_inputs, r_positions, fork_cache(base_cache), split
        )

        result = {
            **_score_from_logits(f_logits, r_logits, a_id=self.a_id, b_id=self.b_id),
            "visual_prefill_sec": float(audit["visual_prefill_sec"]),
            "query_tail_sec": f_sec + r_sec,
            "visual_token_end": split,
            "forward_total_tokens": int(f_inputs["input_ids"].shape[-1]),
            "reverse_total_tokens": int(r_inputs["input_ids"].shape[-1]),
            "sequential_audit": audit,
        }
        return result

    def score_many(
        self,
        images: list[Any],
        probes: dict[str, tuple[str, str]],
    ) -> dict[str, dict[str, Any]]:
        """
        Score several A/B text probes from one shared visual prefix.

        Every probe must keep the exact same image prefix. The first probe builds
        the canonical visual cache; each forward/reverse text tail then forks
        from that cache, preserving the exact-prefix reuse used by score().
        """
        if not probes:
            return {}

        first_name, (first_forward, _) = next(iter(probes.items()))
        base_cache, split, base_inputs, _, audit = _prefill_visual_prefix(
            self.processor,
            self.model,
            self.device,
            images,
            first_forward,
            thinking=self.thinking,
        )

        results: dict[str, dict[str, Any]] = {}
        shared_prefill_sec = float(audit["visual_prefill_sec"])
        for name, (forward, reverse) in probes.items():
            f_inputs = base_inputs if name == first_name else _encode(
                self.processor, images, forward, thinking=self.thinking
            )
            f_positions = _full_positions(self.model, f_inputs, self.device)
            f_ends = _vision_ends(f_inputs)
            if not f_ends or f_ends[-1] != split:
                raise RuntimeError(
                    f"shared visual split mismatch for {name}: base={split} forward={f_ends[-1] if f_ends else None}"
                )
            _verify_visual_prefix(base_inputs, f_inputs, split)

            r_inputs = _encode(self.processor, images, reverse, thinking=self.thinking)
            r_positions = _full_positions(self.model, r_inputs, self.device)
            r_ends = _vision_ends(r_inputs)
            if not r_ends or r_ends[-1] != split:
                raise RuntimeError(
                    f"shared visual split mismatch for {name}: base={split} reverse={r_ends[-1] if r_ends else None}"
                )
            _verify_visual_prefix(base_inputs, r_inputs, split)

            f_logits, f_sec = _tail_logits(
                self.model, self.device, f_inputs, f_positions, fork_cache(base_cache), split
            )
            r_logits, r_sec = _tail_logits(
                self.model, self.device, r_inputs, r_positions, fork_cache(base_cache), split
            )
            results[name] = {
                **_score_from_logits(f_logits, r_logits, a_id=self.a_id, b_id=self.b_id),
                "visual_prefill_sec": shared_prefill_sec if name == first_name else 0.0,
                "shared_visual_prefill_sec": shared_prefill_sec,
                "query_tail_sec": f_sec + r_sec,
                "visual_token_end": split,
                "forward_total_tokens": int(f_inputs["input_ids"].shape[-1]),
                "reverse_total_tokens": int(r_inputs["input_ids"].shape[-1]),
                "sequential_audit": audit if name == first_name else None,
            }
        return results
