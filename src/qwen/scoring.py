from __future__ import annotations

import time
from typing import Any, Callable

from .input import apply_messages, image_content, move_inputs


BypassStatusFn = Callable[[Any], dict[str, Any]]


def _torch() -> Any:
    import torch

    return torch


def synchronize(device: Any) -> None:
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def first_logits(
    processor: Any,
    model: Any,
    device: Any,
    images: list[Any],
    prompt: str,
    *,
    thinking: bool = False,
) -> tuple[Any, dict[str, Any], float]:
    torch = _torch()
    t0 = time.perf_counter()
    messages = [{"role": "user", "content": image_content(images, prompt, no_think=not thinking)}]
    inputs = move_inputs(apply_messages(processor, messages, thinking=thinking), device)
    synchronize(device)
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=False, return_dict=True, use_cache=False)
    synchronize(device)
    return outputs.logits[0, -1, :].detach().float().cpu(), inputs, time.perf_counter() - t0


def score_ab(
    processor: Any,
    model: Any,
    device: Any,
    a_id: int,
    b_id: int,
    images: list[Any],
    forward: str,
    reverse: str,
    *,
    thinking: bool = False,
    bypass_status: BypassStatusFn | None = None,
) -> dict[str, Any]:
    before = bypass_status(model) if bypass_status is not None else {}
    f_logits, f_inputs, f_sec = first_logits(processor, model, device, images, forward, thinking=thinking)
    r_logits, r_inputs, r_sec = first_logits(processor, model, device, images, reverse, thinking=thinking)
    after = bypass_status(model) if bypass_status is not None else {}

    linear_delta = 0
    conv_calls = 0
    if bypass_status is not None:
        linear_delta = int(after.get("linear_patch_forward_calls", after.get("calls", 0))) - int(
            before.get("linear_patch_forward_calls", before.get("calls", 0))
        )
        conv_calls = int(after.get("conv3d_patch_forward_calls", 0))
        if not after.get("installed") or linear_delta <= 0 or conv_calls != 0:
            raise RuntimeError(f"linear patch bypass check failed: {after} delta={linear_delta}")

    f_a = float(f_logits[a_id])
    f_b = float(f_logits[b_id])
    r_a = float(r_logits[a_id])
    r_b = float(r_logits[b_id])
    forward_margin = f_b - f_a
    reverse_margin = r_a - r_b
    score = 0.5 * (forward_margin + reverse_margin)
    return {
        "score": score,
        "pred": int(score > 0),
        "forward_margin": forward_margin,
        "reverse_margin": reverse_margin,
        "forward_support": int(forward_margin > 0),
        "reverse_support": int(reverse_margin > 0),
        "support_status": "SUPPORT" if forward_margin > 0 and reverse_margin > 0 else "REJECT",
        "model_sec": f_sec + r_sec,
        "forward_tokens": int(f_inputs["input_ids"].shape[-1]),
        "reverse_tokens": int(r_inputs["input_ids"].shape[-1]),
        "linear_patch_delta": linear_delta,
        "conv3d_patch_calls": conv_calls,
    }
