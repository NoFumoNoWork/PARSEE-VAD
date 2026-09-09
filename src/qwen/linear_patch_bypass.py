from __future__ import annotations

import os
from typing import Any


DISABLE_ENV = "QWEN35_DISABLE_LINEAR_PATCH_BYPASS"


def linear_patch_bypass_enabled(config: dict[str, Any] | None = None) -> bool:
    env = os.environ.get(DISABLE_ENV, "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return False
    model_cfg = (config or {}).get("model", {}) if isinstance(config, dict) else {}
    if "linear_patch_bypass" in model_cfg:
        return bool(model_cfg["linear_patch_bypass"])
    if "qwen35_linear_patch_bypass" in model_cfg:
        return bool(model_cfg["qwen35_linear_patch_bypass"])
    return True


def install_qwen35_linear_patch_bypass(model: Any, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "installed": False, "reason": "disabled"}
    try:
        import torch.nn.functional as F

        visual = model.model.visual
        patch_embed = visual.patch_embed
        proj = patch_embed.proj
    except Exception as exc:
        return {"enabled": True, "installed": False, "reason": f"missing_qwen35_patch_embed:{exc.__class__.__name__}"}

    if getattr(patch_embed, "_qwen35_linear_patch_bypass_installed", False):
        return {"enabled": True, "installed": True, "already_installed": True}

    original_forward = patch_embed.forward

    def linear_forward(hidden_states: Any) -> Any:
        patch_embed._qwen35_linear_patch_bypass_calls = getattr(patch_embed, "_qwen35_linear_patch_bypass_calls", 0) + 1
        weight = proj.weight.reshape(proj.out_channels, -1)
        bias = proj.bias
        x = hidden_states.to(dtype=weight.dtype).reshape(-1, weight.shape[1])
        return F.linear(x, weight, bias)

    patch_embed._qwen35_original_patch_embed_forward = original_forward
    patch_embed._qwen35_linear_patch_bypass_installed = True
    patch_embed._qwen35_linear_patch_bypass_calls = 0
    patch_embed.forward = linear_forward
    return {
        "enabled": True,
        "installed": True,
        "module": "model.model.visual.patch_embed",
        "implementation": "F.linear over reshaped Conv3d weights and original bias",
    }


def qwen35_linear_patch_bypass_status(model: Any) -> dict[str, Any]:
    try:
        patch_embed = model.model.visual.patch_embed
    except Exception as exc:
        return {"installed": False, "reason": f"missing_qwen35_patch_embed:{exc.__class__.__name__}"}
    installed = bool(getattr(patch_embed, "_qwen35_linear_patch_bypass_installed", False))
    calls = int(getattr(patch_embed, "_qwen35_linear_patch_bypass_calls", 0))
    return {
        "installed": installed,
        "calls": calls,
        "linear_patch_forward_calls": calls,
        "conv3d_patch_forward_calls": 0 if installed else None,
        "has_original_forward": hasattr(patch_embed, "_qwen35_original_patch_embed_forward"),
    }

