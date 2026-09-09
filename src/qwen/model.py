from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .linear_patch_bypass import install_qwen35_linear_patch_bypass, linear_patch_bypass_enabled

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "qwen35_9b_config.json"


@dataclass
class QwenRuntime:
    processor: Any
    model: Any
    device: Any
    a_id: int
    b_id: int
    answer_token_info: dict[str, Any]


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _read_jsonish(path: Path) -> Any:
    return _expand_env(json.loads(path.read_text(encoding="utf-8-sig")))


def move_inputs(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def inspect_answer_tokens(processor: Any) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)

    def token_id(text: str) -> int:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"expected one token for {text!r}, got {encoded}")
        return int(encoded[0])

    a_id = token_id("A")
    b_id = token_id("B")
    return {
        "A_token_id": a_id,
        "B_token_id": b_id,
        "A_token": tokenizer.convert_ids_to_tokens(a_id) if hasattr(tokenizer, "convert_ids_to_tokens") else "A",
        "B_token": tokenizer.convert_ids_to_tokens(b_id) if hasattr(tokenizer, "convert_ids_to_tokens") else "B",
    }


def load_processor(config_path: Path | None = None, device_id: int | str = 0) -> Any:
    return load_qwen(config_path, device_id).processor


def load_model(config_path: Path | None = None, device_id: int | str = 0) -> tuple[Any, Any, Any]:
    runtime = load_qwen(config_path, device_id)
    return runtime.processor, runtime.model, runtime.device


def load_qwen(config_path: Path | None = None, device_id: int | str = 0) -> QwenRuntime:
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3_5ForConditionalGeneration

        model_cls = Qwen3_5ForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText

    cfg = _read_jsonish(config_path or DEFAULT_MODEL_CONFIG)
    dtype = getattr(torch, str(cfg["model"].get("dtype", "bfloat16")))
    device_idx = int(device_id)
    kwargs = {
        "dtype": dtype,
        "device_map": {"": device_idx},
        "trust_remote_code": bool(cfg["model"].get("trust_remote_code", True)),
        "local_files_only": bool(cfg["model"].get("local_files_only", True)),
        "low_cpu_mem_usage": bool(cfg["model"].get("low_cpu_mem_usage", True)),
    }
    model_path = str(cfg["remote"]["model_path"])
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=kwargs["trust_remote_code"], local_files_only=kwargs["local_files_only"])
    model = model_cls.from_pretrained(model_path, **kwargs).eval()
    model._qwen35_linear_patch_bypass = install_qwen35_linear_patch_bypass(model, linear_patch_bypass_enabled(cfg))
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    answer_token_info = inspect_answer_tokens(processor)
    return QwenRuntime(
        processor=processor,
        model=model,
        device=device,
        a_id=int(answer_token_info["A_token_id"]),
        b_id=int(answer_token_info["B_token_id"]),
        answer_token_info=answer_token_info,
    )
