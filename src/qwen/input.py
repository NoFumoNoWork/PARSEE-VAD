from __future__ import annotations

from typing import Any

from .model import move_inputs


def image_content(images: list[Any], prompt: str, *, no_think: bool = True) -> list[dict[str, Any]]:
    text = prompt.strip()
    if no_think:
        text = text + "\n\n/no_think"
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    return content


def apply_messages(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    thinking: bool = False,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    return_dict: bool = True,
    return_tensors: str = "pt",
) -> dict[str, Any]:
    kwargs = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": return_dict,
        "return_tensors": return_tensors,
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)

