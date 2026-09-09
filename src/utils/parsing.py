from __future__ import annotations

from typing import Any

from src.evaluation.metrics import intish


def parse_int_list(text: Any) -> list[int]:
    return [intish(x) for x in str(text).replace(",", ";").split(";") if str(x).strip()]
