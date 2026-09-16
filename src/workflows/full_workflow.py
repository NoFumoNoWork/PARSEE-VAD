from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

from PIL import Image

from src.cache.prefix_cache import fork_cache
from src.datasets import build_dataset_adapter
from src.datasets.paths import resolve_video_path
from src.evaluation.metrics import intish
from src.qwen.sequential import (
    _encode,
    _full_positions,
    _prefill_visual_prefix,
    _score_from_logits,
    _tail_logits,
    _verify_visual_prefix,
    _vision_ends,
)
from src.utils.io import write_json
from src.utils.parsing import parse_int_list
from src.utils.video import SequentialVideoFrameReader
from src.workflows.scoring import PARSEEState, PROPOSITION_RUNTIME_VERSION


REPO_ROOT = Path(__file__).resolve().parents[2]
BLANK_SIZE = (224, 126)
ALIGN = 28
PROBE_ORDER = ["q2", "q3", "p3", "p4"]
PROMPT_KEYS = {
    "q2": "q2_native_anomaly",
    "q3": "q3_visible_dynamics",
    "p3": "p3_forceful_physical_interaction",
    "p4": "p4_consequential_object_environment_interaction",
}
FIELDNAMES = [
    "dataset",
    "pixel_budget",
    "category",
    "video_id",
    "anchor",
    "decision_index",
    "gt",
    "q2_score",
    "q2_forward_margin",
    "q2_reverse_margin",
    "q3_score",
    "q3_forward_margin",
    "q3_reverse_margin",
    "p3_score",
    "p3_forward_margin",
    "p3_reverse_margin",
    "p4_score",
    "p4_forward_margin",
    "p4_reverse_margin",
    "p3_executed",
    "p4_executed",
    "p3_route_reason",
    "p4_route_reason",
    # Final current-window dimensionless PAR diagnostics.
    "q3_evidence",
    "p3_evidence",
    "p4_evidence",
    "par_evidence_raw",
    "par_evidence",
    "par_alpha",
    "par_scale",
    "par_delta",
    "par_score",
    # Compatibility aliases used by existing evaluators/viewers.
    "local_score",
    "semantic_score",
    # One-step positive PAR-correction carry diagnostics.
    "carry_enabled",
    "carry_rho",
    "carry_prev_par_delta",
    "carry_prev_positive_delta",
    "carry_budget",
    "carry_rescue_delta",
    "carry_adjusted_par_delta",
    "carry_score",
    "carry_applied",
    "carry_reason",
    "carry_score_minus_q2",
    "carry_score_minus_raw_par",
    # Final two-decision positive-only, non-recursive SEE diagnostics.
    "pre_see_score",
    "see_rho",
    "see_horizon",
    "see_valley_ratio",
    "see_q2_continuity",
    "see_prev1_score",
    "see_prev2_score",
    "see_prev1_positive",
    "see_prev2_positive",
    "see_state_level",
    "see_valley_threshold",
    "see_valley_error",
    "see_q2_prev",
    "see_q2_continuous",
    "see_gate",
    "see_positive_valley",
    "see_negative_valley",
    "see_weighted_history",
    "see_candidate",
    "see_delta",
    "see_applied",
    "see_cross_zero",
    "see_reason",
    "final_score",
    "current_frames",
    "video_path",
    "frame_substitutions",
    "decode_fallback",
    "failure_status",
    "input_image_hw",
    "resized_image_hw",
    "actual_resized_area_mean",
    "actual_resized_area_max",
    "processor_grid_thw",
    "visual_patch_count",
    "visual_token_count",
    "visual_token_end",
    "frame_decode_sec",
    "image_resize_sec",
    "shared_visual_prefill_sec",
    "q2_tail_sec",
    "q3_tail_sec",
    "p3_tail_sec",
    "p4_tail_sec",
    "total_model_time_sec",
    "total_window_time_sec",
]
FAILURE_FIELDS = ["dataset", "pixel_budget", "category", "video_id", "anchor", "decision_index", "error", "video_path"]


def _seq_slice(value: Any, start: int | None = None, end: int | None = None) -> Any:
    if value is None or not hasattr(value, "shape"):
        return value
    return value[..., start:end]


def _repo_path(raw: str | Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _prompt_pair(prompts: dict[str, Any], name: str) -> tuple[str, str]:
    item = prompts[name]
    return str(item["forward"]), str(item["reverse"])


def _video_id(row: dict[str, str]) -> str:
    return row.get("video_id") or row.get("video") or ""


def _category(row: dict[str, str]) -> str:
    category = row.get("category") or ""
    return "Normal" if category in {"Normal_Videos", "Normal"} else category


def _gt(row: dict[str, str], keys: list[str]) -> int:
    if (row.get("label_scope") or "").lower() == "normal" or _category(row) == "Normal":
        return 0
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return intish(value)
    return 0


def _frames(row: dict[str, str], offsets: list[int]) -> list[int]:
    existing = parse_int_list(row.get("current_frames", "") or row.get("frames", ""))
    if existing:
        return existing
    anchor = intish(row.get("anchor", 0))
    return [max(0, anchor + offset) for offset in offsets]


def _manifest_path(spec: dict[str, Any]) -> Path:
    primary = _repo_path(spec["manifest"])
    if primary.exists():
        return primary
    fallback = spec.get("local_fallback_manifest")
    if fallback:
        fallback_path = _repo_path(fallback)
        if fallback_path.exists():
            return fallback_path
    return primary


def _read_rows(config: dict[str, Any], dataset_name: str) -> list[dict[str, str]]:
    spec = config["datasets"][dataset_name]
    path = _manifest_path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"missing manifest for {dataset_name}: {path}. "
            "See docs/DATASETS.md for the public manifest contract."
        )

    if spec.get("adapter") == "raw":
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            source_rows = [dict(row) for row in csv.DictReader(handle)]
        adapter = None
    else:
        adapter = build_dataset_adapter(str(spec.get("adapter", dataset_name)), path)
        source_rows = adapter.rows()

    rows: list[dict[str, str]] = []
    for row in source_rows:
        item = dict(row)
        item["dataset"] = dataset_name
        resolved = resolve_video_path(item, str(spec.get("root_env") or "") or None)
        if (not resolved or str(resolved) == ".") and adapter is not None:
            resolved = adapter.video_path(item)
        if resolved and str(resolved) != ".":
            item["video_path"] = str(resolved)
        rows.append(item)
    return rows


def _shard_rows(rows: list[dict[str, str]], shard_index: int, shard_count: int) -> list[dict[str, str]]:
    video_keys = sorted({(row["dataset"], _video_id(row)) for row in rows})
    mine = {key for idx, key in enumerate(video_keys) if idx % max(shard_count, 1) == shard_index}
    return sorted(
        [row for row in rows if (row["dataset"], _video_id(row)) in mine],
        key=lambda row: (row["dataset"], _video_id(row), intish(row.get("anchor", 0))),
    )


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return {(_video_id(row), intish(row.get("anchor", 0))) for row in csv.DictReader(handle)}


def _completed_scoring_state(path: Path) -> dict[str, list[tuple[int, float, float, float]]]:
    """Read state needed for safe continuation of an interrupted run.

    Each tuple is (anchor, raw_par_delta, pre_see_score, q2).  These values restore
    the one-step correction state and finite-horizon anomaly-state observer.
    Files from older scorer schemas are rejected rather than silently mixed.
    """
    if not path.exists():
        return {}
    by_video: dict[str, list[tuple[int, float, float, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"par_delta", "carry_score", "q2_score"}
        missing = required - fieldnames
        if missing:
            rows = list(reader)
            if rows:
                raise RuntimeError(
                    f"existing score file {path} is missing {sorted(missing)}; "
                    "start a fresh run directory for the correction-carry scorer"
                )
            return {}
        for row in reader:
            raw_delta = _finite_float(row.get("par_delta"))
            pre_see = _finite_float(row.get("carry_score"))
            q2_score = _finite_float(row.get("q2_score"))
            if raw_delta is None or pre_see is None or q2_score is None:
                raise RuntimeError(f"missing/invalid temporal state in existing score file: {path}")
            vid = _video_id(row)
            by_video.setdefault(vid, []).append(
                (intish(row.get("anchor", 0)), raw_delta, pre_see, q2_score)
            )
    for values in by_video.values():
        values.sort(key=lambda item: item[0])
    return by_video

def _failure_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            keys.add((_video_id(row), intish(row.get("anchor", 0))))
    return keys


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _aligned(value: float, minimum: int = ALIGN) -> int:
    return max(minimum, int(round(value / ALIGN)) * ALIGN)


def _resize_one(image: Image.Image, max_pixels: int | None) -> Image.Image:
    rgb = image.convert("RGB")
    if not max_pixels:
        return rgb
    width, height = rgb.size
    area = width * height
    if area <= max_pixels:
        return rgb
    scale = math.sqrt(max_pixels / float(area))
    new_w = min(width, _aligned(width * scale))
    new_h = min(height, _aligned(height * scale))
    while new_w * new_h > max_pixels and (new_w > ALIGN or new_h > ALIGN):
        if new_w >= new_h and new_w > ALIGN:
            new_w -= ALIGN
        elif new_h > ALIGN:
            new_h -= ALIGN
        else:
            break
    if (new_w, new_h) == (width, height):
        return rgb
    return rgb.resize((new_w, new_h), Image.Resampling.BICUBIC)


def _budget_cfg(config: dict[str, Any], budget: str) -> dict[str, Any]:
    return dict((config.get("pixel_budgets", {}) or {})[budget])


def _apply_processor_budget(processor: Any, cfg: dict[str, Any]) -> None:
    if cfg.get("preserve_processor_default"):
        return
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    max_pixels = int(cfg["max_pixels"])
    min_pixels = int(cfg.get("min_pixels", 1) or 1)
    image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels


def _image_grid_stats(inputs: dict[str, Any]) -> dict[str, Any]:
    grid = inputs.get("image_grid_thw")
    if grid is None:
        return {"processor_grid_thw": "", "visual_patch_count": "", "visual_token_count": ""}
    rows = [[int(x) for x in row] for row in grid.detach().cpu().tolist()]
    patches = sum(t * h * w for t, h, w in rows)
    # Qwen VL merges spatial patches by merge_size=2 in the checked processor config.
    tokens = sum(t * h * w // 4 for t, h, w in rows)
    return {
        "processor_grid_thw": json.dumps(rows, separators=(",", ":")),
        "visual_patch_count": patches,
        "visual_token_count": tokens,
    }


def _build_text_tail_cache(
    scorer: Any,
    prompts: dict[str, Any],
    *,
    num_images: int,
) -> dict[str, dict[str, Any]]:
    """Pre-encode fixed textual probe tails once per shard.

    The cached tensors intentionally exclude the visual prefix.  At runtime each
    branch reconstructs the exact full sequence as:

        current visual prefix tokens + cached textual tail

    and recomputes position ids for that sequence.  This preserves the same
    token-tail construction as the older per-probe `_encode()` path while
    avoiding repeated processor/tokenizer work for fixed proposition text.
    """

    blank_images = [Image.new("RGB", BLANK_SIZE, "black") for _ in range(num_images)]
    cache: dict[str, dict[str, Any]] = {}
    expected_split: int | None = None
    canonical_prefix_ids: list[int] | None = None
    canonical_prefix_mm_types: list[int] | None = None
    for probe in PROBE_ORDER:
        forward, reverse = _prompt_pair(prompts, PROMPT_KEYS[probe])
        for direction, prompt in (("forward", forward), ("reverse", reverse)):
            encoded = _encode(scorer.processor, blank_images, prompt, thinking=scorer.thinking)
            ends = _vision_ends(encoded)
            if len(ends) != num_images:
                raise RuntimeError(
                    f"tail-cache build expected {num_images} visual blocks for {probe}/{direction}, "
                    f"found {len(ends)}"
                )
            split = int(ends[-1])
            if expected_split is None:
                expected_split = split
            elif split != expected_split:
                raise RuntimeError(
                    f"tail-cache visual split mismatch: expected {expected_split}, "
                    f"got {split} for {probe}/{direction}"
                )

            prefix_ids = encoded["input_ids"][0, :split].detach().cpu().tolist()
            mm_types = encoded.get("mm_token_type_ids")
            prefix_mm_types = (
                mm_types[0, :split].detach().cpu().tolist() if mm_types is not None else None
            )
            if canonical_prefix_ids is None:
                canonical_prefix_ids = prefix_ids
                canonical_prefix_mm_types = prefix_mm_types
            else:
                if prefix_ids != canonical_prefix_ids:
                    raise RuntimeError(
                        f"tail-cache prefix tokens differ for {probe}/{direction}; "
                        "prompt text is no longer strictly after the shared visual prefix"
                    )
                if prefix_mm_types != canonical_prefix_mm_types:
                    raise RuntimeError(
                        f"tail-cache mm_token_type_ids differ for {probe}/{direction}; "
                        "refusing unsafe cached-tail reuse"
                    )

            cache[f"{probe}_{direction}"] = {
                "split": split,
                "input_ids": _seq_slice(encoded["input_ids"], split, None).detach().cpu(),
                "mm_token_type_ids": (
                    _seq_slice(encoded.get("mm_token_type_ids"), split, None).detach().cpu()
                    if encoded.get("mm_token_type_ids") is not None
                    else None
                ),
            }
    return cache


def _cached_tail_inputs(base_inputs: dict[str, Any], split: int, tail: dict[str, Any]) -> dict[str, Any]:
    # The number of visual placeholder tokens can vary with the processed image
    # grid.  Cached tails deliberately start *after* a synthetic visual prefix
    # and are appended after the current row's real visual prefix.
    torch = __import__("torch")
    full = dict(base_inputs)
    full["input_ids"] = torch.cat([
        _seq_slice(base_inputs["input_ids"], 0, split),
        tail["input_ids"],
    ], dim=-1)
    full["attention_mask"] = torch.ones_like(full["input_ids"])
    if base_inputs.get("mm_token_type_ids") is not None and tail.get("mm_token_type_ids") is not None:
        full["mm_token_type_ids"] = torch.cat([
            _seq_slice(base_inputs["mm_token_type_ids"], 0, split),
            tail["mm_token_type_ids"],
        ], dim=-1)
    else:
        full.pop("mm_token_type_ids", None)
    return full


def _score_probe(
    scorer: Any,
    images: list[Any],
    base_cache: Any,
    split: int,
    base_inputs: dict[str, Any],
    forward: str,
    reverse: str,
) -> dict[str, Any]:
    f_inputs = _encode(scorer.processor, images, forward, thinking=scorer.thinking)
    f_positions = _full_positions(scorer.model, f_inputs, scorer.device)
    f_ends = _vision_ends(f_inputs)
    if not f_ends or f_ends[-1] != split:
        raise RuntimeError(f"visual split mismatch: base={split} forward={f_ends[-1] if f_ends else None}")
    _verify_visual_prefix(base_inputs, f_inputs, split)

    r_inputs = _encode(scorer.processor, images, reverse, thinking=scorer.thinking)
    r_positions = _full_positions(scorer.model, r_inputs, scorer.device)
    r_ends = _vision_ends(r_inputs)
    if not r_ends or r_ends[-1] != split:
        raise RuntimeError(f"visual split mismatch: base={split} reverse={r_ends[-1] if r_ends else None}")
    _verify_visual_prefix(base_inputs, r_inputs, split)

    f_logits, f_sec = _tail_logits(scorer.model, scorer.device, f_inputs, f_positions, fork_cache(base_cache), split)
    r_logits, r_sec = _tail_logits(scorer.model, scorer.device, r_inputs, r_positions, fork_cache(base_cache), split)
    return {**_score_from_logits(f_logits, r_logits, a_id=scorer.a_id, b_id=scorer.b_id), "query_tail_sec": f_sec + r_sec}


def _score_probe_cached_tail(
    scorer: Any,
    base_cache: Any,
    split: int,
    base_inputs: dict[str, Any],
    forward_tail: dict[str, Any],
    reverse_tail: dict[str, Any],
) -> dict[str, Any]:
    f_inputs = _cached_tail_inputs(base_inputs, split, forward_tail)
    f_positions = _full_positions(scorer.model, f_inputs, scorer.device)
    r_inputs = _cached_tail_inputs(base_inputs, split, reverse_tail)
    r_positions = _full_positions(scorer.model, r_inputs, scorer.device)

    f_logits, f_sec = _tail_logits(scorer.model, scorer.device, f_inputs, f_positions, fork_cache(base_cache), split)
    r_logits, r_sec = _tail_logits(scorer.model, scorer.device, r_inputs, r_positions, fork_cache(base_cache), split)
    return {**_score_from_logits(f_logits, r_logits, a_id=scorer.a_id, b_id=scorer.b_id), "query_tail_sec": f_sec + r_sec}


def _use_cached_text_tails(config: dict[str, Any]) -> bool:
    pipeline = config.get("pipeline", {}) or {}
    return bool(pipeline.get("cached_text_tails", True))


def _tail_cache_num_images(config: dict[str, Any]) -> int:
    offsets = [int(x) for x in (config.get("pipeline", {}) or {}).get("current_offsets", [])]
    return len(offsets) if offsets else 9


def _probe_policy(config: dict[str, Any]) -> str:
    policy = str((config.get("pipeline", {}) or {}).get("probe_policy", "routed")).strip().lower()
    if policy not in {"routed", "all_probes", "q2_only"}:
        raise ValueError(f"unknown pipeline.probe_policy={policy!r}; expected routed, all_probes, or q2_only")
    return policy


def _identity_q2_scoring(q2: float) -> dict[str, Any]:
    q2f = float(q2)
    return {
        "q3_evidence": 0.0,
        "p3_evidence": 0.0,
        "p4_evidence": 0.0,
        "par_evidence_raw": 0.0,
        "par_evidence": 0.0,
        "par_alpha": 0.0,
        "par_scale": 1.0,
        "par_delta": 0.0,
        "par_score": q2f,
        "local_score": q2f,
        "semantic_score": q2f,
        "carry_enabled": 0,
        "carry_rho": 0.0,
        "carry_prev_par_delta": 0.0,
        "carry_prev_positive_delta": 0.0,
        "carry_budget": 0.0,
        "carry_rescue_delta": 0.0,
        "carry_adjusted_par_delta": 0.0,
        "carry_score": q2f,
        "carry_applied": 0,
        "carry_reason": "q2_only",
        "carry_score_minus_q2": 0.0,
        "carry_score_minus_raw_par": 0.0,
        "pre_see_score": q2f,
        "see_rho": 0.0,
        "see_horizon": 0,
        "see_valley_ratio": 0.0,
        "see_q2_continuity": 0.0,
        "see_prev1_score": 0.0,
        "see_prev2_score": 0.0,
        "see_prev1_positive": 0.0,
        "see_prev2_positive": 0.0,
        "see_state_level": 0.0,
        "see_valley_threshold": 0.0,
        "see_valley_error": 0.0,
        "see_q2_prev": 0.0,
        "see_q2_continuous": 0,
        "see_gate": 0,
        "see_positive_valley": 0,
        "see_negative_valley": 0,
        "see_weighted_history": 0.0,
        "see_candidate": q2f,
        "see_delta": 0.0,
        "see_applied": 0,
        "see_cross_zero": 0,
        "see_reason": "q2_only",
        "final_score": q2f,
    }


def _blank_scores() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for probe in PROBE_ORDER:
        out[f"{probe}_score"] = ""
        out[f"{probe}_forward_margin"] = ""
        out[f"{probe}_reverse_margin"] = ""
    return out


def _row_output(
    context: Any,
    scorer: Any,
    row: dict[str, str],
    dataset: str,
    budget: str,
    reader: SequentialVideoFrameReader,
    blank: Image.Image,
    text_tail_cache: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, int]]:
    started = time.perf_counter()
    pipeline = context.config["pipeline"]
    offsets = [int(x) for x in pipeline["current_offsets"]]
    frames = _frames(row, offsets)

    t0 = time.perf_counter()
    decoded, substitutions = reader.get(frames)
    decode_sec = time.perf_counter() - t0
    raw_images = [decoded.get(frame, blank) for frame in frames]
    input_hw = [(img.height, img.width) for img in raw_images]

    budget_cfg = _budget_cfg(context.config, budget)
    max_pixels = budget_cfg.get("max_pixels")
    t0 = time.perf_counter()
    images = [_resize_one(img, int(max_pixels) if max_pixels else None) for img in raw_images]
    resize_sec = time.perf_counter() - t0
    resized_hw = [(img.height, img.width) for img in images]
    resized_areas = [h * w for h, w in resized_hw]

    q2_forward, _ = _prompt_pair(context.prompts, PROMPT_KEYS["q2"])
    t0 = time.perf_counter()
    base_cache, split, base_inputs, _, _ = _prefill_visual_prefix(
        scorer.processor,
        scorer.model,
        scorer.device,
        images,
        q2_forward,
        thinking=scorer.thinking,
    )
    prefill_sec = time.perf_counter() - t0
    grid_stats = _image_grid_stats(base_inputs)

    scores = _blank_scores()
    timings = {f"{probe}_tail_sec": 0.0 for probe in PROBE_ORDER}
    counts = {f"{probe}_probe_count": 0 for probe in PROBE_ORDER}
    executed = {probe: 0 for probe in PROBE_ORDER}

    def run_probe(probe: str) -> dict[str, Any]:
        if text_tail_cache is not None:
            result = _score_probe_cached_tail(
                scorer,
                base_cache,
                split,
                base_inputs,
                text_tail_cache[f"{probe}_forward"],
                text_tail_cache[f"{probe}_reverse"],
            )
        else:
            forward, reverse = _prompt_pair(context.prompts, PROMPT_KEYS[probe])
            result = _score_probe(scorer, images, base_cache, split, base_inputs, forward, reverse)
        scores[f"{probe}_score"] = result["score"]
        scores[f"{probe}_forward_margin"] = result["forward_margin"]
        scores[f"{probe}_reverse_margin"] = result["reverse_margin"]
        timings[f"{probe}_tail_sec"] = float(result["query_tail_sec"])
        counts[f"{probe}_probe_count"] = 1
        executed[probe] = 1
        return result

    q2 = run_probe("q2")
    q2_score = float(q2["score"])
    policy = _probe_policy(context.config)
    if policy == "q2_only":
        q3_score = math.nan
        p3_route = False
        p4_route = False
    else:
        q3 = run_probe("q3")
        q3_score = float(q3["score"])

    p3_cfg = pipeline.get("p3_route", {}) or {}
    p3_route = (
        policy == "all_probes"
        or (
            policy == "routed"
            and q2_score > float(p3_cfg.get("q2_gt", 0.0))
            and q3_score >= float(p3_cfg.get("q3_gte", 0.0))
        )
    )
    p3_score = math.nan
    if p3_route:
        p3 = run_probe("p3")
        p3_score = float(p3["score"])

    p4_cfg = pipeline.get("p4_route", {}) or {}
    p4_route = (
        policy == "all_probes"
        or (
            p3_route
            and policy == "routed"
            and q2_score >= float(p4_cfg.get("q2_gte", 1.0))
            and q3_score >= float(p4_cfg.get("q3_gte", 0.0))
            and p3_score <= float(p4_cfg.get("p3_lte", 0.0))
        )
    )
    if p4_route:
        run_probe("p4")

    gt_keys = list(context.config["datasets"][dataset].get("gt_keys", ["gt"]))
    model_sec = prefill_sec + sum(timings.values())
    return {
        "dataset": dataset,
        "pixel_budget": budget,
        "category": _category(row),
        "video_id": _video_id(row),
        "anchor": intish(row.get("anchor", 0)),
        "decision_index": intish(row.get("decision_index", row.get("anchor", 0))),
        "gt": _gt(row, gt_keys),
        **scores,
        "p3_executed": executed["p3"],
        "p4_executed": executed["p4"],
        "p3_route_reason": (
            "force_all_probes" if policy == "all_probes" and p3_route
            else "Q2>0 and Q3>=0" if p3_route
            else "q2_only" if policy == "q2_only"
            else "not_run"
        ),
        "p4_route_reason": (
            "force_all_probes" if policy == "all_probes" and p4_route
            else "Q2>=1 and Q3>=0 and P3<=0" if p4_route
            else "q2_only" if policy == "q2_only"
            else "not_run"
        ),
        "current_frames": ";".join(str(frame) for frame in frames),
        "video_path": row.get("video_path", ""),
        "frame_substitutions": json.dumps(substitutions, ensure_ascii=False, sort_keys=True),
        "decode_fallback": int(any(frame not in decoded for frame in frames) or bool(substitutions)),
        "failure_status": "",
        "input_image_hw": json.dumps(input_hw, separators=(",", ":")),
        "resized_image_hw": json.dumps(resized_hw, separators=(",", ":")),
        "actual_resized_area_mean": sum(resized_areas) / len(resized_areas),
        "actual_resized_area_max": max(resized_areas),
        **grid_stats,
        "visual_token_end": split,
        "frame_decode_sec": decode_sec,
        "image_resize_sec": resize_sec,
        "shared_visual_prefill_sec": prefill_sec,
        **timings,
        "total_model_time_sec": model_sec,
        "total_window_time_sec": time.perf_counter() - started,
    }, {
        "frame_decode_sec": decode_sec,
        "image_resize_sec": resize_sec,
        "shared_visual_prefill_sec": prefill_sec,
        **timings,
        "total_model_time_sec": model_sec,
        "total_window_time_sec": time.perf_counter() - started,
    }, counts


def _status(context: Any, dataset: str, budget: str, status: str, completed: int, total: int, errors: int, last_video: str) -> None:
    shard = f"shard_{context.shard_index:02d}_of_{context.shard_count:02d}"
    path = context.run_dir / "status" / dataset / budget / f"{shard}.json"
    write_json(path, {
        "status": status,
        "dataset": dataset,
        "pixel_budget": budget,
        "shard": shard,
        "completed": completed,
        "total": total,
        "errors": errors,
        "last_video": last_video,
    })


def run(context: Any) -> dict[str, Any]:
    from src.qwen.sequential import SequentialABScorer

    dataset = os.environ.get("PIXEL_BUDGET_DATASET") or str(context.config.get("active_dataset", "ucf"))
    budget = os.environ.get("PIXEL_BUDGET_BUDGET") or str(context.config.get("active_budget", "default"))
    if dataset not in context.config["datasets"]:
        raise KeyError(f"unknown dataset: {dataset}")
    if budget not in context.config["pixel_budgets"]:
        raise KeyError(f"unknown pixel budget: {budget}")

    started = time.perf_counter()
    rows = _shard_rows(_read_rows(context.config, dataset), context.shard_index, context.shard_count)
    shard = f"shard_{context.shard_index:02d}_of_{context.shard_count:02d}"
    root = context.run_dir / "outputs" / dataset / budget / shard
    tables = root / "tables"
    scores_path = tables / "final_workflow_scores.csv"
    failures_path = tables / "final_workflow_failures.csv"
    completed_keys = _completed_keys(scores_path)
    completed_state = _completed_scoring_state(scores_path)
    retry_failures = os.environ.get("PIXEL_BUDGET_RETRY_FAILURES") == "1"
    if retry_failures:
        raise RuntimeError(
            "PIXEL_BUDGET_RETRY_FAILURES=1 is unsafe for stateful correction carry / SEE because "
            "windows after a failure may have been scored with reset temporal state. "
            "Rerun the affected video/shard in a fresh output directory instead."
        )
    todo = [row for row in rows if (_video_id(row), intish(row.get("anchor", 0))) not in completed_keys]
    completed = len(completed_keys)
    errors = 0
    buffer: list[dict[str, Any]] = []
    timings = {name: 0.0 for name in [
        "frame_decode_sec", "image_resize_sec", "shared_visual_prefill_sec",
        "q2_tail_sec", "q3_tail_sec", "p3_tail_sec", "p4_tail_sec",
        "total_model_time_sec", "total_window_time_sec",
    ]}
    counts = {f"{probe}_probe_count": 0 for probe in PROBE_ORDER}
    counts.update({
        "p3_positive_evidence_count": 0,
        "p4_positive_evidence_count": 0,
        "correction_carry_applied_count": 0,
        "see_applied_count": 0,
        "see_cross_zero_count": 0,
    })
    policy = _probe_policy(context.config)
    # Scoring state is cheap to construct and does not load the MLLM runtime.
    # Keeping it here also makes the paper-final scoring parameters available in the
    # summary when a shard is already complete.
    parsee = PARSEEState.from_pipeline(context.config["pipeline"])
    if not todo:
        _status(context, dataset, budget, "COMPLETE", completed, len(rows), errors, "")
        summary_path = root / "summary.json"
        existing_summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
                if isinstance(loaded, dict):
                    existing_summary = loaded
            except Exception:
                existing_summary = {}
        summary = {
            "status": "COMPLETE",
            "dataset": dataset,
            "pixel_budget": budget,
            "shard": shard,
            "completed": completed,
            "total": len(rows),
            "errors": errors,
            "elapsed_sec": time.perf_counter() - started,
            "scores_csv": str(scores_path),
            "failures_csv": str(failures_path),
            "scoring_rule": "PARSEE-VAD: PAR routing -> memoryless current-window fusion -> one-step correction memory -> finite-horizon SEE",
            "probe_policy": policy,
            "proposition_runtime": existing_summary.get(
                "proposition_runtime", "unknown_existing_completed_output"
            ),
            "text_tail_cache_build_sec": existing_summary.get("text_tail_cache_build_sec", None),
            "par_alpha": float(parsee.config.par.alpha),
            "correction_carry_rho": float(parsee.config.correction_carry.rho),
            "see_rho": float(parsee.config.see.rho),
            "see_horizon": int(parsee.config.see.horizon),
            "resume_note": "no pending windows; model runtime was not loaded",
            **timings,
            **counts,
        }
        write_json(summary_path, summary)
        return summary

    runtime = context.scorer._runtime()
    _apply_processor_budget(runtime.processor, _budget_cfg(context.config, budget))
    scorer = SequentialABScorer(
        runtime.processor,
        runtime.model,
        runtime.device,
        runtime.a_id,
        runtime.b_id,
        thinking=bool((context.config.get("model", {}) or {}).get("thinking", False)),
    )
    text_tail_cache = None
    tail_cache_sec = 0.0
    if _use_cached_text_tails(context.config):
        t0 = time.perf_counter()
        text_tail_cache = _build_text_tail_cache(
            scorer,
            context.prompts,
            num_images=_tail_cache_num_images(context.config),
        )
        tail_cache_sec = time.perf_counter() - t0
    append_every = int((context.config.get("pipeline", {}) or {}).get("append_every", 8) or 8)
    status_every = int((context.config.get("pipeline", {}) or {}).get("status_every_anchors", 8) or 8)

    current_video = None
    reader: SequentialVideoFrameReader | None = None
    blank = Image.new("RGB", BLANK_SIZE, "black")
    _status(context, dataset, budget, "RUNNING", completed, len(rows), errors, "")

    try:
        for idx, row in enumerate(todo, start=1):
            try:
                if _video_id(row) != current_video:
                    if reader is not None:
                        reader.close()
                    current_video = _video_id(row)
                    reader = SequentialVideoFrameReader(Path(row["video_path"]))
                    parsee.reset()
                    # Safe interrupted-run resume:
                    #   1) restore the immediately previous RAW PAR delta for
                    #      one-step correction carry;
                    #   2) restore the last SEE-horizon PRE-SEE scores, which
                    #      are carry_score values (never final_score).
                    current_anchor = intish(row.get("anchor", 0))
                    prior = [
                        item for item in completed_state.get(current_video, [])
                        if item[0] < current_anchor
                    ]
                    if prior:
                        pre_see_scores = [item[2] for item in prior]
                        q2_values = [item[3] for item in prior]
                        previous_raw_par_delta = prior[-1][1]
                        parsee.seed_history(
                            pre_see_scores,
                            previous_raw_par_delta=previous_raw_par_delta,
                            q2_values=q2_values,
                        )
                assert reader is not None
                output, row_timings, row_counts = _row_output(
                    context,
                    scorer,
                    row,
                    dataset,
                    budget,
                    reader,
                    blank,
                    text_tail_cache=text_tail_cache,
                )
                if policy == "q2_only":
                    scored = _identity_q2_scoring(float(output["q2_score"]))
                else:
                    scored = parsee.step(
                        q2=float(output["q2_score"]),
                        q3=float(output["q3_score"]),
                        p3=output.get("p3_score"),
                        p4=output.get("p4_score"),
                        p3_executed=bool(intish(output.get("p3_executed", 0))),
                        p4_executed=bool(intish(output.get("p4_executed", 0))),
                    )
                output.update(scored)
                if scored["p3_evidence"] > 0:
                    counts["p3_positive_evidence_count"] += 1
                if scored["p4_evidence"] > 0:
                    counts["p4_positive_evidence_count"] += 1
                counts["correction_carry_applied_count"] += int(scored["carry_applied"])
                counts["see_applied_count"] += int(scored["see_applied"])
                counts["see_cross_zero_count"] += int(scored["see_cross_zero"])
                buffer.append(output)
                completed += 1
                for key, value in row_timings.items():
                    timings[key] += float(value)
                for key, value in row_counts.items():
                    counts[key] = counts.get(key, 0) + int(value)
                if len(buffer) >= append_every:
                    _append_csv(scores_path, FIELDNAMES, buffer)
                    buffer.clear()
            except Exception as exc:
                errors += 1
                parsee.reset()
                _append_csv(failures_path, FAILURE_FIELDS, [{
                    "dataset": dataset,
                    "pixel_budget": budget,
                    "category": _category(row),
                    "video_id": _video_id(row),
                    "anchor": intish(row.get("anchor", 0)),
                    "decision_index": intish(row.get("decision_index", row.get("anchor", 0))),
                    "error": str(exc),
                    "video_path": row.get("video_path", ""),
                }])
            if idx == len(todo) or idx % status_every == 0:
                if buffer:
                    _append_csv(scores_path, FIELDNAMES, buffer)
                    buffer.clear()
                _status(context, dataset, budget, "RUNNING", completed, len(rows), errors, _video_id(row))
    finally:
        if reader is not None:
            reader.close()
    if buffer:
        _append_csv(scores_path, FIELDNAMES, buffer)

    final_status = "COMPLETE" if errors == 0 else "COMPLETE_WITH_ERRORS"
    _status(context, dataset, budget, final_status, completed, len(rows), errors, current_video or "")
    summary = {
        "status": final_status,
        "dataset": dataset,
        "pixel_budget": budget,
        "shard": shard,
        "completed": completed,
        "total": len(rows),
        "errors": errors,
        "elapsed_sec": time.perf_counter() - started,
        "scores_csv": str(scores_path),
        "failures_csv": str(failures_path),
        "scoring_rule": "PARSEE-VAD: PAR routing -> memoryless current-window fusion -> one-step correction memory -> finite-horizon SEE",
        "probe_policy": policy,
        "proposition_runtime": PROPOSITION_RUNTIME_VERSION if text_tail_cache is not None else "full_multimodal_encode_per_probe",
        "text_tail_cache_build_sec": tail_cache_sec,
        "par_alpha": float(parsee.config.par.alpha),
        "correction_carry_rho": float(parsee.config.correction_carry.rho),
        "see_rho": float(parsee.config.see.rho),
        "see_horizon": int(parsee.config.see.horizon),
        **timings,
        **counts,
    }
    write_json(root / "summary.json", summary)
    return summary
