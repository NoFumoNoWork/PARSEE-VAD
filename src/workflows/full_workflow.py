from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
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
    "p3_positive_delta",
    "p4_positive_delta",
    "p4_strong_negative_delta",
    "semantic_score",
    "prop_state_before",
    "prop_recovery_raw",
    "prop_recovery_candidate",
    "prop_cross_zero",
    "prop_positive_envelope",
    "prop_reason",
    "prop_state_after",
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


@dataclass
class ControlledPropagationState:
    delta: float = 0.95
    beta: float = 1.5
    positive_gamma: float = 0.7
    cross_zero_history: int = 3
    cross_zero_prev_min: float = 0.5
    cross_zero_current_floor: float = -1.0
    cross_zero_cap: float = 0.25
    state_reset_below: float = 0.05
    negative_ceiling: float = -1e-6
    evidence: float = 0.0
    previous_final: float = 0.0
    semantic_history: list[float] = field(default_factory=list)

    @classmethod
    def from_config(cls, pipeline: dict[str, Any]) -> "ControlledPropagationState":
        cfg = pipeline.get("propagation", {}) or {}
        return cls(**{field_name: type(getattr(cls(), field_name))(cfg.get(field_name, getattr(cls(), field_name))) for field_name in (
            "delta", "beta", "positive_gamma", "cross_zero_history", "cross_zero_prev_min",
            "cross_zero_current_floor", "cross_zero_cap", "state_reset_below", "negative_ceiling"
        )})

    def reset(self) -> None:
        self.evidence = 0.0
        self.previous_final = 0.0
        self.semantic_history.clear()

    def step(self, semantic_score: float) -> dict[str, Any]:
        s = float(semantic_score)
        state_before = float(self.evidence)
        recovery_raw = s
        recovery_candidate = 0
        cross_zero = 0
        positive_envelope = 0
        reason = "raw"
        if s >= 0.0:
            if self.previous_final > 0.0 and self.positive_gamma * self.previous_final > s:
                final = self.positive_gamma * self.previous_final
                positive_envelope = 1
                reason = "positive_run_envelope"
            else:
                final = s
                reason = "positive_raw"
        else:
            recovery_raw = s + self.beta * math.tanh(state_before)
            recent = self.semantic_history[-self.cross_zero_history:] if self.cross_zero_history > 0 else []
            recent_strong = len(recent) == self.cross_zero_history and min(recent) >= self.cross_zero_prev_min
            if recent_strong and s >= self.cross_zero_current_floor and recovery_raw > 0.0:
                recovery_candidate = 1
                cross_zero = 1
                final = min(self.cross_zero_cap, recovery_raw)
                reason = "confirmed_shallow_dip_cross_zero"
            else:
                final = min(self.negative_ceiling, recovery_raw)
                if recovery_raw > s:
                    recovery_candidate = 1
                    reason = "negative_valley_recovery_sign_preserved"
                else:
                    reason = "negative_raw"
        self.evidence = max(self.delta * state_before, max(s, 0.0))
        if self.evidence < self.state_reset_below:
            self.evidence = 0.0
        self.semantic_history.append(s)
        self.semantic_history = self.semantic_history[-max(self.cross_zero_history, 1):]
        self.previous_final = float(final)
        return {
            "prop_state_before": state_before,
            "prop_recovery_raw": recovery_raw,
            "prop_recovery_candidate": recovery_candidate,
            "prop_cross_zero": cross_zero,
            "prop_positive_envelope": positive_envelope,
            "prop_reason": reason,
            "prop_state_after": float(self.evidence),
            "final_score": float(final),
        }


def _semantic(q2: float, p3: Any, p4: Any) -> dict[str, float]:
    p3f = _finite_float(p3)
    p4f = _finite_float(p4)
    p3_delta = math.tanh(p3f) if p3f is not None and p3f > 0.0 else 0.0
    p4_delta = math.tanh(p4f) if p4f is not None and p4f > 0.0 else 0.0
    return {
        "p3_positive_delta": p3_delta,
        "p4_positive_delta": p4_delta,
        "p4_strong_negative_delta": 0.0,
        "semantic_score": float(q2) + p3_delta + p4_delta,
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
    q3 = run_probe("q3")
    q2_score = float(q2["score"])
    q3_score = float(q3["score"])

    p3_cfg = pipeline.get("p3_route", {}) or {}
    p3_route = q2_score > float(p3_cfg.get("q2_gt", 0.0)) and q3_score >= float(p3_cfg.get("q3_gte", 0.0))
    p3_score = math.nan
    if p3_route:
        p3 = run_probe("p3")
        p3_score = float(p3["score"])

    p4_cfg = pipeline.get("p4_route", {}) or {}
    p4_route = (
        p3_route
        and q2_score >= float(p4_cfg.get("q2_gte", 1.0))
        and q3_score >= float(p4_cfg.get("q3_gte", 0.0))
        and p3_score <= float(p4_cfg.get("p3_lte", 0.0))
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
        "p3_route_reason": "Q2>0 and Q3>=0" if p3_route else "not_run",
        "p4_route_reason": "Q2>=1 and Q3>=0 and P3<=0" if p4_route else "not_run",
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
    retry_failures = os.environ.get("PIXEL_BUDGET_RETRY_FAILURES") == "1"
    retry_keys = _failure_keys(failures_path) if retry_failures else set()

    if retry_failures:
        todo = [
            row for row in rows
            if (_video_id(row), intish(row.get("anchor", 0))) in retry_keys
            and (_video_id(row), intish(row.get("anchor", 0))) not in completed_keys
        ]
    else:
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
        "p3_positive_count": 0,
        "p4_positive_count": 0,
        "prop_cross_zero_count": 0,
    })
    if not todo:
        _status(context, dataset, budget, "COMPLETE", completed, len(rows), errors, "")
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
            "semantic_rule": "Q2 + tanh(P3) if P3>0 + tanh(P4) if P4>0; P4<=0 ignored",
            "resume_note": "no pending windows; model runtime was not loaded",
            **timings,
            **counts,
        }
        write_json(root / "summary.json", summary)
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
    append_every = int((context.config.get("pipeline", {}) or {}).get("append_every", 8) or 8)
    status_every = int((context.config.get("pipeline", {}) or {}).get("status_every_anchors", 8) or 8)

    current_video = None
    reader: SequentialVideoFrameReader | None = None
    prop = ControlledPropagationState.from_config(context.config["pipeline"])
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
                    prop.reset()
                assert reader is not None
                output, row_timings, row_counts = _row_output(context, scorer, row, dataset, budget, reader, blank)
                semantic = _semantic(float(output["q2_score"]), output.get("p3_score"), output.get("p4_score"))
                output.update(semantic)
                propagated = prop.step(float(semantic["semantic_score"]))
                output.update(propagated)
                if semantic["p3_positive_delta"] > 0:
                    counts["p3_positive_count"] += 1
                if semantic["p4_positive_delta"] > 0:
                    counts["p4_positive_count"] += 1
                counts["prop_cross_zero_count"] += int(propagated["prop_cross_zero"])
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
                prop.reset()
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
        "semantic_rule": "Q2 + tanh(P3) if P3>0 + tanh(P4) if P4>0; P4<=0 ignored",
        **timings,
        **counts,
    }
    write_json(root / "summary.json", summary)
    return summary
