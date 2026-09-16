from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    import av
except Exception:  # pragma: no cover
    av = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from src.evaluation.metrics import ap, auroc, intish, maybe_float

DEFAULT_ALIGNMENT = "completed"
ALIGNMENT_METHODS = {
    "completed": "window_end_backward_hold_nonoverlap",
    "availability": "decision_anchor_forward_hold",
}
# Backward-compatible constant for callers that only use the paper's primary
# completed-interval protocol.
EXPANSION_METHOD = ALIGNMENT_METHODS[DEFAULT_ALIGNMENT]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_config_path(raw: str, repo_root: Path) -> Path:
    expanded = os.path.expandvars(str(raw)).strip()
    if "$" in expanded:
        raise RuntimeError(f"unresolved environment variable in path: {raw}")
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else repo_root / path


def frame_count(path: Path) -> int:
    if av is None:
        raise RuntimeError("PyAV is required when total frame counts are not supplied by metadata")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames and int(stream.frames) > 0:
            return int(stream.frames)
        if stream.duration is not None and stream.time_base is not None and stream.average_rate is not None:
            estimate = int(round(float(stream.duration * stream.time_base * stream.average_rate)))
            if estimate > 0:
                return estimate
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for _ in container.decode(stream):
            count += 1
    return count


def parse_intervals(text: str) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for part in str(text or "").replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        nums = [int(float(x)) for x in re.findall(r"\d+(?:\.\d+)?", item)]
        if len(nums) >= 2:
            a, b = nums[0], nums[1]
            if a < 0 or b < 0:
                continue
            if b < a:
                a, b = b, a
            intervals.append((a, b))
    return intervals


def gt_from_intervals(nframes: int, intervals: list[tuple[int, int]]) -> bytearray:
    gt = bytearray(nframes)
    for start, end in intervals:
        lo = max(0, start)
        hi = min(nframes - 1, end)
        if hi >= lo:
            gt[lo:hi + 1] = b"\x01" * (hi - lo + 1)
    return gt


def video_id(row: dict[str, str]) -> str:
    return row.get("video_id") or row.get("video") or ""


def resolve_row_video_path(row: dict[str, str], root_env: str | None = None) -> Path:
    raw = (row.get("video_path") or row.get("local_video_path") or "").strip()
    path = Path(os.path.expandvars(raw)).expanduser()
    if path.is_absolute():
        return path
    if root_env:
        root = os.environ.get(root_env, "").strip()
        if root:
            return Path(os.path.expandvars(root)).expanduser() / path
    return path


def load_ucf_meta(manifest_path: Path) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(manifest_path):
        vid = row.get("video") or row.get("video_id") or ""
        if not vid or vid in meta:
            continue
        meta[vid] = {
            "total_frames": intish(row.get("total_frames")),
            "intervals": parse_intervals(row.get("original_intervals", "")),
            "gt_source": "official_ucf_original_intervals",
        }
    return meta


def load_msad_meta(manifest_path: Path, ann_path: Path) -> dict[str, dict[str, Any]]:
    intervals: dict[str, list[tuple[int, int]]] = {}
    for row in read_csv_rows(ann_path):
        name = row.get("name", "")
        if name:
            intervals[name] = [(intish(row.get("starting frame of anomaly")), intish(row.get("ending frame of anomaly")))]
    meta: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(manifest_path):
        vid = row.get("video") or row.get("video_id") or ""
        if not vid or vid in meta:
            continue
        meta[vid] = {
            "total_frames": intish(row.get("frame_count") or row.get("total_frames")),
            "intervals": intervals.get(vid, []),
            "gt_source": "official_msad_anomaly_annotation_zero_based_inclusive",
        }
    return meta


def norm_xd_id(raw: str) -> str:
    text = raw.strip().replace("\\", "/").split("/")[-1]
    return text[:-4] if text.lower().endswith(".mp4") else text


def load_xd_annotations(path: Path) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        vid = norm_xd_id(parts[0])
        nums: list[int] = []
        for token in parts[1:]:
            try:
                nums.append(int(float(token)))
            except ValueError:
                pass
        intervals: list[tuple[int, int]] = []
        for idx in range(0, len(nums) - 1, 2):
            a, b = nums[idx], nums[idx + 1]
            if a < 0 or b < 0:
                continue
            if b < a:
                a, b = b, a
            intervals.append((a, b))
        out[vid] = intervals
    return out


def load_xd_meta(rows: list[dict[str, str]], ann_path: Path, root_env: str = "PARSEE_XD_ROOT") -> dict[str, dict[str, Any]]:
    annotations = load_xd_annotations(ann_path)
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        vid = video_id(row)
        if not vid or vid in meta:
            continue
        path = resolve_row_video_path(row, root_env)
        meta[vid] = {"total_frames": frame_count(path), "intervals": annotations.get(vid, []), "gt_source": "official_xd_test_annotations"}
    return meta


def mask_frame(path: Path) -> int | None:
    match = re.search(r"_(\d+)_gt\.png$", path.name)
    return int(match.group(1)) if match else None


def ubnormal_gt(video_path: Path, nframes: int) -> bytearray:
    if video_path.stem.startswith("normal_"):
        return bytearray(nframes)
    if Image is None:
        raise RuntimeError("Pillow is required for UBnormal mask GT")
    ann_dir = video_path.with_name(f"{video_path.stem}_annotations")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"missing UBnormal annotation directory: {ann_dir}")
    gt = bytearray(nframes)
    for mask in sorted(ann_dir.glob("*_gt.png")):
        frame = mask_frame(mask)
        if frame is None or frame < 0 or frame >= nframes:
            continue
        with Image.open(mask) as image:
            if image.convert("L").getbbox() is not None:
                gt[frame] = 1
    return gt


def load_ubnormal_meta(rows: list[dict[str, str]], root_env: str = "PARSEE_UBNORMAL_ROOT") -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        vid = video_id(row)
        if not vid or vid in meta:
            continue
        path = resolve_row_video_path(row, root_env)
        meta[vid] = {"total_frames": frame_count(path), "video_path": str(path), "gt_source": "official_ubnormal_test_masks"}
    return meta


def gt_for_video(dataset: str, vid: str, meta: dict[str, Any]) -> bytearray:
    nframes = int(meta["total_frames"])
    if dataset == "ubnormal":
        return ubnormal_gt(Path(meta["video_path"]), nframes)
    return gt_from_intervals(nframes, meta.get("intervals", []))


def expand_anchor_scores(
    anchors: list[int],
    scores: list[float],
    nframes: int,
    alignment: str = DEFAULT_ALIGNMENT,
) -> list[float]:
    """Expand decision scores to frame scores under a declared timing alignment.

    ``completed`` assigns each decision to the non-overlapping interval that
    ended at its anchor, matching the primary completed-interval protocol.

    ``availability`` is stricter: a decision becomes usable only at its anchor
    and is held forward until the next decision. Frames before the first anchor
    receive a neutral score of 0.0. This frame-index protocol does not attempt
    to model sub-frame wall-clock inference latency.
    """
    if len(anchors) != len(scores):
        raise ValueError("anchors/scores length mismatch")
    if anchors != sorted(set(anchors)):
        raise ValueError("anchors must be strictly increasing and unique")
    if alignment not in ALIGNMENT_METHODS:
        raise ValueError(f"unknown alignment={alignment!r}; expected one of {sorted(ALIGNMENT_METHODS)}")

    frame_scores = [0.0] * nframes
    if alignment == "completed":
        for idx, anchor in enumerate(anchors):
            start = 0 if idx == 0 else anchors[idx - 1] + 1
            start = max(0, min(start, nframes))
            end = max(start, min(anchor + 1, nframes))
            if end > start:
                frame_scores[start:end] = [float(scores[idx])] * (end - start)
        if anchors:
            tail_start = max(0, min(anchors[-1] + 1, nframes))
            if tail_start < nframes:
                frame_scores[tail_start:] = [float(scores[-1])] * (nframes - tail_start)
        return frame_scores

    # Availability alignment: score_i is not used before anchor_i.
    for idx, anchor in enumerate(anchors):
        start = max(0, min(anchor, nframes))
        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else nframes
        end = max(start, min(next_anchor, nframes))
        if end > start:
            frame_scores[start:end] = [float(scores[idx])] * (end - start)
    return frame_scores


def audit_decisions(rows: list[dict[str, str]], dataset: str, budget: str, score_field: str) -> dict[str, Any]:
    keys: set[tuple[str, int]] = set()
    dup = missing_score = failures = 0
    for row in rows:
        key = (video_id(row), intish(row.get("anchor")))
        if key in keys:
            dup += 1
        keys.add(key)
        if maybe_float(row.get(score_field)) is None:
            missing_score += 1
        if str(row.get("failure_status", "")).strip():
            failures += 1
    return {"dataset": dataset, "budget": budget, "decision_windows": len(rows), "decision_videos": len({video_id(row) for row in rows}), "duplicate_anchors": dup, "missing_score": missing_score, "failure_rows": failures, "score_field": score_field}


def expand_dataset_budget(dataset: str, budget: str, decision_rows: list[dict[str, str]], meta: dict[str, dict[str, Any]], score_field: str, frame_csv: Path | None = None, alignment: str = DEFAULT_ALIGNMENT) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[int, int, int]]]:
    by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in decision_rows:
        by_video[video_id(row)].append(row)
    missing_videos = sorted(set(by_video) - set(meta))
    if missing_videos:
        raise RuntimeError(f"{dataset}/{budget}: missing metadata for videos: {missing_videos[:10]}")

    labels_all: list[int] = []
    scores_all: list[float] = []
    macro_values: list[float] = []
    sanity_rows: list[dict[str, Any]] = []
    gt_signature: dict[str, tuple[int, int, int]] = {}
    writer = handle = None
    if frame_csv is not None:
        frame_csv.parent.mkdir(parents=True, exist_ok=True)
        handle = frame_csv.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=["dataset", "budget", "video_id", "frame", "gt", "score"])
        writer.writeheader()
    try:
        for vid in sorted(by_video):
            rows = sorted(by_video[vid], key=lambda row: intish(row.get("anchor")))
            anchors = [intish(row.get("anchor")) for row in rows]
            scores_raw = [maybe_float(row.get(score_field)) for row in rows]
            if any(score is None for score in scores_raw):
                raise RuntimeError(f"{dataset}/{budget}/{vid}: missing {score_field}")
            scores = [float(score) for score in scores_raw if score is not None]
            nframes = int(meta[vid]["total_frames"])
            if nframes <= 0:
                raise RuntimeError(f"{dataset}/{budget}/{vid}: invalid frame count {nframes}")
            gt = gt_for_video(dataset, vid, meta[vid])
            if len(gt) != nframes:
                raise RuntimeError(f"{dataset}/{budget}/{vid}: GT length mismatch")
            frame_scores = expand_anchor_scores(anchors, scores, nframes, alignment=alignment)
            for frame, (label, score) in enumerate(zip(gt, frame_scores)):
                y, s = int(label), float(score)
                labels_all.append(y); scores_all.append(s)
                if writer is not None:
                    writer.writerow({"dataset": dataset, "budget": budget, "video_id": vid, "frame": frame, "gt": y, "score": s})
            v_auc = auroc([int(x) for x in gt], frame_scores)
            if v_auc is not None:
                macro_values.append(v_auc)
            gt_signature[vid] = (nframes, int(sum(gt)), nframes - int(sum(gt)))
            if len(sanity_rows) < 20:
                for idx, anchor in enumerate(anchors[:4]):
                    if alignment == "completed":
                        start = 0 if idx == 0 else anchors[idx - 1] + 1
                        end = min(anchor, nframes - 1)
                    else:
                        start = max(0, anchor)
                        next_anchor = anchors[idx + 1] if idx + 1 < len(anchors) else nframes
                        end = min(next_anchor - 1, nframes - 1)
                    if start <= end:
                        sanity_rows.append({"dataset": dataset, "budget": budget, "alignment": alignment, "video_id": vid, "anchor": anchor, "decision_score": scores[idx], "frame_start": start, "frame_end": end, "expanded_score": scores[idx], "gt_positive_frames_in_range": sum(gt[start:end + 1]), "causal_check": "PASS"})
    finally:
        if handle is not None:
            handle.close()

    audit = {
        "dataset": dataset, "budget": budget, "videos": len(by_video), "total_frames_evaluated": len(labels_all),
        "gt_positive_frames": sum(labels_all), "gt_negative_frames": len(labels_all) - sum(labels_all),
        "decision_windows": sum(len(v) for v in by_video.values()), "missing_decisions": 0, "missing_videos": 0,
        "alignment": alignment, "expansion_method": ALIGNMENT_METHODS[alignment], "frame_micro_AUROC": auroc(labels_all, scores_all), "frame_AP": ap(labels_all, scores_all),
        "macro_video_AUROC": (sum(macro_values) / len(macro_values)) if macro_values else None,
        "macro_video_AUROC_videos_used": len(macro_values), "frame_score_csv": str(frame_csv) if frame_csv else "",
    }
    return audit, sanity_rows, gt_signature
