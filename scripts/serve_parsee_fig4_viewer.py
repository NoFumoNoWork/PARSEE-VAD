#!/usr/bin/env python
"""PARSEE-VAD multi-dataset qualitative viewer.

The viewer reads the official score-only replay CSVs directly.  The current
CSV schema is Q2 -> PAR -> SEE final.  Routing remains Q2/Q3-gated exactly as
in the original full workflow, and no model inference is performed here.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# When this script is executed as ``python scripts/serve_parsee_fig4_viewer.py``,
# Python places ``scripts/`` rather than the repository root on sys.path.
# Add the repository root before importing project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



# Public release defaults come only from environment variables or explicit CLI
# arguments. No author-specific machine paths are embedded in the repository.

def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


PORT = 8780
DEFAULT_MSAD_ANOMALY_ROOT = _env_path("PARSEE_MSAD_ANOMALY_ROOT")
DEFAULT_MSAD_NORMAL_ROOT = _env_path("PARSEE_MSAD_NORMAL_ROOT")
DEFAULT_MSAD_ANNOTATION = _env_path("PARSEE_MSAD_ANNOTATION_CSV")
DEFAULT_UCF_ROOT = _env_path("PARSEE_UCF_ROOT")
DEFAULT_UCF_ANNOTATION = _env_path("PARSEE_UCF_ANNOTATION_TXT")


CATEGORY_PART = {
    "Abuse": "Anomaly-Videos-Part-1",
    "Arrest": "Anomaly-Videos-Part-1",
    "Arson": "Anomaly-Videos-Part-1",
    "Assault": "Anomaly-Videos-Part-1",
    "Burglary": "Anomaly-Videos-Part-2",
    "Explosion": "Anomaly-Videos-Part-2",
    "Fighting": "Anomaly-Videos-Part-2",
    "RoadAccidents": "Anomaly-Videos-Part-3",
    "Robbery": "Anomaly-Videos-Part-3",
    "Shooting": "Anomaly-Videos-Part-3",
    "Shoplifting": "Anomaly-Videos-Part-4",
    "Stealing": "Anomaly-Videos-Part-4",
    "Vandalism": "Anomaly-Videos-Part-4",
}


@dataclass
class AppState:
    dataset: str
    rows_by_video: dict[str, list[dict[str, Any]]]
    category_by_video: dict[str, str]
    categories: list[str]
    gt_intervals: dict[str, list[list[int]]]
    ucf_root: Path | None
    msad_anomaly_root: Path | None
    msad_normal_root: Path | None
    video_paths: dict[str, Path]
    score_csvs: list[Path]
    title: str
    gate_audit: dict[str, Any]


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def finite_or_none(value: Any) -> float | None:
    number = to_float(value)
    return number if math.isfinite(number) else None


def parse_ints(value: Any) -> list[int]:
    return [int(part) for part in re.findall(r"-?\d+", str(value or ""))]


def has_official_replay_header(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = set(next(reader, []))
    except Exception:
        return False
    required = {
        "video_id", "category", "anchor",
        "q2_score", "q3_score", "p3_score", "p4_score",
        "par_score", "final_score",
    }
    return required <= header


def resolve_dataset_score_csvs(
    explicit: list[Path] | None,
    dataset: str,
) -> list[Path]:
    paths = [path.resolve() for path in explicit] if explicit else []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if not has_official_replay_header(path):
            raise ValueError(
                f"{path} is not an official PARSEE-VAD window replay table; "
                "required columns include Q2/Q3/P3/P4, par_score, and final_score."
            )
    return paths


def load_workflow_rows(
    paths: list[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], list[str]]:
    """Load canonical PARSEE-VAD window replay rows without recomputation."""
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    required_columns = {
        "video_id", "category", "anchor",
        "q2_score", "q3_score", "p3_score", "p4_score",
        "par_score", "final_score",
    }

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            missing = required_columns - fields
            if missing:
                raise ValueError(f"{path}: missing official replay columns {sorted(missing)}")

            for raw in reader:
                if str(raw.get("failure_status", "") or "").strip():
                    continue

                video_id = str(raw.get("video_id", "")).strip()
                category = str(raw.get("category", "")).strip()
                anchor = to_int(raw.get("anchor"), -1)
                q2 = finite_or_none(raw.get("q2_score"))
                q3 = finite_or_none(raw.get("q3_score"))
                par = finite_or_none(raw.get("par_score", raw.get("local_score")))
                final = finite_or_none(raw.get("final_score"))
                if (
                    not video_id or not category or anchor < 0
                    or q2 is None or q3 is None
                    or par is None or final is None
                ):
                    continue

                current_frames = raw.get("current_frames") or raw.get("current_frame_indices") or ""
                if not parse_ints(current_frames):
                    current_frames = ";".join(
                        str(max(0, anchor + offset))
                        for offset in (-80, -70, -60, -50, -40, -30, -20, -10, 0)
                    )

                p3_exec = to_int(raw.get("p3_executed_replay", raw.get("p3_executed", 0)))
                p4_exec = to_int(raw.get("p4_executed_replay", raw.get("p4_executed", 0)))

                item: dict[str, Any] = {
                    "dataset": str(raw.get("dataset", "")).strip(),
                    "pixel_budget": raw.get("pixel_budget", ""),
                    "category": category,
                    "video_id": video_id,
                    "anchor": anchor,
                    "decision_index": to_int(raw.get("decision_index", anchor)),
                    "GT": to_int(raw.get("gt", raw.get("GT", 0))),
                    "q2_score": q2,
                    "q3_score": q3,
                    "p3_score": finite_or_none(raw.get("p3_score")),
                    "p4_score": finite_or_none(raw.get("p4_score")),
                    "q2_forward_margin": finite_or_none(raw.get("q2_forward_margin")),
                    "q2_reverse_margin": finite_or_none(raw.get("q2_reverse_margin")),
                    "q3_forward_margin": finite_or_none(raw.get("q3_forward_margin")),
                    "q3_reverse_margin": finite_or_none(raw.get("q3_reverse_margin")),
                    "p3_forward_margin": finite_or_none(raw.get("p3_forward_margin")),
                    "p3_reverse_margin": finite_or_none(raw.get("p3_reverse_margin")),
                    "p4_forward_margin": finite_or_none(raw.get("p4_forward_margin")),
                    "p4_reverse_margin": finite_or_none(raw.get("p4_reverse_margin")),
                    "current_p3_executed": p3_exec,
                    "current_p4_executed": p4_exec,
                    "current_p3_positive_delta": finite_or_none(raw.get("p3_positive_delta")) or 0.0,
                    "current_p4_positive_delta": finite_or_none(raw.get("p4_positive_delta")) or 0.0,
                    "current_p3_route_reason": raw.get("p3_route_reason_replay", raw.get("p3_route_reason", "")),
                    "current_p4_route_reason": raw.get("p4_route_reason_replay", raw.get("p4_route_reason", "")),
                    # UI compatibility names. These are read directly from the official replay.
                    "current_semantic_score": par,
                    "current_par_score": par,
                    "current_final_score": final,
                    "par_evidence": finite_or_none(raw.get("par_evidence")),
                    "par_delta": finite_or_none(raw.get("par_delta")),
                    "see_prev1_score": finite_or_none(raw.get("see_prev1_score")),
                    "see_prev2_score": finite_or_none(raw.get("see_prev2_score")),
                    "see_delta": finite_or_none(raw.get("see_delta")),
                    "see_reason": raw.get("see_reason", ""),
                    "prop_reason": raw.get("see_reason", ""),
                    "current_frames": current_frames,
                    "video_path": raw.get("video_path", ""),
                    "visual_tokens": raw.get("visual_token_count", raw.get("visual_tokens", "")),
                    "visual_prefill_sec": raw.get("shared_visual_prefill_sec", raw.get("visual_prefill_sec", "")),
                    "p3_tail_sec": raw.get("p3_tail_sec", ""),
                    "p4_tail_sec": raw.get("p4_tail_sec", ""),
                }
                merged[(video_id, anchor)] = item

    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    category_by_video: dict[str, str] = {}
    for item in merged.values():
        video_id = str(item["video_id"])
        rows_by_video.setdefault(video_id, []).append(item)
        category_by_video[video_id] = str(item["category"])

    for rows in rows_by_video.values():
        rows.sort(key=lambda row: int(row["anchor"]))

    rows_by_video = dict(sorted(rows_by_video.items()))
    categories = sorted(set(category_by_video.values()))
    return rows_by_video, category_by_video, categories


def audit_official_replay(
    rows_by_video: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize the persisted official replay; no scoring is recomputed here."""
    rows = [row for video_rows in rows_by_video.values() for row in video_rows]
    p3_count = sum(int(row.get("current_p3_executed", 0)) for row in rows)
    p4_count = sum(int(row.get("current_p4_executed", 0)) for row in rows)
    total = len(rows)
    par_changed = sum(
        abs(float(row["current_semantic_score"]) - float(row["q2_score"])) > 1e-8
        for row in rows
    )
    see_changed = sum(
        abs(float(row["current_final_score"]) - float(row["current_semantic_score"])) > 1e-8
        for row in rows
    )
    return {
        "windows": total,
        "videos": len(rows_by_video),
        "p3_routes": p3_count,
        "p4_routes": p4_count,
        "probes_per_window": (p3_count + p4_count) / total if total else 0.0,
        "par_changed": par_changed,
        "see_changed": see_changed,
        "source_p3_execution_mismatches": 0,
        "source_p4_execution_mismatches": 0,
        "max_source_semantic_diff": None,
        "max_source_final_diff": None,
    }


def load_ucf_gt_intervals(path: Path) -> dict[str, list[list[int]]]:
    intervals_by_video: dict[str, list[list[int]]] = {}
    if not path.exists():
        return intervals_by_video
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            video_id, _category, s1, e1, s2, e2 = parts
            intervals: list[list[int]] = []
            for start_text, end_text in ((s1, e1), (s2, e2)):
                start = to_int(start_text, -1)
                end = to_int(end_text, -1)
                if start >= 0 and end >= start:
                    intervals.append([start, end])
            intervals_by_video[video_id] = intervals
    return intervals_by_video


def resolve_msad_annotation(explicit: Path | None) -> Path | None:
    candidate = explicit or DEFAULT_MSAD_ANNOTATION
    if candidate is None:
        return None
    path = candidate.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_msad_gt_intervals(path: Path) -> dict[str, list[list[int]]]:
    intervals_by_video: dict[str, list[list[int]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "starting frame of anomaly", "ending frame of anomaly"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is not an MSAD anomaly annotation CSV; missing {sorted(missing)}"
            )
        for row in reader:
            video_id = str(row.get("name", "")).strip()
            start = to_int(row.get("starting frame of anomaly"), -1)
            end = to_int(row.get("ending frame of anomaly"), -1)
            if video_id and start >= 0 and end >= start:
                intervals_by_video[video_id] = [[start, end]]
    return intervals_by_video


def apply_gt_from_intervals(
    rows_by_video: dict[str, list[dict[str, Any]]],
    intervals_by_video: dict[str, list[list[int]]],
) -> None:
    for video_id, rows in rows_by_video.items():
        intervals = intervals_by_video.get(video_id, [])
        for row in rows:
            anchor = int(row["anchor"])
            row["GT"] = int(any(start <= anchor <= end for start, end in intervals))


def derive_gt_from_rows(
    rows_by_video: dict[str, list[dict[str, Any]]],
) -> dict[str, list[list[int]]]:
    result: dict[str, list[list[int]]] = {}
    for video, rows in rows_by_video.items():
        spans: list[list[int]] = []
        for i, row in enumerate(rows):
            if int(row["GT"]) != 1:
                continue
            anchor = int(row["anchor"])
            prev_anchor = int(rows[i - 1]["anchor"]) if i > 0 else anchor
            next_anchor = int(rows[i + 1]["anchor"]) if i + 1 < len(rows) else anchor
            start = (
                int(round((prev_anchor + anchor) / 2))
                if i > 0
                else min(parse_ints(row["current_frames"]) or [anchor])
            )
            end = (
                int(round((anchor + next_anchor) / 2))
                if i + 1 < len(rows)
                else max(parse_ints(row["current_frames"]) or [anchor])
            )
            spans.append([start, end])

        merged: list[list[int]] = []
        for start, end in sorted(spans):
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        result[video] = merged
    return result


def resolve_video_path(state: AppState, row: dict[str, Any]) -> Path | None:
    video_id = str(row.get("video_id", "")).strip()
    cached = state.video_paths.get(video_id)
    if cached is not None and cached.exists():
        return cached

    raw_text = str(row.get("video_path", "")).strip()
    if raw_text:
        raw = Path(raw_text)
        if raw.exists():
            state.video_paths[video_id] = raw
            return raw

    if state.dataset == "msad":
        root = (
            state.msad_normal_root
            if str(row.get("category", "")).strip().lower() == "normal"
            else state.msad_anomaly_root
        )
        if root is None:
            return None
        for suffix in (".mp4", ".avi"):
            candidate = root / f"{video_id}{suffix}"
            if candidate.exists():
                state.video_paths[video_id] = candidate
                return candidate
        return None

    if state.ucf_root is None:
        return None

    if raw_text:
        normalized = raw_text.replace("\\", "/")
        marker = "Anomaly-Detection-Dataset/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidate = state.ucf_root / Path(*relative.split("/"))
            if candidate.exists():
                state.video_paths[video_id] = candidate
                return candidate

    category = str(row.get("category", "")).strip()
    part = CATEGORY_PART.get(category)
    if part:
        directory = state.ucf_root / part / category
        for suffix in (".mp4", ".avi"):
            candidate = directory / f"{video_id}{suffix}"
            if candidate.exists():
                state.video_paths[video_id] = candidate
                return candidate
    return None


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def resize_to_area(image: Image.Image, area: int = 224 * 224) -> Image.Image:
    width, height = image.size
    ratio = width / max(1, height)
    new_width = max(1, int(round(math.sqrt(area * ratio))))
    new_height = max(1, int(round(math.sqrt(area / ratio))))
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def frame_index_from_pts(frame: Any, stream: Any, fps: float) -> int | None:
    if frame.pts is None or stream.time_base is None:
        return None
    try:
        return int(round(float(frame.pts * stream.time_base) * fps))
    except Exception:
        return None


def decode_frames_sequential(video_path: Path, frame_ids: list[int]) -> dict[int, Image.Image]:
    import av

    wanted = sorted({idx for idx in frame_ids if idx >= 0})
    wanted_set = set(wanted)
    frames: dict[int, Image.Image] = {}
    if not wanted:
        return frames
    max_wanted = max(wanted)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx in wanted_set:
                frames[idx] = frame.to_image().convert("RGB")
                if len(frames) == len(wanted_set):
                    break
            if idx > max_wanted:
                break
    return frames


def decode_frames(video_path: Path, frame_ids: list[int]) -> dict[int, Image.Image]:
    import av

    wanted = sorted({idx for idx in frame_ids if idx >= 0})
    frames: dict[int, Image.Image] = {}
    if not wanted:
        return frames
    min_wanted = min(wanted)
    max_wanted = max(wanted)
    wanted_set = set(wanted)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps_value = stream.average_rate or stream.guessed_rate
        fps = float(fps_value) if fps_value else 30.0
        time_base = float(stream.time_base) if stream.time_base else 0.0

        if min_wanted > 0 and fps > 0 and time_base > 0:
            seek_start = max(0, min_wanted - 90)
            try:
                container.seek(
                    int((seek_start / fps) / time_base),
                    stream=stream,
                    any_frame=False,
                    backward=True,
                )
            except Exception:
                return decode_frames_sequential(video_path, wanted)

        decoded_without_pts = 0
        fallback_start = max(0, min_wanted - 120)
        for frame in container.decode(stream):
            idx = frame_index_from_pts(frame, stream, fps)
            if idx is None:
                idx = fallback_start + decoded_without_pts
                decoded_without_pts += 1
            if idx in wanted_set:
                frames[idx] = frame.to_image().convert("RGB")
                if len(frames) == len(wanted_set):
                    break
            if idx > max_wanted + 120:
                break

    if len(frames) != len(wanted_set):
        exact = decode_frames_sequential(video_path, wanted)
        for idx in wanted_set - set(frames):
            if idx in exact:
                frames[idx] = exact[idx]
    return frames


def frame_in_intervals(frame_id: int, intervals: list[list[int]]) -> bool:
    return any(start <= frame_id <= end for start, end in intervals)


def labeled_mosaic(
    video_path: Path,
    frame_ids: list[int],
    intervals: list[list[int]] | None = None,
    area: int = 224 * 224,
    gap: int = 8,
) -> Image.Image:
    intervals = intervals or []
    if not frame_ids:
        canvas = Image.new("RGB", (640, 160), "#111111")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 60), "No frame indices available", fill="white", font=font(22))
        return canvas

    decoded = decode_frames(video_path, frame_ids)
    tiles: list[tuple[int, Image.Image]] = []
    for frame_id in frame_ids:
        image = decoded.get(frame_id)
        if image is None:
            image = Image.new("RGB", (224, 126), "#111111")
        tiles.append((frame_id, resize_to_area(image, area)))

    cols = min(3, max(1, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    max_w = max(image.width for _, image in tiles)
    max_h = max(image.height for _, image in tiles)
    canvas = Image.new(
        "RGB",
        (cols * max_w + (cols - 1) * gap, rows * max_h + (rows - 1) * gap),
        "black",
    )
    draw = ImageDraw.Draw(canvas)
    label_font = font(16)

    for idx, (frame_id, tile) in enumerate(tiles):
        row, col = divmod(idx, cols)
        x = col * (max_w + gap)
        y = row * (max_h + gap)
        canvas.paste(tile, (x, y))
        if frame_in_intervals(frame_id, intervals):
            for inset in range(4):
                draw.rectangle(
                    [x + inset, y + inset, x + tile.width - 1 - inset, y + tile.height - 1 - inset],
                    outline="#ff9100",
                )
        text = f"frame {frame_id}"
        bbox = draw.textbbox((0, 0), text, font=label_font)
        draw.rectangle(
            [x + 5, y + 5, x + 13 + bbox[2], y + 11 + bbox[3]],
            fill="black",
        )
        draw.text((x + 9, y + 8), text, fill="white", font=label_font)
    return canvas


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PARSEE-VAD Qualitative Viewer</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1115;
      --panel: #171a21;
      --panel2: #20242d;
      --text: #f5f7fb;
      --muted: #aab2c2;
      --line: #343a46;
      --gt: rgba(255,145,0,.24);
      --q2: #ff9f1c;
      --p3: #84d76d;
      --p4: #f06292;
      --semantic: #ffb454;
      --second: #c792ea;
      --final: #8AB4FF;
      --bad: #ff8c8c;
      --good: #8fe0a4;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      height: 100vh;
      overflow: hidden;
    }
    header {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #11141a;
      z-index: 2;
    }
    h1 { margin: 0 0 10px; font-size: 20px; }
    .controls {
      display: grid;
      grid-template-columns: 150px minmax(170px,.3fr) minmax(150px,.25fr) 160px 150px auto;
      gap: 10px;
      align-items: end;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    select, input {
      width: 100%; background: var(--panel); border: 1px solid var(--line);
      border-radius: 6px; color: var(--text); padding: 9px 10px; font-size: 14px;
    }
    .button {
      background: var(--panel2); color: var(--text); border: 1px solid var(--line);
      border-radius: 6px; padding: 9px 12px; cursor: pointer; font-size: 14px;
    }
    .button:hover { border-color: #8ab4ff; background: #263042; }
    main {
      display: grid; grid-template-columns: 360px minmax(0,1fr);
      min-height: 0; height: 100%; overflow: hidden;
    }
    aside {
      border-right: 1px solid var(--line); background: var(--panel);
      min-height: 0; height: 100%; overflow-y: auto; overflow-x: hidden;
    }
    .item { padding: 9px 12px; border-bottom: 1px solid var(--line); cursor: pointer; }
    .item:hover, .item.active { background: var(--panel2); }
    .item-top { display: flex; justify-content: space-between; gap: 10px; font-weight: 700; font-size: 13px; }
    .item-meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .viewer { padding: 16px; min-height: 0; height: 100%; overflow-y: auto; overflow-x: hidden; }
    .card {
      background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
      padding: 12px; margin-bottom: 14px;
    }
    .meta { display: grid; gap: 5px; color: var(--muted); font-size: 13px; }
    code { color: #9ec1ff; }
    .mosaic-toolbar { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; color: var(--muted); font-size: 13px; flex-wrap: wrap; }
    .mosaic-toolbar button {
      background: var(--panel2); color: var(--text); border: 1px solid var(--line);
      border-radius: 6px; padding: 6px 10px; cursor: pointer;
    }
    .mosaic-toolbar button:hover, .mosaic-toolbar button.active { border-color: #8ab4ff; background: #263042; }
    .mosaic-wrap {
      width: 100%; max-height: 68vh; overflow: auto; background: black;
      border: 1px solid var(--line); border-radius: 8px; display: flex;
      justify-content: center; align-items: flex-start;
    }
    #mosaic { width: 100%; max-width: none; object-fit: contain; background: black; display: block; transform-origin: top left; }
    .plot-card { padding: 12px 12px 8px; }
    .plot-title { font-weight: 700; margin: 0 0 3px; font-size: 15px; }
    .plot-subtitle { color: var(--muted); font-size: 12px; margin-bottom: 7px; }
    canvas {
      width: 100%; height: 310px; background: #0b0d12;
      border: 1px solid var(--line); border-radius: 8px; display: block;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 13px; color: var(--muted); font-size: 12px; margin-top: 8px; }
    .key { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 22px; height: 3px; display: inline-block; border-radius: 2px; }
    .swatch.gt { height: 12px; background: var(--gt); border: 1px solid rgba(255,145,0,.65); }
    .marker { display:inline-block; width:10px; height:10px; border:2px solid currentColor; background:transparent; }
    .marker.circle { border-radius: 50%; }
    .marker.triangle { width:0; height:0; border-left:6px solid transparent; border-right:6px solid transparent; border-bottom:11px solid currentColor; border-top:0; background:transparent; }
    .note { color: var(--muted); font-size: 12px; line-height: 1.45; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; grid-template-rows: 240px minmax(0,1fr); }
      aside { border-right:0; border-bottom:1px solid var(--line); }
      .controls { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
<header>
  <h1 id="pageTitle">PARSEE-VAD Qualitative Viewer</h1>
  <div class="controls">
    <label>Dataset<select id="dataset"></select></label>
    <label>Category<select id="category"></select></label>
    <label>Filter
      <select id="filter">
        <option value="">All windows</option>
        <option value="gt1">GT abnormal windows</option>
        <option value="gt0">GT normal windows</option>
        <option value="q2fp">Q2 false positives</option>
        <option value="p3route">P3 routed</option>
        <option value="p4route">P4 routed</option>
        <option value="parchanged">PAR changed score</option>
        <option value="seechanged">SEE changed score</option>
        <option value="scorechanged">PAR or SEE changed score</option>
        <option value="seecandidate">SEE rescued negative PAR score</option>
      </select>
    </label>
    <label>Search video<input id="videoSearch" placeholder="video id"></label>
    <label>Search frame<input id="frameSearch" placeholder="anchor frame"></label>
    <button class="button" id="reset">Reset view</button>
  </div>
</header>
<main>
  <aside id="list"></aside>
  <section class="viewer">
    <div class="mosaic-toolbar">
      <span>Image zoom</span>
      <button data-zoom="0.2">20%</button><button data-zoom="0.4">40%</button>
      <button data-zoom="0.6">60%</button><button data-zoom="0.8">80%</button>
      <button data-zoom="1.0">100%</button><button data-zoom="1.2">120%</button>
      <button id="zoomOut">-</button><button id="zoomIn">+</button>
    </div>

    <div class="card meta" id="meta"></div>
    <div class="mosaic-wrap"><img id="mosaic" alt="selected 9-frame observation"></div>

    <div class="card plot-card">
      <div class="plot-title">Q2, final score, and routed proposition probes</div>
      <div class="plot-subtitle">Q2 and final PARSEE-VAD scores are continuous curves; P3/P4 markers show only the windows where the frozen gates execute them.</div>
      <canvas id="routingChart"></canvas>
      <div class="legend">
        <span class="key"><span class="swatch" style="background:var(--q2)"></span>Q2 anomaly evidence</span>
        <span class="key"><span class="swatch" style="background:var(--final)"></span>Final score after PAR + SEE</span>
        <span class="key" style="color:var(--p3)"><span class="marker circle"></span>P3 executed (filled if P3&gt;0)</span>
        <span class="key" style="color:var(--p4)"><span class="marker triangle"></span>P4 executed (filled if P4&gt;0; position = P4 score)</span>
        <span class="key"><span class="swatch gt"></span>GT abnormal interval</span>
      </div>
    </div>

    <div class="card note" id="audit"></div>
  </section>
</main>
<script>
let payload = null;
let videos = {};
let selectedVideo = "";
let selectedIndex = 0;
let imageZoom = 1.0;

const datasetEl = document.getElementById("dataset");
const categoryEl = document.getElementById("category");
const filterEl = document.getElementById("filter");
const videoSearchEl = document.getElementById("videoSearch");
const frameSearchEl = document.getElementById("frameSearch");

function fmt(value, digits=3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "NA";
}
function signedFmt(value, digits=3) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "NA";
  const prefix = number > 0 ? "+" : "";
  return `${prefix}${number.toFixed(digits)}`;
}
function rowsAll() { return videos[selectedVideo] || []; }
function rowMatchesFilter(row) {
  const f = filterEl.value;
  const par = Number(row.current_semantic_score);
  const finalScore = Number(row.current_final_score);
  const parDelta = par - Number(row.q2_score);
  const seeDelta = finalScore - par;
  const eps = 1e-8;
  if (f === "gt1") return Number(row.GT) === 1;
  if (f === "gt0") return Number(row.GT) === 0;
  if (f === "q2fp") return Number(row.GT) === 0 && Number(row.q2_score) > 0;
  if (f === "p3route") return Number(row.current_p3_executed) === 1;
  if (f === "p4route") return Number(row.current_p4_executed) === 1;
  if (f === "parchanged") return Number.isFinite(parDelta) && Math.abs(parDelta) > eps;
  if (f === "seechanged") return Number.isFinite(seeDelta) && Math.abs(seeDelta) > eps;
  if (f === "scorechanged") return (Number.isFinite(parDelta) && Math.abs(parDelta) > eps) || (Number.isFinite(seeDelta) && Math.abs(seeDelta) > eps);
  if (f === "seecandidate") return Number(row.current_semantic_score) < 0 && Number(row.current_final_score) > Number(row.current_semantic_score);
  return true;
}
function rows() { return rowsAll().filter(rowMatchesFilter); }
function visibleVideos() {
  const category = categoryEl.value;
  const query = videoSearchEl.value.trim().toLowerCase();
  return Object.keys(videos).filter(video => {
    if (category && category !== "__all__" && payload.category_by_video[video] !== category) return false;
    if (query && !video.toLowerCase().includes(query)) return false;
    return videos[video].some(rowMatchesFilter);
  });
}
function clampSelection() {
  const rs = rows();
  if (!rs.length) { selectedIndex = 0; return; }
  selectedIndex = Math.max(0, Math.min(selectedIndex, rs.length - 1));
}
function selectedRow() { clampSelection(); return rows()[selectedIndex] || null; }

function renderList() {
  const list = document.getElementById("list");
  const shown = visibleVideos();
  if (!shown.includes(selectedVideo)) {
    selectedVideo = shown[0] || "";
    selectedIndex = 0;
  }
  list.innerHTML = "";
  shown.forEach(video => {
    const allRows = videos[video] || [];
    const filtered = allRows.filter(rowMatchesFilter);
    const gt1 = filtered.filter(row => Number(row.GT) === 1).length;
    const p3 = allRows.filter(row => Number(row.current_p3_executed) === 1).length;
    const p4 = allRows.filter(row => Number(row.current_p4_executed) === 1).length;
    const parChanged = allRows.filter(row => Math.abs(Number(row.current_semantic_score) - Number(row.q2_score)) > 1e-8).length;
    const seeChanged = allRows.filter(row => Math.abs(Number(row.current_final_score) - Number(row.current_semantic_score)) > 1e-8).length;
    const item = document.createElement("div");
    item.className = `item ${video === selectedVideo ? "active" : ""}`;
    item.innerHTML = `
      <div class="item-top"><span>${video}</span><span>${filtered.length}</span></div>
      <div class="item-meta">GT+ ${gt1} | P3 ${p3} | P4 ${p4}</div>
      <div class="item-meta">PAR changed ${parChanged} | SEE changed ${seeChanged}</div>`;
    item.addEventListener("click", () => { selectedVideo = video; selectedIndex = 0; render(); });
    list.appendChild(item);
  });
}

function searchFrame() {
  const query = frameSearchEl.value.trim();
  if (!/^\d+$/.test(query)) return;
  const target = Number(query);
  const rs = rows();
  if (!rs.length) return;
  let best = 0, bestDist = Infinity;
  rs.forEach((row, index) => {
    const dist = Math.abs(Number(row.anchor) - target);
    if (dist < bestDist) { bestDist = dist; best = index; }
  });
  selectedIndex = best;
}

function moveSelection(delta) {
  const rs = rows();
  if (!rs.length) return;
  selectedIndex = Math.max(0, Math.min(rs.length - 1, selectedIndex + delta));
  render();
}

function setImageZoom(value) {
  imageZoom = Math.max(.2, Math.min(1.2, value));
  const img = document.getElementById("mosaic");
  img.style.width = `${Math.round(imageZoom * 100)}%`;
  document.querySelectorAll("[data-zoom]").forEach(btn => {
    btn.classList.toggle("active", Math.abs(Number(btn.dataset.zoom) - imageZoom) < .01);
  });
}

function renderMeta(row) {
  const p3Run = Number(row.current_p3_executed) === 1;
  const p4Run = Number(row.current_p4_executed) === 1;
  const q2 = Number(row.q2_score);
  const par = Number(row.current_semantic_score);
  const finalScore = Number(row.current_final_score);
  const parDelta = par - q2;
  const seeDelta = finalScore - par;
  const totalDelta = finalScore - q2;
  document.getElementById("meta").innerHTML = `
    <div><b>${payload.dataset_label}: ${selectedVideo}</b> — decision anchor <code>${row.anchor}</code></div>
    <div>Category <code>${row.category}</code>; GT <code>${row.GT}</code>; pixel budget <code>${row.pixel_budget || ""}</code></div>
    <div>Q2 <code>${fmt(q2)}</code> &nbsp;|&nbsp; Q3 <code>${fmt(row.q3_score)}</code></div>
    <div>
      P3 gate <code>Q2&gt;0 ∧ Q3≥0</code> → <code>${p3Run ? "EXECUTE" : "skip"}</code>
      ${p3Run ? `; P3=<code>${fmt(row.p3_score)}</code>; +Δ=<code>${fmt(row.current_p3_positive_delta)}</code>` : ""}
    </div>
    <div>
      P4 gate <code>Q2≥1 ∧ Q3≥0 ∧ P3≤0</code> → <code>${p4Run ? "EXECUTE" : "skip"}</code>
      ${p4Run ? `; P4=<code>${fmt(row.p4_score)}</code>; +Δ=<code>${fmt(row.current_p4_positive_delta)}</code>` : ""}
    </div>
    <div>Score path: Q2 <code>${fmt(q2)}</code> → PAR <code>${fmt(par)}</code> (<code>${signedFmt(parDelta)}</code>) → Final <code>${fmt(finalScore)}</code> (<code>${signedFmt(seeDelta)}</code>)</div>
    <div>PAR evidence <code>${fmt(row.par_evidence)}</code>; SEE prev <code>${fmt(row.see_prev1_score)}</code>, <code>${fmt(row.see_prev2_score)}</code>; total Q2→Final Δ <code>${signedFmt(totalDelta)}</code>; SEE reason <code>${row.see_reason || row.prop_reason || "NA"}</code></div>
    <div>Current 9F <code>${row.current_frames || ""}</code></div>`;
}

function chartGeometry(width, height, rs, values) {
  const pad = {l: 58, r: 18, t: 18, b: 38};
  const xs = rs.map(row => Number(row.anchor));
  const ys = values.filter(Number.isFinite);
  const minX = Math.min(0, ...xs);
  const maxX = Math.max(...xs, 1);
  let minY = ys.length ? Math.min(0, ...ys) : -1;
  let maxY = ys.length ? Math.max(0, ...ys) : 1;
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const yPad = Math.max((maxY - minY) * .1, .2);
  minY -= yPad; maxY += yPad;
  const x = frame => pad.l + (Number(frame) - minX) / Math.max(1, maxX - minX) * (width - pad.l - pad.r);
  const y = score => pad.t + (maxY - Number(score)) / Math.max(1e-9, maxY - minY) * (height - pad.t - pad.b);
  return {pad,minX,maxX,minY,maxY,x,y};
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr,dpr);
  return {ctx,width:rect.width,height:rect.height};
}

function drawBackground(ctx, width, height, geometry, rs) {
  const {pad,x,y,minY,maxY,minX,maxX} = geometry;
  ctx.clearRect(0,0,width,height);
  ctx.fillStyle = "#0b0d12"; ctx.fillRect(0,0,width,height);
  const intervals = payload.gt_intervals[selectedVideo] || [];
  ctx.fillStyle = "rgba(255,145,0,.24)";
  intervals.forEach(([start,end]) => {
    const x0 = Math.max(pad.l,x(start)); const x1 = Math.min(width-pad.r,x(end));
    if (x1 >= x0) ctx.fillRect(x0,pad.t,Math.max(2,x1-x0),height-pad.t-pad.b);
  });
  ctx.strokeStyle = "#343a46"; ctx.lineWidth=1; ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,height-pad.b); ctx.lineTo(width-pad.r,height-pad.b); ctx.stroke();
  if (minY <= 0 && maxY >= 0) {
    ctx.strokeStyle = "rgba(255,255,255,.55)"; ctx.setLineDash([7,6]); ctx.beginPath();
    ctx.moveTo(pad.l,y(0)); ctx.lineTo(width-pad.r,y(0)); ctx.stroke(); ctx.setLineDash([]);
  }
  const row = selectedRow();
  if (row) {
    const frameIds = String(row.current_frames || "").split(/[,\s;]+/).filter(Boolean).map(Number).filter(Number.isFinite);
    const left = frameIds.length ? Math.min(...frameIds) : Number(row.anchor);
    const right = frameIds.length ? Math.max(...frameIds) : Number(row.anchor);
    const x0 = Math.max(pad.l,x(left)); const x1 = Math.min(width-pad.r,x(right));
    ctx.fillStyle="rgba(255,255,255,.06)"; ctx.fillRect(x0,pad.t,Math.max(2,x1-x0),height-pad.t-pad.b);
    ctx.strokeStyle="rgba(255,255,255,.75)"; ctx.lineWidth=1; ctx.strokeRect(x0,pad.t+1,Math.max(2,x1-x0),height-pad.t-pad.b-2);
  }
  ctx.fillStyle="#aab2c2"; ctx.font="12px Arial";
  ctx.fillText(maxY.toFixed(2),8,pad.t+4);
  if (minY <= 0 && maxY >= 0) ctx.fillText("0",38,y(0)+4);
  ctx.fillText(minY.toFixed(2),8,height-pad.b);
  ctx.fillText(String(Math.round(minX)),pad.l,height-12);
  ctx.fillText(String(Math.round(maxX)),width-pad.r-48,height-12);
}

function drawLine(ctx, rs, geometry, key, color, width=2.2) {
  ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); let started=false;
  rs.forEach(row => {
    const value=Number(row[key]);
    if (!Number.isFinite(value)) { started=false; return; }
    const px=geometry.x(row.anchor), py=geometry.y(value);
    if (!started) { ctx.moveTo(px,py); started=true; } else ctx.lineTo(px,py);
  });
  ctx.stroke();
  const selected=selectedRow();
  rs.forEach(row => {
    const value=Number(row[key]); if (!Number.isFinite(value)) return;
    const isSelected=selected && Number(row.anchor)===Number(selected.anchor);
    ctx.fillStyle=isSelected ? "#ffffff" : color; ctx.beginPath();
    ctx.arc(geometry.x(row.anchor),geometry.y(value),isSelected?4.7:2.4,0,Math.PI*2); ctx.fill();
    if (isSelected) { ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke(); }
  });
}

function drawP3Marker(ctx, x, y, positive, selected) {
  ctx.beginPath(); ctx.arc(x,y,selected?6.2:4.8,0,Math.PI*2);
  ctx.lineWidth=2; ctx.strokeStyle="#84d76d";
  ctx.fillStyle=positive ? "#84d76d" : "#0b0d12";
  ctx.fill(); ctx.stroke();
  if (selected) { ctx.strokeStyle="#ffffff"; ctx.lineWidth=1; ctx.stroke(); }
}
function drawP4Marker(ctx, x, y, positive, selected) {
  const r=selected?7:5.7;
  ctx.beginPath(); ctx.moveTo(x,y-r); ctx.lineTo(x-r*.9,y+r*.75); ctx.lineTo(x+r*.9,y+r*.75); ctx.closePath();
  ctx.lineWidth=2; ctx.strokeStyle="#f06292"; ctx.fillStyle=positive?"#f06292":"#0b0d12";
  ctx.fill(); ctx.stroke();
  if (selected) { ctx.strokeStyle="#ffffff"; ctx.lineWidth=1; ctx.stroke(); }
}

function drawRoutingChart() {
  const canvas=document.getElementById("routingChart");
  const {ctx,width,height}=setupCanvas(canvas); const rs=rowsAll();
  if (!rs.length) return;
  const values=[];
  rs.forEach(row => {
    values.push(Number(row.q2_score));
    values.push(Number(row.current_final_score));
    if (Number(row.current_p3_executed)===1) values.push(Number(row.p3_score));
    if (Number(row.current_p4_executed)===1) values.push(Number(row.p4_score));
  });
  const g=chartGeometry(width,height,rs,values); drawBackground(ctx,width,height,g,rs);
  drawLine(ctx,rs,g,"q2_score","#ff9f1c",2.2);
  drawLine(ctx,rs,g,"current_final_score","#8AB4FF",2.9);
  const selected=selectedRow();
  rs.forEach(row => {
    const isSelected=selected && Number(row.anchor)===Number(selected.anchor);
    if (Number(row.current_p3_executed)===1 && Number.isFinite(Number(row.p3_score)))
      drawP3Marker(ctx,g.x(row.anchor),g.y(row.p3_score),Number(row.p3_score)>0,isSelected);
    if (Number(row.current_p4_executed)===1 && Number.isFinite(Number(row.p4_score)))
      drawP4Marker(ctx,g.x(row.anchor),g.y(row.p4_score),Number(row.p4_score)>0,isSelected);
  });
}

function render() {
  renderList(); const row=selectedRow();
  if (!row) {
    document.getElementById("meta").innerHTML="<div>No matching windows.</div>";
    document.getElementById("mosaic").removeAttribute("src");
    drawRoutingChart(); return;
  }
  renderMeta(row);
  const mosaic=document.getElementById("mosaic");
  mosaic.onerror=()=>document.getElementById("meta").insertAdjacentHTML("afterbegin",'<div style="color:var(--bad)"><b>image load failed</b>: check the Python terminal.</div>');
  mosaic.src=`/mosaic?dataset=${encodeURIComponent(payload.dataset)}&video_id=${encodeURIComponent(selectedVideo)}&anchor=${encodeURIComponent(row.anchor)}&_=${Date.now()}`;
  setImageZoom(imageZoom); drawRoutingChart();
}

function chartClick(event) {
  const rs=rowsAll(); if (!rs.length) return;
  const canvas=event.currentTarget; const rect=canvas.getBoundingClientRect();
  const anchors=rs.map(row=>Number(row.anchor)); const minX=Math.min(0,...anchors), maxX=Math.max(...anchors,1);
  const padL=58,padR=18; const frame=minX+(event.clientX-rect.left-padL)/Math.max(1,rect.width-padL-padR)*(maxX-minX);
  let bestAnchor=anchors[0],bestDist=Infinity;
  anchors.forEach(anchor=>{ const d=Math.abs(anchor-frame); if(d<bestDist){bestDist=d;bestAnchor=anchor;} });
  const filtered=rows(); let best=0,fd=Infinity;
  filtered.forEach((row,index)=>{ const d=Math.abs(Number(row.anchor)-bestAnchor); if(d<fd){fd=d;best=index;} });
  selectedIndex=best; render();
}

async function loadData(dataset="") {
  const url=`/data?dataset=${encodeURIComponent(dataset)}&_=${Date.now()}`;
  const response=await fetch(url,{cache:"no-store"});
  if (!response.ok) throw new Error(await response.text());
  payload=await response.json(); videos=payload.rows_by_video||{};
  document.getElementById("pageTitle").textContent=payload.title||"PARSEE-VAD Qualitative Viewer";
  datasetEl.innerHTML=payload.available_datasets.map(item=>`<option value="${item.key}">${item.label}</option>`).join("");
  datasetEl.value=payload.dataset;
  categoryEl.innerHTML='<option value="__all__">All categories</option>'+payload.categories.map(category=>{
    const count=Object.keys(videos).filter(video=>payload.category_by_video[video]===category).length;
    return `<option value="${category}">${category} (${count} videos)</option>`;
  }).join("");
  categoryEl.value="__all__"; selectedVideo=""; selectedIndex=0;
  const a=payload.gate_audit||{};
  document.getElementById("audit").innerHTML=`
    <b>Canonical PARSEE-VAD replay:</b> loaded from the supplied window-score CSV; no scoring is recomputed by the viewer.
    Loaded ${a.windows ?? "?"} windows / ${a.videos ?? "?"} videos; P3 routes=${a.p3_routes ?? "?"}, P4 routes=${a.p4_routes ?? "?"}, probes/window=${fmt(a.probes_per_window,3)}.
    Windows changed by PAR=${a.par_changed ?? "?"}; changed by SEE=${a.see_changed ?? "?"}.`;
  render();
}

datasetEl.addEventListener("input",()=>loadData(datasetEl.value).catch(err=>alert(err)));
categoryEl.addEventListener("input",()=>{selectedVideo="";selectedIndex=0;render();});
filterEl.addEventListener("input",()=>{selectedIndex=0;render();});
videoSearchEl.addEventListener("input",()=>{selectedVideo="";selectedIndex=0;render();});
frameSearchEl.addEventListener("input",()=>{searchFrame();render();});
document.getElementById("reset").addEventListener("click",()=>{
  categoryEl.value="__all__"; filterEl.value=""; videoSearchEl.value=""; frameSearchEl.value="";
  selectedVideo=""; selectedIndex=0; imageZoom=1; render();
});
document.querySelectorAll("[data-zoom]").forEach(btn=>btn.addEventListener("click",()=>setImageZoom(Number(btn.dataset.zoom))));
document.getElementById("zoomOut").addEventListener("click",()=>setImageZoom(imageZoom-.1));
document.getElementById("zoomIn").addEventListener("click",()=>setImageZoom(imageZoom+.1));
document.getElementById("routingChart").addEventListener("click",chartClick);
window.addEventListener("resize",()=>{drawRoutingChart();});
window.addEventListener("keydown",event=>{
  const tag=(event.target&&event.target.tagName||"").toLowerCase();
  if (["input","select","textarea"].includes(tag)||event.target?.isContentEditable) return;
  if(event.key==="ArrowLeft"){event.preventDefault();moveSelection(-1);} else if(event.key==="ArrowRight"){event.preventDefault();moveSelection(1);}
});
loadData().catch(err=>{ document.getElementById("meta").innerHTML=`<div style="color:var(--bad)">${err}</div>`; });
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    states: dict[str, AppState] = {}
    default_dataset: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_data(self, data: bytes, content_type: str, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def choose_state(self, query: dict[str, list[str]]) -> AppState:
        requested = query.get("dataset", [""])[0]
        key = requested or self.default_dataset
        state = self.states.get(key)
        if state is None:
            raise KeyError(
                f"dataset '{key}' is not loaded; available={sorted(self.states)}"
            )
        return state

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send_data(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return

            if parsed.path == "/data":
                state = self.choose_state(query)
                labels = {"msad": "MSAD", "ucf": "UCF-Crime"}
                payload = {
                    "rows_by_video": state.rows_by_video,
                    "category_by_video": state.category_by_video,
                    "categories": state.categories,
                    "gt_intervals": state.gt_intervals,
                    "dataset": state.dataset,
                    "dataset_label": labels.get(state.dataset, state.dataset),
                    "available_datasets": [
                        {"key": key, "label": labels.get(key, key)}
                        for key in self.states
                    ],
                    "title": state.title,
                    "score_csvs": [str(path) for path in state.score_csvs],
                    "gate_audit": state.gate_audit,
                }
                self.send_data(json_bytes(payload), "application/json; charset=utf-8")
                return

            if parsed.path == "/mosaic":
                state = self.choose_state(query)
                video_id = query.get("video_id", [""])[0]
                anchor = to_int(query.get("anchor", ["0"])[0])
                rows = state.rows_by_video.get(video_id, [])
                row = next((item for item in rows if int(item["anchor"]) == anchor), None)
                if row is None:
                    self.send_data(b"window not found", "text/plain", 404)
                    return

                video_path = resolve_video_path(state, row)
                if video_path is None:
                    print(
                        f"[video] NOT FOUND dataset={state.dataset} video={video_id}; "
                        f"csv_path={row.get('video_path','')}",
                        flush=True,
                    )
                    self.send_data(b"video not found", "text/plain", 404)
                    return

                frame_ids = parse_ints(row.get("current_frames")) or [anchor]
                intervals = state.gt_intervals.get(video_id, [])
                print(
                    f"[image] dataset={state.dataset} {video_id} anchor={anchor} "
                    f"frames={frame_ids} path={video_path}",
                    flush=True,
                )
                image = labeled_mosaic(video_path, frame_ids, intervals)
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=92)
                self.send_data(buffer.getvalue(), "image/jpeg")
                return

            self.send_data(f"unknown path {parsed.path}".encode(), "text/plain", 404)
        except Exception as exc:
            import traceback
            print(
                f"[http] GET {self.path} failed: {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            self.send_data(
                f"{exc.__class__.__name__}: {exc}".encode("utf-8"),
                "text/plain",
                500,
            )


def build_state(
    dataset: str,
    score_csvs: list[Path],
    *,
    msad_annotation: Path | None,
    ucf_annotation: Path | None,
    msad_anomaly_root: Path | None,
    msad_normal_root: Path | None,
    ucf_root: Path | None,
) -> AppState:
    rows_by_video, category_by_video, categories = load_workflow_rows(score_csvs)
    if not rows_by_video:
        raise RuntimeError(f"{dataset}: no valid workflow windows loaded")

    if dataset == "msad":
        annotation_path = resolve_msad_annotation(msad_annotation)
        if annotation_path is not None:
            gt_intervals = load_msad_gt_intervals(annotation_path)
            apply_gt_from_intervals(rows_by_video, gt_intervals)
        else:
            print(
                "warning: MSAD annotation file not found; deriving display spans from official GT rows",
                file=sys.stderr,
            )
            gt_intervals = derive_gt_from_rows(rows_by_video)
        title = "PARSEE-VAD Fig.4 Viewer — MSAD"
    else:
        gt_intervals = load_ucf_gt_intervals(ucf_annotation) if ucf_annotation is not None else {}
        if not gt_intervals:
            print(
                "warning: UCF annotation not supplied/readable; deriving display spans from GT rows",
                file=sys.stderr,
            )
            gt_intervals = derive_gt_from_rows(rows_by_video)
        title = "PARSEE-VAD Fig.4 Viewer — UCF-Crime"

    audit = audit_official_replay(rows_by_video)
    return AppState(
        dataset=dataset,
        rows_by_video=rows_by_video,
        category_by_video=category_by_video,
        categories=categories,
        gt_intervals=gt_intervals,
        ucf_root=ucf_root.resolve() if ucf_root is not None else None,
        msad_anomaly_root=msad_anomaly_root.resolve() if msad_anomaly_root is not None else None,
        msad_normal_root=msad_normal_root.resolve() if msad_normal_root is not None else None,
        video_paths={},
        score_csvs=score_csvs,
        title=title,
        gate_audit=audit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PARSEE-VAD Fig.4 viewer using canonical window replay CSVs."
    )
    parser.add_argument(
        "--msad-scores-csv", type=Path, action="append", default=None,
        help="MSAD PARSEE-VAD window replay CSV. May be passed more than once."
    )
    parser.add_argument(
        "--ucf-scores-csv", type=Path, action="append", default=None,
        help="UCF PARSEE-VAD window replay CSV. May be passed more than once."
    )
    parser.add_argument("--msad-anomaly-root", type=Path, default=DEFAULT_MSAD_ANOMALY_ROOT)
    parser.add_argument("--msad-normal-root", type=Path, default=DEFAULT_MSAD_NORMAL_ROOT)
    parser.add_argument("--msad-annotation", type=Path, default=DEFAULT_MSAD_ANNOTATION)
    parser.add_argument("--ucf-root", type=Path, default=DEFAULT_UCF_ROOT)
    parser.add_argument("--ucf-annotation", type=Path, default=DEFAULT_UCF_ANNOTATION)
    parser.add_argument(
        "--default-dataset", choices=["msad", "ucf"], default="msad",
        help="Dataset initially shown in the page."
    )
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    states: dict[str, AppState] = {}
    requests = {
        "msad": resolve_dataset_score_csvs(args.msad_scores_csv, "msad"),
        "ucf": resolve_dataset_score_csvs(args.ucf_scores_csv, "ucf"),
    }

    for dataset, paths in requests.items():
        if not paths:
            print(
                f"warning: {dataset} official replay CSV was not found; "
                f"dataset will not appear in the page. "
                f"Pass --{dataset}-scores-csv explicitly.",
                file=sys.stderr,
            )
            continue
        try:
            state = build_state(
                dataset,
                paths,
                msad_annotation=args.msad_annotation,
                ucf_annotation=args.ucf_annotation,
                msad_anomaly_root=args.msad_anomaly_root,
                msad_normal_root=args.msad_normal_root,
                ucf_root=args.ucf_root,
            )
            states[dataset] = state
        except Exception as exc:
            print(
                f"warning: failed to load {dataset}: {exc.__class__.__name__}: {exc}",
                file=sys.stderr,
            )

    if not states:
        raise RuntimeError(
            "No dataset could be loaded. Pass --msad-scores-csv and/or --ucf-scores-csv plus the required dataset roots/annotations."
        )

    default_dataset = args.default_dataset if args.default_dataset in states else next(iter(states))
    Handler.states = states
    Handler.default_dataset = default_dataset

    print("PARSEE-VAD Fig.4 viewer: canonical replay (no scoring recomputation)")
    for dataset, state in states.items():
        a = state.gate_audit
        print(f"\n[{dataset}] loaded {a['windows']} windows across {a['videos']} videos")
        for path in state.score_csvs:
            print(f"  score: {path}")
        print(
            f"  current routes: P3={a['p3_routes']}, P4={a['p4_routes']}, "
            f"probes/window={a['probes_per_window']:.3f}"
        )
        print(
            f"  changed windows: PAR={a['par_changed']}, SEE={a['see_changed']}"
        )

    print(f"\nhttp://127.0.0.1:{args.port}/")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
