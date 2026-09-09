from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

try:
    import av
except Exception:  # pragma: no cover
    av = None

from PIL import Image


OFFSETS = [-80, -70, -60, -50, -40, -30, -20, -10, 0]
STRIDE = 60
EXPECTED_ABNORMAL_TEST = 158
EXPECTED_NORMAL_TEST = 53
EXPECTED_TEST_TOTAL = EXPECTED_ABNORMAL_TEST + EXPECTED_NORMAL_TEST


def frame_count(path: Path) -> int:
    if av is None:
        raise RuntimeError("PyAV is required to build the UBnormal manifest")
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


def mask_frame(path: Path) -> int | None:
    match = re.search(r"_(\d+)_gt\.png$", path.name)
    return int(match.group(1)) if match else None


def abnormal_frames(video_path: Path) -> set[int]:
    if video_path.stem.startswith("normal_"):
        return set()
    ann_dir = video_path.with_name(f"{video_path.stem}_annotations")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"missing annotation directory: {ann_dir}")

    frames: set[int] = set()
    for mask in sorted(ann_dir.glob("*_gt.png")):
        frame = mask_frame(mask)
        if frame is None:
            continue
        with Image.open(mask) as image:
            if image.convert("L").getbbox() is not None:
                frames.add(frame)
    return frames


def scene_key(path: Path) -> tuple[int, str]:
    scene = path.parent.name
    match = re.search(r"(\d+)$", scene)
    return (int(match.group(1)) if match else 0, path.name)


def read_split(path: Path) -> list[str]:
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        item = item.replace("\\", "/").split("/")[-1]
        if item.lower().endswith(".mp4"):
            item = item[:-4]
        names.append(item)
    return names


def index_videos(data_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(data_root.glob("Scene*/*.mp4"), key=scene_key):
        if path.stem in out:
            duplicates.setdefault(path.stem, [out[path.stem]]).append(path)
            continue
        out[path.stem] = path
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise RuntimeError(f"duplicate video stems in UBnormal tree: {sample}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ["PARSEE_UBNORMAL_ROOT"]) if os.environ.get("PARSEE_UBNORMAL_ROOT") else None,
        help="UBnormal root. Defaults to PARSEE_UBNORMAL_ROOT.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/ubnormal_test_windows.csv"),
    )
    parser.add_argument(
        "--abnormal-test-list",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests" / "ubnormal_abnormal_test_video_names.txt",
    )
    parser.add_argument(
        "--normal-test-list",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests" / "ubnormal_normal_test_video_names.txt",
    )
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    if args.data_root is None:
        parser.error("--data-root is required unless PARSEE_UBNORMAL_ROOT is set")
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    video_index = index_videos(data_root)
    if not video_index:
        raise FileNotFoundError(f"no mp4 videos found under {data_root}/Scene*")

    abnormal_names = read_split(args.abnormal_test_list)
    normal_names = read_split(args.normal_test_list)
    if len(abnormal_names) != EXPECTED_ABNORMAL_TEST:
        raise RuntimeError(f"abnormal test split has {len(abnormal_names)} videos, expected {EXPECTED_ABNORMAL_TEST}")
    if len(normal_names) != EXPECTED_NORMAL_TEST:
        raise RuntimeError(f"normal test split has {len(normal_names)} videos, expected {EXPECTED_NORMAL_TEST}")

    split_names = abnormal_names + normal_names
    if len(set(split_names)) != EXPECTED_TEST_TOTAL:
        raise RuntimeError("duplicate names found in official UBnormal test split")

    missing = [name for name in split_names if name not in video_index]
    if missing:
        sample = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} official test videos missing under {data_root}: {sample}")

    videos = [video_index[name] for name in split_names]

    rows: list[dict[str, object]] = []
    total_positive = 0
    total_negative = 0

    for video_idx, path in enumerate(videos, start=1):
        nframes = frame_count(path)
        positives = abnormal_frames(path)
        category = "Normal" if path.stem.startswith("normal_") else "Abnormal"
        video_id = f"{path.parent.name}/{path.stem}"

        for decision_index, anchor in enumerate(range(0, nframes, args.stride)):
            frames = [max(0, anchor + off) for off in OFFSETS]
            gt = int(anchor in positives)
            total_positive += gt
            total_negative += 1 - gt
            rows.append(
                {
                    "dataset": "ubnormal",
                    "category": category,
                    "video_id": video_id,
                    "anchor": anchor,
                    "decision_index": decision_index,
                    "gt": gt,
                    "current_frames": ";".join(map(str, frames)),
                    "video_path": path.relative_to(data_root).as_posix(),
                }
            )

        if video_idx % 25 == 0 or video_idx == len(videos):
            print(
                f"[{video_idx}/{len(videos)}] videos, "
                f"rows={len(rows)}, positive={total_positive}, negative={total_negative}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "category",
        "video_id",
        "anchor",
        "decision_index",
        "gt",
        "current_frames",
        "video_path",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("WROTE", args.output)
    print("videos", len(videos))
    print("abnormal_test_videos", len(abnormal_names))
    print("normal_test_videos", len(normal_names))
    print("windows", len(rows))
    print("positive_windows", total_positive)
    print("negative_windows", total_negative)


if __name__ == "__main__":
    main()
