from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any


class SequentialVideoFrameReader:
    """Sequential PyAV reader with bounded recent-frame caching and causal fallback."""

    def __init__(self, video_path: Path) -> None:
        import av

        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.stream = self.container.streams.video[0]
        self.frames = enumerate(self.container.decode(self.stream))
        self.cache: dict[int, Any] = {}
        self.recent: OrderedDict[int, Any] = OrderedDict()
        self.pinned: set[int] = set()
        self.last_decoded_index = -1
        self.last_decoded_image: Any | None = None
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.container.close()
            self.closed = True

    def pin(self, frame_ids: list[int] | None) -> None:
        if frame_ids:
            self.pinned.update(frame for frame in frame_ids if frame >= 0)

    def unpin(self, frame_ids: list[int] | None) -> None:
        if frame_ids:
            for frame in frame_ids:
                self.pinned.discard(frame)

    def get(self, frame_ids: list[int]) -> tuple[dict[int, Any], dict[int, int]]:
        wanted = sorted({frame for frame in frame_ids if frame >= 0})
        out: dict[int, Any] = {}
        if not wanted:
            return out, {}
        max_wanted = max(wanted)
        wanted_set = set(wanted)
        while self.last_decoded_index < max_wanted:
            try:
                idx, frame = next(self.frames)
            except StopIteration:
                break
            self.last_decoded_index = idx
            image = frame.to_image().convert("RGB")
            self.last_decoded_image = image
            self.recent[idx] = image
            self.recent.move_to_end(idx)
            while len(self.recent) > 256:
                self.recent.popitem(last=False)
            if idx in wanted_set or idx in self.pinned:
                self.cache[idx] = image
        for frame in wanted:
            if frame in self.cache:
                out[frame] = self.cache[frame]
        missing = [frame for frame in wanted if frame not in out]
        subs: dict[int, int] = {}
        decoded = sorted(idx for idx in self.cache if idx < max_wanted + 1)
        for frame in missing:
            prior = [idx for idx in decoded if idx < frame] or [idx for idx in self.recent if idx < frame]
            if prior:
                fallback = prior[-1]
                out[frame] = self.cache.get(fallback) or self.recent[fallback]
                self.cache[frame] = out[frame]
                subs[frame] = fallback
            elif self.last_decoded_image is not None and self.last_decoded_index < frame:
                out[frame] = self.last_decoded_image
                self.cache[frame] = out[frame]
                subs[frame] = self.last_decoded_index
            else:
                raise RuntimeError(f"cannot repair missing frame {frame}; no earlier decoded frame in {self.video_path}")
        return out, subs

    def evict_unpinned_before(self, frame_index: int) -> None:
        for frame in list(self.cache):
            if frame < frame_index and frame not in self.pinned:
                del self.cache[frame]
