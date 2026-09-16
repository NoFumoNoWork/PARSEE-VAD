#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import intish
from src.utils.io import read_yaml
from src.workflows.scoring import PARSEEConfig, PARSEEState, finite_float

DEFAULT_CONFIG = ROOT / "configs" / "workflows" / "parsee_final.yaml"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def video_id(row: dict[str, Any]) -> str:
    return str(row.get("video_id") or row.get("video") or "")


def infer_routes(
    row: dict[str, Any],
    q2: float,
    q3: float,
    pipeline: dict[str, Any],
) -> tuple[bool, bool]:
    p3_cfg = pipeline.get("p3_route", {}) or {}
    p4_cfg = pipeline.get("p4_route", {}) or {}

    p3_exec = q2 > float(p3_cfg.get("q2_gt", 0.0)) and q3 >= float(p3_cfg.get("q3_gte", 0.0))
    p3 = finite_float(row.get("p3_score"))
    p4_exec = (
        q2 >= float(p4_cfg.get("q2_gte", 1.0))
        and q3 >= float(p4_cfg.get("q3_gte", 0.0))
        and p3 is not None
        and p3 <= float(p4_cfg.get("p3_lte", 0.0))
    )
    return p3_exec, p4_exec


def replay(rows: list[dict[str, str]], pipeline: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("dataset") or ""), video_id(row)), []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        state = PARSEEState.from_pipeline(pipeline)
        for row in sorted(grouped[key], key=lambda item: intish(item.get("anchor", 0))):
            q2 = float(row["q2_score"])
            q3 = float(row["q3_score"])
            p3_exec, p4_exec = infer_routes(row, q2, q3, pipeline)
            if p3_exec and finite_float(row.get("p3_score")) is None:
                raise ValueError(f"missing P3 for routed row {key} anchor={row.get('anchor')}")
            if p4_exec and finite_float(row.get("p4_score")) is None:
                raise ValueError(f"missing P4 for routed row {key} anchor={row.get('anchor')}")
            scored = state.step(
                q2=q2,
                q3=q3,
                p3=row.get("p3_score"),
                p4=row.get("p4_score"),
                p3_executed=p3_exec,
                p4_executed=p4_exec,
            )
            item: dict[str, Any] = dict(row)
            item.update(
                {
                    "p3_executed_replay": int(p3_exec),
                    "p4_executed_replay": int(p4_exec),
                    "p3_route_reason_replay": "Q2>0 and Q3>=0" if p3_exec else "not_run",
                    "p4_route_reason_replay": "Q2>=1 and Q3>=0 and P3<=0" if p4_exec else "not_run",
                    **scored,
                }
            )
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the paper-final PARSEE-VAD scorer on existing Q2/Q3/P3/P4 logits."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="Optionally write the resolved scoring/routing parameters used for this replay.",
    )
    args = parser.parse_args()

    config = read_yaml(args.config) or {}
    pipeline = config.get("pipeline", {}) or {}
    rows = read_rows(args.input)
    replayed = replay(rows, pipeline)
    write_rows(args.output, replayed)

    if args.write_config is not None:
        resolved = {
            "version": pipeline.get("version", ""),
            "p3_route": pipeline.get("p3_route", {}) or {},
            "p4_route": pipeline.get("p4_route", {}) or {},
            "scoring": asdict(PARSEEConfig.from_pipeline(pipeline)),
        }
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(json.dumps(resolved, indent=2), encoding="utf-8")

    videos = len({(row.get("dataset", ""), video_id(row)) for row in replayed})
    print(f"WROTE {args.output} rows={len(replayed)} videos={videos}")


if __name__ == "__main__":
    main()
