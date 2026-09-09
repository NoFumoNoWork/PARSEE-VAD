from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN_GROUP = "parsee_vad_pixel_budget_sweep"
DEFAULT_CONFIG = ROOT / "configs" / "workflows" / "pixel_budget_sweep.yaml"


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor the PARSEE-VAD pixel-budget matrix.")
    parser.add_argument("--run-id", default="main"); parser.add_argument("--run-dir", type=Path, default=None); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}; shards = int((config.get("launch",{}) or {}).get("shards",4))
    run_dir = args.run_dir.resolve() if args.run_dir else ROOT/"runs"/RUN_GROUP/args.run_id
    print(f"# {RUN_GROUP} :: {run_dir}"); total_done=total_expected=total_errors=complete=configs=0
    for dataset,budgets in (config.get("matrix",{}) or {}).items():
        for budget in budgets:
            configs += 1; done=expected=errors=0; states=[]
            for idx in range(shards):
                path=run_dir/"status"/dataset/budget/f"shard_{idx:02d}_of_{shards:02d}.json"
                if not path.exists(): states.append("NOT_STARTED"); continue
                status=read_json(path); states.append(str(status.get("status","UNKNOWN"))); done+=int(status.get("completed",0) or 0); expected+=int(status.get("total",0) or 0); errors+=int(status.get("errors",0) or 0)
            if states and all(s=="COMPLETE" for s in states): state="COMPLETE"; complete+=1
            elif any(s=="COMPLETE_WITH_ERRORS" for s in states): state="COMPLETE_WITH_ERRORS"
            elif any(s=="FAILED" for s in states): state="FAILED"
            elif any(s=="RUNNING" for s in states): state="RUNNING"
            elif all(s=="NOT_STARTED" for s in states): state="NOT_STARTED"
            else: state=",".join(states)
            total_done+=done; total_expected+=expected; total_errors+=errors; print(f"- {dataset}/{budget}: {state} completed={done}/{expected} errors={errors}")
    print(f"configs_complete: {complete}/{configs}"); print(f"completed_total: {total_done}/{total_expected}"); print(f"errors_total: {total_errors}")
    print(f"summary: {run_dir/'summaries'/'pixel_budget_sweep_summary.csv'}")


if __name__ == "__main__": main()
