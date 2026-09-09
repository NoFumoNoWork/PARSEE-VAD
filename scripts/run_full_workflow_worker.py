from __future__ import annotations

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.qwen.model import DEFAULT_MODEL_CONFIG, load_qwen
from src.utils.io import write_json
from src.workflows import full_workflow


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkerContext:
    run_dir: Path
    shard_index: int
    shard_count: int
    config: dict[str, Any]
    prompts: dict[str, Any]
    scorer: Any


class LazyQwenScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.runtime = None

    def _runtime(self) -> Any:
        if self.runtime is None:
            model = self.config.get("model", {}) or {}
            device = int(model.get("device", 0) or 0)
            model_config = Path(str(model.get("config", DEFAULT_MODEL_CONFIG)))
            if not model_config.is_absolute():
                model_config = REPO_ROOT / model_config
            self.runtime = load_qwen(model_config, device)
        return self.runtime


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one PARSEE-VAD full-workflow shard.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "workflows" / "pixel_budget_sweep.yaml")
    parser.add_argument("--prompts", type=Path, default=REPO_ROOT / "configs" / "workflows" / "full_workflow_prompts.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    config = read_yaml(args.config)
    status_path = args.run_dir / "status" / "worker" / f"shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json"
    try:
        context = WorkerContext(
            run_dir=args.run_dir,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            config=config,
            prompts=read_yaml(args.prompts),
            scorer=LazyQwenScorer(config),
        )
        summary = full_workflow.run(context)
        write_json(status_path, {"status": "COMPLETE", "summary": summary})
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "FAILED",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
