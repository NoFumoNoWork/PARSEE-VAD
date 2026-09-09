from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import ap, auroc, floatish, intish
from src.workflows.full_workflow import _read_rows, _video_id

RUN_GROUP = "parsee_vad_pixel_budget_sweep"
DEFAULT_CONFIG = ROOT / "configs" / "workflows" / "pixel_budget_sweep.yaml"
DEFAULT_PROMPTS = ROOT / "configs" / "workflows" / "full_workflow_prompts.yaml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(run_id: str) -> Path:
    return ROOT / "runs" / RUN_GROUP / run_id


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def snapshot_config(base: Path, config_path: Path, prompts_path: Path, model_config_path: Path) -> None:
    cfg = base / "configs"
    cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, cfg / "workflow.yaml")
    shutil.copy2(prompts_path, cfg / "prompts.yaml")
    shutil.copy2(model_config_path, cfg / "model.json")


def ensure_run_identity(base: Path, config_path: Path, prompts_path: Path, model_config_path: Path, selected_matrix: list[tuple[str, str]], retry_failures: bool) -> None:
    identity = {
        "experiment": RUN_GROUP, "run_dir": str(base), "git_commit": git_commit(),
        "config_sha256": file_sha256(config_path), "prompts_sha256": file_sha256(prompts_path),
        "model_config_sha256": file_sha256(model_config_path),
        "matrix": selected_matrix, "retry_failures": bool(retry_failures),
    }
    manifest = base / "run_manifest.json"
    if manifest.exists():
        old = read_json(manifest)
        for key in ("config_sha256", "prompts_sha256", "model_config_sha256"):
            if old.get(key) != identity[key]:
                raise RuntimeError(f"run identity mismatch for {key}; choose a new --run-id instead of resuming this directory")
        old.update({"matrix": selected_matrix, "retry_failures": bool(retry_failures)})
        write_json(manifest, old)
    else:
        write_json(manifest, identity)


def matrix(config: dict[str, Any], dataset: str | None, budget: str | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ds, budgets in (config.get("matrix", {}) or {}).items():
        if dataset and ds != dataset: continue
        for pb in budgets:
            if budget and pb != budget: continue
            out.append((str(ds), str(pb)))
    return out


def config_status(base: Path, dataset: str, budget: str, shards: int) -> str:
    status_dir = base / "status" / dataset / budget
    states: list[str] = []
    for idx in range(shards):
        path = status_dir / f"shard_{idx:02d}_of_{shards:02d}.json"
        if not path.exists(): return "NOT_STARTED"
        states.append(str(read_json(path).get("status", "NOT_STARTED")))
    if all(state == "COMPLETE" for state in states): return "COMPLETE"
    if any(state == "COMPLETE_WITH_ERRORS" for state in states): return "COMPLETE_WITH_ERRORS"
    if any(state == "FAILED" for state in states): return "FAILED"
    return "RUNNING"


def live_worker_pids(base: Path, dataset: str, budget: str) -> list[int]:
    launch_path = base / "status" / dataset / budget / "launch.json"
    if not launch_path.exists(): return []
    live: list[int] = []
    for worker in (read_json(launch_path).get("workers", []) or []):
        pid = worker.get("pid")
        if not pid: continue
        try: os.kill(int(pid), 0)
        except ProcessLookupError: continue
        except PermissionError: live.append(int(pid))
        else: live.append(int(pid))
    return live


def launch_config(base: Path, config: dict[str, Any], config_path: Path, prompts_path: Path, dataset: str, budget: str, dry_run: bool, retry_failures: bool) -> list[dict[str, Any]]:
    launch = config.get("launch", {}) or {}
    gpus = [str(gpu) for gpu in launch.get("gpus", [0])]
    shards = int(launch.get("shards", len(gpus) or 1))
    records: list[dict[str, Any]] = []
    for idx in range(shards):
        gpu = gpus[idx % len(gpus)]
        shard = f"shard_{idx:02d}_of_{shards:02d}"
        command = [sys.executable, "-m", "scripts.run_full_workflow_worker", "--config", str(config_path), "--prompts", str(prompts_path), "--run-dir", str(base), "--shard-index", str(idx), "--shard-count", str(shards)]
        record: dict[str, Any] = {"dataset": dataset, "pixel_budget": budget, "shard": shard, "gpu": gpu, "command": command}
        if not dry_run:
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PIXEL_BUDGET_DATASET"] = dataset; env["PIXEL_BUDGET_BUDGET"] = budget
            env["PIXEL_BUDGET_RETRY_FAILURES"] = "1" if retry_failures else "0"
            log_path = base / "logs" / dataset / budget / f"{shard}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record.update({"pid": proc.pid, "log": str(log_path)})
        records.append(record)
    return records


def wait_config(base: Path, dataset: str, budget: str, shards: int, poll_sec: int) -> str:
    while True:
        state = config_status(base, dataset, budget, shards)
        if state in {"COMPLETE", "COMPLETE_WITH_ERRORS", "FAILED"}: return state
        time.sleep(poll_sec)


def read_scores(base: Path, dataset: str, budget: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((base / "outputs" / dataset / budget).glob("shard_*_of_*/tables/final_workflow_scores.csv")):
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle: rows.extend(csv.DictReader(handle))
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values: return None
    values = sorted(values); idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def score_metrics(rows: list[dict[str, str]], key: str) -> tuple[float | None, float | None]:
    valid = [row for row in rows if row.get("gt") not in (None, "") and row.get(key) not in (None, "")]
    labels = [intish(row["gt"]) for row in valid]; scores = [floatish(row[key]) for row in valid]
    return auroc(labels, scores), ap(labels, scores)


def summarize_config(base: Path, dataset: str, budget: str) -> dict[str, Any]:
    rows = read_scores(base, dataset, budget)
    failure_rows = 0
    for path in (base / "outputs" / dataset / budget).glob("shard_*_of_*/tables/final_workflow_failures.csv"):
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle: failure_rows += sum(1 for _ in csv.DictReader(handle))
    q2_auc, q2_ap = score_metrics(rows, "q2_score"); sem_auc, sem_ap = score_metrics(rows, "semantic_score"); final_auc, final_ap = score_metrics(rows, "final_score")
    lat = [floatish(r.get("total_window_time_sec")) for r in rows if r.get("total_window_time_sec") not in (None, "")]
    pix = [floatish(r.get("actual_resized_area_mean")) for r in rows if r.get("actual_resized_area_mean") not in (None, "")]
    toks = [floatish(r.get("visual_token_count")) for r in rows if r.get("visual_token_count") not in (None, "")]
    summary = {
        "dataset": dataset, "pixel_budget": budget, "windows": len(rows), "videos": len({_video_id(r) for r in rows}), "failures": failure_rows,
        "auroc_q2": q2_auc, "ap_q2": q2_ap, "auroc_semantic": sem_auc, "ap_semantic": sem_ap, "auroc_final": final_auc, "ap_final": final_ap,
        "delta_auc_vs_q2": None if q2_auc is None or final_auc is None else final_auc-q2_auc,
        "delta_ap_vs_q2": None if q2_ap is None or final_ap is None else final_ap-q2_ap,
        "mean_latency": mean(lat) if lat else None, "median_latency": median(lat) if lat else None, "p95_latency": percentile(lat, .95),
        "mean_pixels": mean(pix) if pix else None, "median_pixels": median(pix) if pix else None, "p95_pixels": percentile(pix, .95),
        "mean_visual_tokens": mean(toks) if toks else None, "median_visual_tokens": median(toks) if toks else None, "p95_visual_tokens": percentile(toks, .95),
        "p3_executed": sum(intish(r.get("p3_executed")) for r in rows), "p3_positive": sum(1 for r in rows if floatish(r.get("p3_score", ""), -1e9) > 0),
        "p4_executed": sum(intish(r.get("p4_executed")) for r in rows), "p4_positive": sum(1 for r in rows if floatish(r.get("p4_score", ""), -1e9) > 0),
        "propagation_cross_zero": sum(intish(r.get("prop_cross_zero")) for r in rows),
    }
    write_json(base / "summaries" / dataset / budget / "summary.json", summary)
    return summary


def write_total_summary(base: Path, rows: list[dict[str, Any]]) -> None:
    path = base / "summaries" / "pixel_budget_sweep_summary.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset","pixel_budget","windows","videos","failures","auroc_q2","ap_q2","auroc_semantic","ap_semantic","auroc_final","ap_final","delta_auc_vs_q2","delta_ap_vs_q2","mean_latency","median_latency","p95_latency","mean_pixels","median_pixels","p95_pixels","mean_visual_tokens","median_visual_tokens","p95_visual_tokens","p3_executed","p3_positive","p4_executed","p4_positive","propagation_cross_zero"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or resume the PARSEE-VAD pixel-budget matrix from scratch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--run-id", default="main", help="Output identity under runs/parsee_vad_pixel_budget_sweep/<run-id>.")
    parser.add_argument("--dataset", choices=["ucf","msad","xd","ubnormal"]); parser.add_argument("--budget", choices=["default","256sq","384sq","512sq"])
    parser.add_argument("--resume", action="store_true", help="Accepted for explicitness; an existing matching run directory is resumed automatically.")
    parser.add_argument("--retry-failures", action="store_true"); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--no-wait", action="store_true"); parser.add_argument("--poll-sec", type=int, default=30)
    args = parser.parse_args()
    config_path = args.config.resolve(); prompts_path = args.prompts.resolve(); config = read_yaml(config_path)
    selected = matrix(config, args.dataset, args.budget); base = run_dir(args.run_id)
    model_config_path = Path(str((config.get("model", {}) or {}).get("config", "configs/model/qwen35_9b_config.json")))
    if not model_config_path.is_absolute(): model_config_path = ROOT / model_config_path
    base.mkdir(parents=True, exist_ok=True); (base/"logs").mkdir(exist_ok=True); (base/"status").mkdir(exist_ok=True); (base/"outputs").mkdir(exist_ok=True)
    ensure_run_identity(base, config_path, prompts_path, model_config_path, selected, args.retry_failures); snapshot_config(base, config_path, prompts_path, model_config_path)
    # Fail before GPU launch if a selected manifest is missing.
    for dataset, _ in selected: _read_rows(config, dataset)
    shards = int((config.get("launch", {}) or {}).get("shards", 4)); summaries: list[dict[str, Any]] = []
    for dataset, budget in selected:
        state = config_status(base, dataset, budget, shards)
        if state == "COMPLETE": print(f"SKIP COMPLETE {dataset}/{budget}", flush=True); summaries.append(summarize_config(base,dataset,budget)); continue
        if state == "COMPLETE_WITH_ERRORS" and not args.retry_failures: print(f"SKIP COMPLETE_WITH_ERRORS {dataset}/{budget}; use --retry-failures", flush=True); summaries.append(summarize_config(base,dataset,budget)); continue
        if state == "RUNNING":
            live = live_worker_pids(base,dataset,budget)
            if live:
                print(f"WAIT RUNNING {dataset}/{budget} pids={','.join(map(str,live))}", flush=True)
                if args.no_wait: continue
                final = wait_config(base,dataset,budget,shards,args.poll_sec); print(f"DONE {dataset}/{budget} state={final}", flush=True)
                if final in {"COMPLETE","COMPLETE_WITH_ERRORS"}: summaries.append(summarize_config(base,dataset,budget))
                if final == "FAILED": break
                continue
            print(f"RESUME STALE {dataset}/{budget}", flush=True)
        print(f"LAUNCH {dataset}/{budget} previous={state}", flush=True)
        records = launch_config(base,config,config_path,prompts_path,dataset,budget,args.dry_run,args.retry_failures)
        write_json(base/"status"/dataset/budget/"launch.json", {"state":"DRY_RUN" if args.dry_run else "RUNNING","retry_failures":bool(args.retry_failures),"workers":records})
        if args.dry_run or args.no_wait: continue
        final = wait_config(base,dataset,budget,shards,args.poll_sec); print(f"DONE {dataset}/{budget} state={final}", flush=True)
        if final in {"COMPLETE","COMPLETE_WITH_ERRORS"}: summaries.append(summarize_config(base,dataset,budget))
        if final == "FAILED": break
    if summaries: write_total_summary(base,summaries); print(f"SUMMARY {base/'summaries'/'pixel_budget_sweep_summary.csv'}", flush=True)


if __name__ == "__main__": main()
