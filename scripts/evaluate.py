from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.frame_level import (
    EXPANSION_METHOD, audit_decisions, expand_dataset_budget, load_msad_meta,
    load_ucf_meta, load_ubnormal_meta, load_xd_meta, read_csv_rows,
    resolve_config_path, write_csv,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "evaluation" / "official_framelevel.yaml"
DATASETS = ["ucf", "xd", "msad", "ubnormal"]


def read_scores(run_dir: Path, dataset: str, budget: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = run_dir / "outputs" / dataset / budget
    for path in sorted(pattern.glob("shard_*_of_*/tables/final_workflow_scores.csv")):
        rows.extend(read_csv_rows(path))
    if not rows:
        raise FileNotFoundError(f"no score CSVs found under {pattern}")
    return rows


def load_meta(dataset: str, first_rows: list[dict[str, str]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    spec = config["inputs"][dataset]
    if dataset == "ucf":
        return load_ucf_meta(resolve_config_path(spec["gt_manifest"], REPO_ROOT))
    if dataset == "msad":
        return load_msad_meta(resolve_config_path(spec["gt_manifest"], REPO_ROOT), resolve_config_path(spec["annotation_csv"], REPO_ROOT))
    if dataset == "xd":
        return load_xd_meta(first_rows, resolve_config_path(spec["annotation_txt"], REPO_ROOT), str(spec.get("root_env", "PARSEE_XD_ROOT")))
    if dataset == "ubnormal":
        return load_ubnormal_meta(first_rows, str(spec.get("root_env", "PARSEE_UBNORMAL_ROOT")))
    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official frame-level evaluation for PARSEE-VAD decision scores.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Pixel-budget run directory containing outputs/<dataset>/<budget>/...")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--budgets", nargs="+", default=None, help="Override budgets from config, e.g. 384sq 512sq")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-frame-scores", action="store_true", help="Write per-frame score CSVs (can be large).")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    budgets = args.budgets or list(config.get("budgets", ["384sq", "512sq"]))
    score_field = str(config.get("score_field", "final_score"))
    output = args.output_dir.resolve() if args.output_dir else run_dir / "official_framelevel"
    output.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    protocol_audit: list[dict[str, Any]] = []
    sanity: list[dict[str, Any]] = []
    signatures: dict[tuple[str, str], dict[str, tuple[int, int, int]]] = {}

    for dataset in args.datasets:
        rows_by_budget = {budget: read_scores(run_dir, dataset, budget) for budget in budgets}
        meta = load_meta(dataset, rows_by_budget[budgets[0]], config)
        for budget, rows in rows_by_budget.items():
            audit_row = audit_decisions(rows, dataset, budget, score_field)
            frame_path = output / "frame_scores" / f"{dataset}_{budget}_frame_scores.csv" if args.write_frame_scores else None
            audit, sanity_rows, signature = expand_dataset_budget(dataset, budget, rows, meta, score_field, frame_path)
            audit_row.update(audit)
            protocol_audit.append(audit_row)
            all_metrics.append({
                "budget": budget, "dataset": dataset, "videos": audit["videos"], "frames": audit["total_frames_evaluated"],
                "positive_frames": audit["gt_positive_frames"], "negative_frames": audit["gt_negative_frames"],
                "frame_micro_AUROC": audit["frame_micro_AUROC"], "frame_AP": audit["frame_AP"],
                "macro_video_AUROC": audit["macro_video_AUROC"], "macro_video_AUROC_videos_used": audit["macro_video_AUROC_videos_used"],
            })
            sanity.extend(sanity_rows)
            signatures[(dataset, budget)] = signature
        base_sig = signatures[(dataset, budgets[0])]
        for budget in budgets[1:]:
            if signatures[(dataset, budget)] != base_sig:
                raise RuntimeError(f"{dataset}: GT/video signatures differ between {budgets[0]} and {budget}")

    write_csv(output / "official_framelevel_metrics.csv", all_metrics)
    write_csv(output / "protocol_audit.csv", protocol_audit)
    write_csv(output / "causal_expansion_sanity.csv", sanity)

    by_key = {(row["dataset"], row["budget"]): row for row in all_metrics}
    table_rows: list[dict[str, Any]] = []
    for budget in budgets:
        row: dict[str, Any] = {"Budget": budget}
        if ("ucf", budget) in by_key: row["UCF_frame_AUC"] = by_key[("ucf", budget)]["frame_micro_AUROC"]
        if ("xd", budget) in by_key: row["XD_frame_AP"] = by_key[("xd", budget)]["frame_AP"]
        if ("msad", budget) in by_key:
            row["MSAD_frame_AUC"] = by_key[("msad", budget)]["frame_micro_AUROC"]
            row["MSAD_frame_AP"] = by_key[("msad", budget)]["frame_AP"]
        if ("ubnormal", budget) in by_key:
            row["UBnormal_micro_AUC"] = by_key[("ubnormal", budget)]["frame_micro_AUROC"]
            row["UBnormal_macro_video_AUC"] = by_key[("ubnormal", budget)]["macro_video_AUROC"]
        table_rows.append(row)
    write_csv(output / "table1_official_metrics.csv", table_rows)
    summary = {"run_dir": str(run_dir), "score_field": score_field, "expansion_method": EXPANSION_METHOD, "budgets": budgets, "metrics": all_metrics, "table1": table_rows}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "table1": table_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
