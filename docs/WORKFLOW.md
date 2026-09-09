# PARSEE-VAD Workflow

This document specifies the formal inference and evaluation protocol implemented by the public code.

## Decision Unit

Each manifest row is one causal decision window. The standard input is nine ordered frames at offsets:

```text
[-80, -70, -60, -50, -40, -30, -20, -10, 0]
```

The anchor is offset `0`; future frames are never used for inference.

## Model Protocol

- Qwen3.5-9B
- `thinking: false`
- nine independent image inputs (not a mosaic)
- A/B forced-choice scoring from logits
- forward/reverse candidate-order averaging
- one shared visual-prefix prefill followed by clean forked tails for Q2/Q3/P3/P4

Probe prompts are not chained into each other.

## Probes and Routing

Q2 and Q3 are always evaluated. P3 runs iff:

```text
Q2 > 0 and Q3 >= 0
```

P4 runs iff:

```text
Q2 >= 1 and Q3 >= 0 and P3 <= 0
```

P4 therefore runs only after the P3 route was taken and P3 supplied non-positive evidence.

The exact prompt text is in `configs/workflows/full_workflow_prompts.yaml`.

## Semantic Fusion

```python
score = q2_score
if p3_score > 0:
    score += tanh(p3_score)
if p4_score > 0:
    score += tanh(p4_score)
```

Negative P3/P4 values do not reduce the semantic score.

## Controlled Propagation

Propagation is applied after semantic fusion and resets at every video boundary. The reference parameters are in `configs/workflows/pixel_budget_sweep.yaml` (`delta=0.95`, `beta=1.5`, `positive_gamma=0.7`, cross-zero controls and reset threshold).

Propagation is causal in decision time: it consumes only the current semantic score and retained state from earlier anchors.

## Pixel Budgets

```text
default
256sq =  65,536 pixels
384sq = 147,456 pixels
512sq = 262,144 pixels
```

Non-default resizing preserves aspect ratio, never upscales an image already below the requested area, aligns dimensions to the Qwen patch/merge grid, and disables processor-side minimum-pixel upsampling.

## Decision-Score Outputs

Each dataset/budget/shard writes:

```text
final_workflow_scores.csv
final_workflow_failures.csv
summary.json
```

The main CSV contains identifiers, Q2/Q3/P3/P4 scores and margins, routing/execution flags, semantic/final scores, propagation state, resize/token statistics and latency fields.

The sweep summary in `summaries/pixel_budget_sweep_summary.csv` is a decision-window diagnostic summary. Paper benchmark metrics should be taken from the official frame-level evaluator below.

## Official Frame-Level Expansion

Official evaluation aligns each decision score retrospectively with the non-overlapping decision interval ending at that anchor:

```text
first anchor a0:       [0, a0]
anchor ak (k > 0):     [a(k-1)+1, ak]
tail after last anchor: hold last score through video_end
```

This is piecewise-constant backward assignment, not linear interpolation. Model inference remains causal; the frame-level reconstruction is an evaluation alignment applied after inference.

## Official Metrics

`scripts/evaluate.py` consumes completed decision CSVs and official annotations without recomputing VLM logits. It reports:

- UCF-Crime: micro frame AUROC;
- XD-Violence: frame AP;
- MSAD: micro frame AUROC and frame AP;
- UBnormal: micro frame AUROC and macro video AUROC.

AUROC and AP use `scikit-learn` metric semantics; this matters because backward hold creates many tied frame scores.

## Run Identity and Resume

Runs are stored under `runs/parsee_vad_pixel_budget_sweep/<run-id>/`. `run_manifest.json` records the git commit plus SHA-256 hashes of workflow, prompt, and model configs. An existing run id may only be resumed when those hashes match.
