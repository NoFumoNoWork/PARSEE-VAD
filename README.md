# PARSEE-VAD

PARSEE-VAD is a training-free online video anomaly detection workflow built around a vision-language model and proposition-aware evidence routing. Each decision uses only current and past frames; the model scores short A/B prompts from logits, selectively executes additional evidence probes, reranks the native anomaly score with bounded positive evidence, and optionally applies causal temporal propagation.

## Method

Each decision anchor observes nine ordered frames:

```text
[-80, -70, -60, -50, -40, -30, -20, -10, 0]
```

The frames are passed as nine independent images. PARSEE-VAD computes:

- **Q2**: native visible anomaly / suspicious-behavior score;
- **Q3**: directly visible physical development across the ordered frames;
- **P3**: forceful physical interaction between people;
- **P4**: consequential interaction with an object, vehicle, or the environment.

Routing is fixed by the experiment protocol:

```text
P3 runs when Q2 > 0 and Q3 >= 0.
P4 runs when Q2 >= 1 and Q3 >= 0 and P3 <= 0.
```

Positive P3/P4 evidence reranks Q2:

```text
semantic_score = Q2
               + tanh(P3) if P3 > 0
               + tanh(P4) if P4 > 0
```

A non-positive P4 score never suppresses the semantic score. See [`docs/WORKFLOW.md`](docs/WORKFLOW.md) for the exact cache, routing, propagation, pixel-budget and evaluation protocol.

## Repository Layout

```text
PARSEE-VAD/
├── configs/
│   ├── datasets/
│   ├── evaluation/
│   ├── model/
│   └── workflows/
├── data/manifests/
├── docs/
├── requirements/
├── scripts/
├── src/
└── tests/
```

Generated runs, raw videos, model weights and private logs remain outside version control.

## Installation

The reference environment used PyTorch 2.9.1 and Transformers 5.13.1. Install a PyTorch build appropriate for your CUDA/runtime, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

`requirements/environment_snapshot.txt` records the fuller reference environment for debugging reproducibility; it is not required as the primary installation specification.

## Paths and Model Setup

Export the model and dataset roots (see `.env.example`):

```bash
export PARSEE_QWEN35_9B_PATH=/path/to/Qwen3.5-9B
export PARSEE_UCF_ROOT=/path/to/UCF-Crime
export PARSEE_MSAD_ROOT=/path/to/MSAD
export PARSEE_XD_ROOT=/path/to/XD-Violence
export PARSEE_UBNORMAL_ROOT=/path/to/UBnormal
```

Public manifests should store video paths **relative** to these dataset roots. Absolute paths are accepted for private/local runs but should not be committed.

The model config is `configs/model/qwen35_9b_config.json`, which resolves `${PARSEE_QWEN35_9B_PATH}` at runtime.

## Decision Manifests

The full sweep expects:

```text
data/manifests/ucf_test_windows.csv
data/manifests/msad_test_windows.csv
data/manifests/xd_violence_test_windows.csv
data/manifests/ubnormal_test_windows.csv
```

The stable contract and dataset-specific GT inputs are documented in [`docs/DATASETS.md`](docs/DATASETS.md). UBnormal includes a public builder:

```bash
python -m scripts.build_ubnormal_manifest
```

when `PARSEE_UBNORMAL_ROOT` is set.

## Run the Pixel-Budget Sweep

Run all four datasets and all four budgets:

```bash
python -m scripts.run_pixel_budget_sweep --run-id main --resume
```

Or run one configuration:

```bash
python -m scripts.run_pixel_budget_sweep --run-id ucf384 --dataset ucf --budget 384sq
```

Outputs live under:

```text
runs/parsee_vad_pixel_budget_sweep/<run-id>/
```

The launcher stores workflow/prompt/model-config SHA-256 hashes in `run_manifest.json` and refuses to resume the same run id after either file changes.

Monitor a run with:

```bash
python -m scripts.monitor_pixel_budget_sweep --run-id main
```

## Official Frame-Level Evaluation

The inference sweep writes decision-anchor scores. Official benchmark metrics are computed afterward from `final_score` using non-overlapping window-end backward hold:

```text
first anchor a0:        score(a0) -> [0, a0]
subsequent anchor ak:   score(ak) -> [a(k-1)+1, ak]
last score:             held through video_end
```

No linear interpolation is used.

Set the official annotation paths in `.env.example`, then run:

```bash
python -m scripts.evaluate   --run-dir runs/parsee_vad_pixel_budget_sweep/main
```

The default evaluation config (`configs/evaluation/official_framelevel.yaml`) evaluates `384sq` and `512sq`. Use `--budgets` to evaluate other completed budgets.

Headline outputs include:

- UCF-Crime frame AUROC;
- XD-Violence frame AP;
- MSAD frame AUROC and AP;
- UBnormal micro frame AUROC and macro video AUROC.

`scikit-learn` is used for AUROC/AP so tied frame scores are handled by standard metric semantics. Per-frame CSV materialization is optional via `--write-frame-scores`.

## Tests

The CPU-only regression tests cover positive-only semantic fusion, propagation reset, non-overlapping frame expansion, AP ties and the no-upscale pixel-budget rule:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Reproducibility Boundary

The default workflow recomputes configurations from the supplied decision manifests. The formal Q2/Q3/P3/P4 prompts, routing thresholds, semantic fusion, propagation and pixel-budget resizing are defined by the checked-in configs and implementation.
