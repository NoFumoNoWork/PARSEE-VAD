# PARSEE-VAD

**PARSEE-VAD: Efficient Training-Free Online Video Anomaly Detection via Proposition-Aware Routing and Streaming Evidence Escalation**

PARSEE-VAD is a training-free, causal video-anomaly detection pipeline built on a
frozen visual-language model. The public code follows the paper-final method:
**Expose → Acquire → Maintain**.

## Method at a Glance

Each decision anchor observes nine causal frames:

```text
[-80, -70, -60, -50, -40, -30, -20, -10, 0]
```

The proposition-logit interface reads balanced A/B logits for:

- **Q2** — native visible anomaly / suspicious-behavior evidence;
- **Q3** — visible physical-development state;
- **P3** — forceful physical interaction between people;
- **P4** — consequential interaction with an object, vehicle, or environment.

### 1. Proposition-Aware Routing

```text
P3 executes iff Q2 > 0 and Q3 >= 0
P4 executes iff Q2 >= 1 and Q3 >= 0 and P3 <= 0
```

### 2. Bounded current-window fusion

With `alpha = 0.75`, routed proposition evidence is fused into the current Q2
score without temporal state. The exact operator is implemented in
[`src/workflows/scoring.py`](src/workflows/scoring.py).

### 3. Streaming Evidence Escalation

The temporal stage has two bounded parts:

```text
one-step correction carry: rho_c = 0.5
finite-horizon SEE:         H = 2, tau = 0.2, eta = 0.5, rho = 0.5
```

The correction carry stores only the previous **raw** positive PAR correction;
inherited rescue is never written back. SEE stores only pre-SEE scores, so rescued
outputs do not recursively enter future state.

The canonical paper configuration is
[`configs/workflows/parsee_final.yaml`](configs/workflows/parsee_final.yaml).
The equations and state semantics are documented in
[`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Repository Layout

```text
PARSEE-VAD/
├── configs/
│   ├── datasets/
│   ├── evaluation/
│   ├── model/
│   └── workflows/
│       └── parsee_final.yaml
├── data/
│   └── manifests/
├── docs/
│   ├── DATASETS.md
│   ├── RUNTIME.md
│   └── WORKFLOW.md
├── requirements/
├── scripts/
│   ├── evaluate.py
│   ├── replay_parsee.py
│   ├── run_full_workflow_worker.py
│   └── run_pixel_budget_sweep.py
├── src/
│   ├── evaluation/
│   ├── qwen/
│   └── workflows/
│       ├── full_workflow.py
│       └── scoring.py
└── tests/
```

Raw datasets, model weights, generated runs, and private machine-specific artifacts
are intentionally excluded.

## Installation

The recorded reference environment uses Python 3.11, PyTorch 2.9.1+cu128, and
Transformers 5.13.1.

```bash
pip install -r requirements.txt
```

For development/tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Paths and Public Manifests

Set model/dataset roots as environment variables; see [`.env.example`](.env.example):

```bash
export PARSEE_QWEN35_9B_PATH=/path/to/Qwen3.5-9B
export PARSEE_UCF_ROOT=/path/to/UCF-Crime
export PARSEE_MSAD_ROOT=/path/to/MSAD
export PARSEE_XD_ROOT=/path/to/XD-Violence
export PARSEE_UBNORMAL_ROOT=/path/to/UBnormal
```

Decision manifests are expected under `data/manifests/`. Their schema and the
official annotation inputs are described in [`docs/DATASETS.md`](docs/DATASETS.md).
Do not commit manifests containing absolute private paths.

## Fresh Inference

The default workflow points to the paper-final configuration:

```bash
python -m scripts.run_pixel_budget_sweep \
  --config configs/workflows/parsee_final.yaml \
  --run-id main \
  --resume
```

The runner supports the configured UCF-Crime, XD-Violence, MSAD, and UBnormal
matrices and writes the exact workflow/model/prompt snapshots into each run.

## CPU Replay from Routed Logits

If Q2/Q3/P3/P4 logits have already been produced, the final scorer can be replayed
without loading the VLM:

```bash
python -m scripts.replay_parsee \
  path/to/final_workflow_scores.csv \
  --output runs/replay/ucf_512.csv
```

Replay reconstructs the deployed P3/P4 route and resets temporal state at each
video boundary.

## Frame-Level Evaluation

The evaluator supports the two timing alignments reported with the paper.

Primary completed-interval alignment:

```bash
python -m scripts.evaluate \
  --run-dir runs/parsee_vad_pixel_budget_sweep/main \
  --alignment completed
```

Stricter availability alignment:

```bash
python -m scripts.evaluate \
  --run-dir runs/parsee_vad_pixel_budget_sweep/main \
  --alignment availability
```

`completed` assigns each decision to the non-overlapping interval ending at its
anchor. `availability` does not use a decision before its anchor; it holds the score
forward until the next decision, with a neutral score before the first anchor. This
frame-index alignment does not model sub-frame wall-clock inference latency.

Headline metrics are:

- UCF-Crime: frame AUROC;
- XD-Violence: frame AP;
- MSAD: frame AUROC and AP;
- UBnormal: micro frame AUROC and macro video AUROC.

## Runtime

The final direct MSAD 512sq full-set measurements and the measurement protocol are
summarized in [`docs/RUNTIME.md`](docs/RUNTIME.md). The reported 71.7% reduction is
a **specialist-query reduction**; the measured dense-to-routed end-to-end latency
reduction is 13.2%.

## Reproducibility Boundary

`src/workflows/scoring.py` is the canonical implementation of the paper-final score
construction. `configs/workflows/parsee_final.yaml` explicitly records every method
parameter used by that implementation. Public experiments should reuse these two
artifacts rather than duplicating equations or relying on Python defaults.
