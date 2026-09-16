# PARSEE-VAD

**Efficient Training-Free Online Video Anomaly Detection via Proposition-Aware Reasoning and Streaming Evidence Escalation**

PARSEE-VAD is a **training-free, causal online video anomaly detection (VAD)** framework built on frozen multimodal language models (MLLMs).

The method is organized around a simple principle:

> **Expose → Acquire → Maintain**

Instead of transmitting an increasingly rich visual or textual history, PARSEE-VAD first exposes structured semantic evidence from the **current causal observation**, selectively acquires additional specialist evidence only when warranted, and carries forward only a small bounded scalar state for short-term continuity.

The implementation in this repository corresponds to the paper-final PARSEE-VAD pipeline.

---

## Overview

At each decision anchor, PARSEE-VAD observes nine ordered causal frames:

```text
[-80, -70, -60, -50, -40, -30, -20, -10, 0]
````

where `0` denotes the current anchor. No future frames are used.

The pipeline contains two main modules:

1. **Proposition-Aware Reasoning (PAR)**

   * reads proposition-level evidence directly from MLLM logits;
   * conditionally acquires specialist propositions;
   * forms a bounded current-window anomaly score.

2. **Streaming Evidence Escalation (SEE)**

   * acts only after the current-window score has been formed;
   * preserves established evidence across short local valleys;
   * stores only bounded scalar state rather than visual/text history.

The canonical implementation is:

```text
current causal frames
        │
        ▼
shared visual prefix
        │
        ▼
Q2 / Q3 proposition logits
        │
        ├── PAR routing ──► optional P3 / P4 logits
        │
        ▼
bounded current-window fusion
        │
        ▼
one-step correction carry
        │
        ▼
finite-horizon SEE
        │
        ▼
final streaming anomaly score
```

---

## Proposition Evidence

PARSEE-VAD uses four visually grounded binary propositions.

| Proposition | Role                                                |
| ----------- | --------------------------------------------------- |
| **Q2**      | coarse anomaly evidence                             |
| **Q3**      | directly visible physical dynamics                  |
| **P3**      | forceful human interaction                          |
| **P4**      | concrete object / vehicle / environment interaction |

Q2 and Q3 are always evaluated.

P3 and P4 are specialist propositions that are acquired only when their semantic branch is warranted.

Each proposition is evaluated as an A/B forced-choice query in both answer orders. The two sign-aligned margins are averaged to reduce option-position sensitivity.

The exact proposition prompts are stored in:

```text
configs/workflows/full_workflow_prompts.yaml
```

---

## Proposition-Aware Reasoning

### 1. Conditional specialist acquisition

The deployed routing rule is:

```text
P3 executes iff:
    Q2 > 0
    and Q3 >= 0

P4 executes iff:
    P3 was executed
    and Q2 >= 1
    and Q3 >= 0
    and P3 <= 0
```

The routing decision controls **which specialist logits are materialized**. It does not itself modify the anomaly score.

The canonical thresholds are defined in:

```text
configs/workflows/parsee_final.yaml
```

---

### 2. Bounded current-window fusion

Let

```text
[x]+ = max(x, 0)
```

The current-window evidence terms are

```text
e_Q3 = -tanh([-Q3]+)

e_P3 = r3 * tanh(Q3) * tanh(P3)

e_P4 = r4 * tanh(Q3) * tanh(P4)
```

and

```text
E = clip(e_Q3 + e_P3 + e_P4, -1, 1)
```

The PAR-local score is

```text
L = Q2 + alpha * [Q2]+ * E
```

with

```text
alpha = 0.75
```

This stage is **memoryless**: it uses only evidence from the current causal observation.

Executed specialist evidence is signed. A proposition can therefore strengthen or suppress the current score depending on its logit evidence.

The canonical implementation is:

```text
src/workflows/scoring.py
```

---

## Streaming Evidence Escalation

SEE begins only after the PAR-local score has been formed.

It consists of two bounded temporal mechanisms.

### 1. One-step correction carry

The current raw PAR correction is

```text
Delta_PAR = L - Q2
```

If a positive correction in the previous decision is followed by a negative current correction, part of the previous raw correction may cancel the current suppression.

The deployed decay is

```text
rho_c = 0.5
```

Only the previous **raw** PAR correction is stored.

Inherited rescue is never written back into the state, preventing recursive self-support.

---

### 2. Finite-horizon evidence maintenance

SEE uses a short pre-SEE history:

```text
H   = 2
tau = 0.2
eta = 0.5
rho = 0.5
```

The continuity gate is activated only when short positive history has already been established and the current score forms a sufficiently deep local valley while coarse Q2 evidence remains consistent.

Crucially, SEE stores **pre-SEE scores**, not rescued outputs.

Therefore:

```text
rescued score
    ✗ does not become future supporting evidence
```

The mechanism is designed as sparse continuity support rather than broad temporal smoothing.

For the exact equations and state-update order, see:

```text
docs/WORKFLOW.md
```

---

## Canonical Paper Configuration

All paper-final parameters are explicitly recorded in:

```text
configs/workflows/parsee_final.yaml
```

The main configuration uses:

```text
Backbone: Qwen3.5-9B
Input:    9 causal images
Thinking: disabled

PAR:
    alpha = 0.75

P3 route:
    Q2 > 0
    Q3 >= 0

P4 route:
    Q2 >= 1
    Q3 >= 0
    P3 <= 0

Correction carry:
    rho_c = 0.5

SEE:
    H   = 2
    tau = 0.2
    eta = 0.5
    rho = 0.5
```

The same inference-rule configuration is used across the paper benchmarks rather than retuning the scalar parameters independently for each dataset.

---

## Paper Results

The paper evaluates PARSEE-VAD on three online VAD benchmarks.

| Dataset     | AUROC (%) |    AP (%) |
| ----------- | --------: | --------: |
| UCF-Crime   | **85.15** |         — |
| XD-Violence | **93.00** | **79.02** |
| MSAD        | **90.55** | **82.02** |

On the full MSAD evaluation:

```text
Dense specialist acquisition: 2.000 queries / window
PAR routed acquisition:       0.565 queries / window
Reduction:                    71.7%
```

The reduction in specialist queries should not be interpreted as an equivalent reduction in end-to-end latency; visual preprocessing and the shared visual prefix remain fixed costs.

See `docs/RUNTIME.md` for direct runtime measurements.

---

## Runtime

The final reported MSAD 512sq measurements use the same NVIDIA RTX 5880 Ada GPU model.

| Variant         | Specialists / window | Model mean | End-to-end mean |
| --------------- | -------------------: | ---------: | --------------: |
| Q2 only         |                0.000 |    0.802 s |         1.271 s |
| Dense all-probe |                2.000 |    1.073 s |         1.581 s |
| PARSEE routed   |                0.565 |    0.884 s |         1.372 s |

Relative to dense all-probe execution, routed PARSEE-VAD uses:

```text
71.7% fewer specialist queries
17.6% lower mean model-section time
13.2% lower mean end-to-end time
```

The full measurement protocol is documented in:

```text
docs/RUNTIME.md
```

---

## Repository Layout

```text
PARSEE-VAD/
├── configs/
│   ├── datasets/
│   ├── evaluation/
│   ├── model/
│   └── workflows/
│       ├── full_workflow_prompts.yaml
│       └── parsee_final.yaml
│
├── data/
│   └── manifests/
│
├── docs/
│   ├── DATASETS.md
│   ├── RUNTIME.md
│   └── WORKFLOW.md
│
├── requirements/
│
├── scripts/
│   ├── evaluate.py
│   ├── replay_parsee.py
│   ├── run_full_workflow_worker.py
│   └── run_pixel_budget_sweep.py
│
├── src/
│   ├── evaluation/
│   ├── qwen/
│   └── workflows/
│       ├── full_workflow.py
│       └── scoring.py
│
└── tests/
```

Raw datasets, model weights, generated experiment runs, and machine-specific private artifacts are intentionally excluded from the repository.

---

## Installation

The recorded reference environment uses:

```text
Python        3.11.15
PyTorch       2.9.1+cu128
Transformers  5.13.1
CUDA          12.8
Precision     bfloat16
```

Install the runtime dependencies with:

```bash
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Model and Dataset Paths

Machine-specific paths are not stored in the repository.

Set the required roots through environment variables; see `.env.example`.

For example:

```bash
export PARSEE_QWEN35_9B_PATH=/path/to/Qwen3.5-9B

export PARSEE_UCF_ROOT=/path/to/UCF-Crime
export PARSEE_XD_ROOT=/path/to/XD-Violence
export PARSEE_MSAD_ROOT=/path/to/MSAD
```

An additional UBnormal code path is also supported:

```bash
export PARSEE_UBNORMAL_ROOT=/path/to/UBnormal
```

UBnormal is retained as an additional supported dataset path and is not one of the three headline benchmarks reported above.

Decision manifests are expected under:

```text
data/manifests/
```

See:

```text
docs/DATASETS.md
```

for manifest schemas and dataset-specific annotation requirements.

Do not commit manifests containing private absolute paths.

---

## Fresh Inference

The default runner uses the paper-final workflow configuration.

### Run the configured matrix

```bash
python -m scripts.run_pixel_budget_sweep \
  --config configs/workflows/parsee_final.yaml \
  --run-id main \
  --resume
```

### Run one dataset and visual budget

For example, MSAD at 512sq:

```bash
python -m scripts.run_pixel_budget_sweep \
  --config configs/workflows/parsee_final.yaml \
  --dataset msad \
  --budget 512sq \
  --run-id msad_512 \
  --resume
```

The run directory stores snapshots of the workflow, prompts, and model configuration used for that execution.

Outputs are written under:

```text
runs/parsee_vad_pixel_budget_sweep/<run-id>/
```

---

## CPU Replay from Existing Proposition Logits

If Q2/Q3/P3/P4 logits have already been produced, the complete PARSEE scoring path can be replayed without loading the multimodal model.

```bash
python -m scripts.replay_parsee \
  path/to/final_workflow_scores.csv \
  --output runs/replay/parsee_scores.csv
```

To explicitly record the resolved scoring configuration:

```bash
python -m scripts.replay_parsee \
  path/to/final_workflow_scores.csv \
  --output runs/replay/parsee_scores.csv \
  --write-config runs/replay/resolved_config.json
```

Replay reconstructs the deployed routing policy and resets temporal state at video boundaries.

---

## Frame-Level Evaluation

PARSEE-VAD natively emits one causal score per completed decision interval.

The evaluator implements two timing alignments.

### Completed-interval alignment

This is the primary benchmark representation used for the headline results.

```bash
python -m scripts.evaluate \
  --run-dir runs/parsee_vad_pixel_budget_sweep/main \
  --alignment completed
```

Each immutable decision score is assigned to the non-overlapping interval ending at its decision anchor.

---

### Availability alignment

A stricter release-time representation is also implemented:

```bash
python -m scripts.evaluate \
  --run-dir runs/parsee_vad_pixel_budget_sweep/main \
  --alignment availability
```

Under this alignment, a decision is not applied to frames before the anchor at which it becomes available.

This distinction concerns frame-index score availability; it does not attempt to simulate sub-frame wall-clock model latency.

---

## Shared Prefix Reuse

Q2, Q3, P3, and P4 operate on the same causal visual observation.

The implementation therefore reuses a shared visual prefix before branching into proposition-specific text continuations.

Because the main Qwen3.5 backbone uses a hybrid decoder, prefix-state reuse is treated as an implementation approximation rather than assumed to be mathematically identical to independent full forwarding.

The paper separately measures routing-path agreement, proposition-logit deviation, score fidelity, and runtime benefit.

---

## Reproducibility Boundary

The canonical paper-final score construction is:

```text
src/workflows/scoring.py
```

The canonical parameter configuration is:

```text
configs/workflows/parsee_final.yaml
```

The canonical method description is:

```text
docs/WORKFLOW.md
```

Public experiments should reuse these artifacts rather than duplicating scoring equations or relying on implicit Python defaults.

The repository intentionally does not include:

* raw benchmark videos;
* pretrained model weights;
* private filesystem paths;
* machine usernames or hostnames;
* private runtime artifacts.

---

## Tests

Run:

```bash
python -m pytest -q
```

The tests cover core paper-final behavior including:

* signed current-window proposition fusion;
* negative-Q3 suppression;
* signed P3/P4 specialist evidence;
* proposition-aware routing;
* one-step non-recursive correction carry;
* finite-horizon SEE state;
* temporal reset at video boundaries;
* completed-interval and availability-aligned evaluation behavior.

---

## Additional Documentation

| Document           | Purpose                                                                      |
| ------------------ | ---------------------------------------------------------------------------- |
| `docs/WORKFLOW.md` | Exact PAR/SEE equations, routing rules, state semantics, and score alignment |
| `docs/DATASETS.md` | Dataset manifests and annotation requirements                                |
| `docs/RUNTIME.md`  | Hardware, timing protocol, and final runtime measurements                    |

---

## Citation

Citation information will be added after the review process.
