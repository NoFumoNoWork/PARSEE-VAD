# PARSEE-VAD Workflow

This document specifies the paper-final causal inference and replay protocol.
The method follows **Expose → Acquire → Maintain**: expose proposition logits,
selectively acquire specialist evidence, then maintain short-lived evidence with
bounded temporal state.

## Decision Unit

Each decision anchor observes nine ordered frames:

```text
[-80, -70, -60, -50, -40, -30, -20, -10, 0]
```

The anchor is offset `0`; future frames are never used.

## Model Protocol

- Qwen3.5-9B for the main configuration;
- `thinking: false`;
- nine independent images rather than a mosaic;
- balanced A/B forced-choice scoring from logits;
- forward/reversed answer-order averaging;
- a shared visual-prefix prefill followed by clean proposition tails.

Q2/Q3/P3/P4 prompts remain independent; proposition prompt text is not chained.

## Proposition Semantics

- **Q2**: native visible anomaly / suspicious-behavior evidence.
- **Q3**: visible physical-development state; it is not an independent anomaly verifier.
- **P3**: forceful physical interaction between people.
- **P4**: consequential interaction with an object, vehicle, or environment.

## Proposition-Aware Routing

```text
P3 executes iff Q2 > 0 and Q3 >= 0
P4 executes iff Q2 >= 1 and Q3 >= 0 and P3 <= 0
```

The route is causal and determined from logits already available in the current
window.

## Current-Window Fusion

The bounded current-window fusion strength is:

```text
alpha = 0.75
```

Let `[x]+ = max(x, 0)`:

```text
e_Q3 = -tanh([-Q3]+)
e_Pk = tanh(Q3) * tanh(Pk)       for an executed specialist k
e     = clip(e_Q3 + e_P3 + e_P4, -1, 1)
```

For positive native suspicion:

```text
PAR = Q2 * (1 + alpha * e)       if Q2 > 0
PAR = Q2                         otherwise
```

The raw current-window correction is:

```text
Delta_PAR = PAR - Q2
```

No historical state enters this operator.

## One-Step Correction Carry

The carry stage stores only the immediately previous **raw** PAR correction.
Its decay is:

```text
rho_c = 0.5
```

When the current raw correction is negative and the previous raw correction was
positive:

```text
budget = rho_c * max(Delta_PAR(t-1), 0)
rescue = min(budget, -Delta_PAR(t))
carry_score = Q2(t) + Delta_PAR(t) + rescue
```

The rescue may cancel current over-suppression but may not raise the current score
above Q2. Crucially, the inherited rescue is never written back: the next state is
always the current **raw** `Delta_PAR(t)`. The mechanism is therefore one-step and
non-recursive.

## Streaming Evidence Escalation (SEE)

SEE repairs short temporal valleys using a finite delay line of **pre-SEE** scores.
The paper-final parameters are:

```text
H   = 2
tau = 0.2
eta = 0.5
rho = 0.5
```

For the previous `H` pre-SEE scores, define the positive state level:

```text
h_t = mean(max(x_{t-i}, 0), i=1..H)
```

SEE is eligible only when:

1. `H` positive pre-SEE history values are established;
2. current and previous Q2 are positive;
3. `Q2_t >= eta * Q2_{t-1}`;
4. the current pre-SEE score lies below `tau * h_t`.

For a non-negative valley, SEE restores only the missing amount to the relative
threshold:

```text
final = tau * h_t
```

For a negative valley, the implementation uses the bounded weighted estimate in
`src/workflows/scoring.py` with `rho = 0.5`.

Rescued outputs are never written into the SEE history. Only the pre-SEE input is
stored, so temporal rescue cannot recursively amplify itself.

## Canonical State Order

```text
Q2/Q3 logits
   │
   ├── Proposition-Aware Routing ──► optional P3/P4 logits
   │
   ▼
bounded current-window fusion (alpha=0.75)
   │
   ▼
one-step correction carry (rho_c=0.5)
   │
   ▼
finite-horizon SEE (H=2, tau=0.2, eta=0.5, rho=0.5)
   │
   ▼
final_score
```

## CPU Replay Boundary

Replay requires, for every decision window:

```text
video_id, anchor, q2_score, q3_score
p3_score when the P3 route executes
p4_score when the P4 route executes
```

Run:

```bash
python -m scripts.replay_parsee INPUT.csv --output OUTPUT.csv
```

The script reconstructs routing and resets both temporal states at video boundaries.

## Pixel Budgets

```text
default = 262,144 pixels
256sq   =  65,536 pixels
384sq   = 147,456 pixels
512sq   = 262,144 pixels
```

Resizing preserves aspect ratio, never upscales an image already below the area
budget, aligns dimensions to the Qwen visual grid, and disables processor-side
minimum-pixel upsampling.

## Frame-Level Alignment

Two causal timing views are implemented in `scripts/evaluate.py`.

### Completed interval

```text
score(a0) -> [0, a0]
score(ak) -> [a(k-1)+1, ak]
last score -> [last_anchor+1, video_end]
```

This is the primary completed-interval protocol.

### Availability

```text
neutral 0 -> [0, a0-1]
score(ak) -> [ak, a(k+1)-1]
last score -> [last_anchor, video_end]
```

This stricter frame-index protocol never applies a decision before its anchor. It
does not attempt to model sub-frame wall-clock model latency.

## Canonical Configuration

All paper-final parameters are explicit in:

```text
configs/workflows/parsee_final.yaml
```

Do not tune these parameters separately on UCF-Crime, XD-Violence, MSAD, or
UBnormal test sets.
