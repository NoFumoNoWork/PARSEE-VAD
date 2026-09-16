from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


PROPOSITION_RUNTIME_VERSION = "cached_text_tail_exact_position"


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clip(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


@dataclass(frozen=True)
class PARConfig:
    """Configuration for bounded memoryless current-window proposition fusion."""

    alpha: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError(f"PAR alpha must be in [0, 1], got {self.alpha}")


@dataclass(frozen=True)
class CorrectionCarryConfig:
    """One-step carryover of *positive PAR correction*, not score.

    If the previous window established a positive semantic correction and the
    current window applies a negative PAR correction, a decayed fraction of
    the previous *raw* positive correction may cancel part of the current
    suppression.  The rescue is capped at -par_delta, so history can never
    raise the current score above the current Q2 baseline.

    Only the immediately previous raw PAR delta is stored.  Inherited rescue
    is never written back into the carry state, making this mechanism strictly
    one-step and non-recursive.
    """

    enabled: bool = True
    rho: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.rho) <= 1.0:
            raise ValueError(f"correction-carry rho must be in [0, 1], got {self.rho}")


@dataclass(frozen=True)
class SEEConfig:
    """Gated finite-horizon anomaly-state persistence for temporal valleys."""

    enabled: bool = True
    rho: float = 0.5
    horizon: int = 2
    valley_ratio: float = 0.2
    q2_continuity: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.rho) <= 1.0:
            raise ValueError(f"SEE rho must be in [0, 1], got {self.rho}")
        if int(self.horizon) < 1:
            raise ValueError(f"SEE horizon must be >= 1, got {self.horizon}")
        if not 0.0 <= float(self.valley_ratio) <= 1.0:
            raise ValueError(f"SEE valley_ratio must be in [0, 1], got {self.valley_ratio}")
        if not 0.0 <= float(self.q2_continuity) <= 1.0:
            raise ValueError(f"SEE q2_continuity must be in [0, 1], got {self.q2_continuity}")


@dataclass(frozen=True)
class PARSEEConfig:
    """Paper-final PARSEE-VAD scoring configuration.

    Score order: current-window fusion -> one-step correction carry ->
    finite-horizon Streaming Evidence Escalation (SEE).
    """

    par: PARConfig = PARConfig()
    correction_carry: CorrectionCarryConfig = CorrectionCarryConfig()
    see: SEEConfig = SEEConfig()

    @classmethod
    def from_pipeline(cls, pipeline: dict[str, Any]) -> "PARSEEConfig":
        par_raw = pipeline.get("par", {}) or {}
        carry_raw = pipeline.get("correction_carry", {}) or {}
        see_raw = pipeline.get("see", {}) or {}
        return cls(
            par=PARConfig(
                alpha=float(par_raw.get("alpha", 0.75)),
            ),
            correction_carry=CorrectionCarryConfig(
                enabled=bool(carry_raw.get("enabled", True)),
                rho=float(carry_raw.get("rho", 0.5)),
            ),
            see=SEEConfig(
                enabled=bool(see_raw.get("enabled", True)),
                rho=float(see_raw.get("rho", 0.5)),
                horizon=int(see_raw.get("horizon", 2)),
                valley_ratio=float(see_raw.get("valley_ratio", 0.2)),
                q2_continuity=float(see_raw.get("q2_continuity", 0.5)),
            ),
        )


def dimensionless_par(
    q2: float,
    q3: float,
    p3: Any,
    p4: Any,
    *,
    p3_executed: bool,
    p4_executed: bool,
    config: PARConfig,
) -> dict[str, float]:
    """Compute bounded memoryless current-window fusion.

    The function name is retained for compatibility with existing result schemas;
    PAR routing itself is performed upstream and determines which specialists are
    executed.


      e_Q3 = -tanh([-q3]_+)
      e_Pk = tanh(q3) * tanh(pk), for an executed specialist k
      E    = clip(e_Q3 + e_P3 + e_P4, -1, 1)

      L_t = q2_t * (1 + alpha * E_t),  if q2_t > 0
            q2_t,                      otherwise.

    The fusion operator is strictly current-window: no historical value enters here.
    """

    q2f = float(q2)
    q3f = float(q3)
    alpha = float(config.alpha)

    q3_evidence = -math.tanh(max(-q3f, 0.0))

    p3f = finite_float(p3)
    p4f = finite_float(p4)
    p3_evidence = (
        math.tanh(q3f) * math.tanh(p3f)
        if p3_executed and p3f is not None
        else 0.0
    )
    p4_evidence = (
        math.tanh(q3f) * math.tanh(p4f)
        if p4_executed and p4f is not None
        else 0.0
    )

    evidence_raw = q3_evidence + p3_evidence + p4_evidence
    evidence = _clip(evidence_raw, -1.0, 1.0)

    if q2f > 0.0:
        scale = 1.0 + alpha * evidence
        par_score = q2f * scale
    else:
        scale = 1.0
        par_score = q2f

    par_delta = par_score - q2f
    return {
        "q3_evidence": float(q3_evidence),
        "p3_evidence": float(p3_evidence),
        "p4_evidence": float(p4_evidence),
        "par_evidence_raw": float(evidence_raw),
        "par_evidence": float(evidence),
        "par_alpha": float(alpha),
        "par_scale": float(scale),
        "par_delta": float(par_delta),
        "par_score": float(par_score),
        # Public aliases retained because existing analysis scripts use these
        # generic score names for the raw current-window PAR result.
        "local_score": float(par_score),
        "semantic_score": float(par_score),
    }


@dataclass
class PositiveCorrectionCarryState:
    """One-step, non-recursive carry of raw positive fusion correction."""

    config: CorrectionCarryConfig
    previous_raw_par_delta: float = 0.0

    def reset(self) -> None:
        self.previous_raw_par_delta = 0.0

    def seed_previous_raw_par_delta(self, value: float) -> None:
        value = float(value)
        self.previous_raw_par_delta = value if math.isfinite(value) else 0.0

    def step(self, q2: float, par_delta: float, par_score: float) -> dict[str, Any]:
        q2f = float(q2)
        raw_delta = float(par_delta)
        raw_score = float(par_score)
        rho = float(self.config.rho)

        prev_delta = float(self.previous_raw_par_delta)
        prev_positive = max(prev_delta, 0.0)
        budget = rho * prev_positive

        rescue = 0.0
        applied = 0
        if self.config.enabled and raw_delta < 0.0 and budget > 0.0:
            # Never allow historical evidence to make the current semantic
            # correction positive. It may only cancel current over-suppression.
            rescue = min(budget, -raw_delta)
            applied = int(rescue > 0.0)

        adjusted_delta = raw_delta + rescue
        carry_score = q2f + adjusted_delta

        # Numerical safety: by construction, a rescued negative correction
        # cannot push the score above the current Q2 baseline.
        if raw_delta < 0.0:
            carry_score = min(carry_score, q2f)
            adjusted_delta = carry_score - q2f

        if not self.config.enabled:
            reason = "disabled"
        elif raw_delta >= 0.0:
            reason = "current_not_suppressed"
        elif prev_positive <= 0.0:
            reason = "no_previous_positive_correction"
        elif applied:
            reason = "previous_positive_correction_offsets_suppression"
        else:
            reason = "carry_not_helpful"

        # Crucial: store the *raw current PAR delta*, never the inherited rescue.
        # Therefore the mechanism cannot recursively amplify itself.
        self.previous_raw_par_delta = raw_delta

        return {
            "carry_enabled": int(self.config.enabled),
            "carry_rho": float(rho),
            "carry_prev_par_delta": float(prev_delta),
            "carry_prev_positive_delta": float(prev_positive),
            "carry_budget": float(budget),
            "carry_rescue_delta": float(rescue),
            "carry_adjusted_par_delta": float(adjusted_delta),
            "carry_score": float(carry_score),
            "carry_applied": int(applied),
            "carry_reason": reason,
            # Useful audit quantity: should be zero except tiny floating error.
            "carry_score_minus_q2": float(carry_score - q2f),
            "carry_score_minus_raw_par": float(carry_score - raw_score),
        }


@dataclass
class StreamingEvidenceEscalationState:
    """Finite-horizon gated persistence aimed at anomaly valleys, not crossings.

    The state stores only pre-SEE scores and Q2 values.  A short delay-line
    estimates a recently established anomaly level.  Feedback is one-sided,
    finite-horizon, and non-recursive: rescued outputs never enter the state.
    """

    config: SEEConfig
    score_history: list[float] = field(default_factory=list)
    q2_history: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.score_history.clear()
        self.q2_history.clear()

    def seed_history(self, scores: list[float], q2_values: list[float] | None = None) -> None:
        horizon = int(self.config.horizon)
        self.score_history = [float(v) for v in scores if math.isfinite(float(v))][-horizon:]
        self.q2_history = [] if q2_values is None else [float(v) for v in q2_values if math.isfinite(float(v))][-horizon:]

    def step(self, score: float, q2: float) -> dict[str, Any]:
        x_t = float(score)
        q2_t = float(q2)
        rho = float(self.config.rho)
        horizon = int(self.config.horizon)
        tau = float(self.config.valley_ratio)
        eta = float(self.config.q2_continuity)

        recent_scores = list(reversed(self.score_history[-horizon:]))
        recent_q2 = list(reversed(self.q2_history[-horizon:]))
        padded_scores = recent_scores + [0.0] * (horizon - len(recent_scores))
        padded_q2 = recent_q2 + [0.0] * (horizon - len(recent_q2))
        positive_history = [max(v, 0.0) for v in padded_scores]

        # Two-step finite state readout; current x_t and rescued scores are excluded.
        h_t = float(sum(positive_history) / max(horizon, 1))
        established = int(len(self.score_history) >= horizon and all(v > 0.0 for v in self.score_history[-horizon:]))
        q2_prev = padded_q2[0] if self.q2_history else 0.0
        q2_continuous = int(bool(self.q2_history) and q2_t > 0.0 and q2_prev > 0.0 and q2_t >= eta * q2_prev)

        threshold = tau * h_t
        valley_error = max(threshold - x_t, 0.0)
        gate = int(self.config.enabled and established and q2_continuous and h_t > 0.0 and x_t < threshold)

        candidate = x_t
        feedback = 0.0
        reason = "no_relative_valley"
        if gate and x_t >= 0.0:
            # Positive valley: restore only the missing amount to tau of recent state.
            feedback = valley_error
            candidate = x_t + feedback
            reason = "positive_temporal_valley"
        elif gate and x_t < 0.0:
            # Extreme valley: bounded weighted state estimate, still using only past state.
            weighted_history = sum((rho ** (lag + 1)) * v for lag, v in enumerate(positive_history))
            denominator = 1.0 + sum(rho ** (lag + 1) for lag in range(horizon))
            candidate = max(x_t, (x_t + weighted_history) / denominator)
            feedback = candidate - x_t
            reason = "negative_temporal_valley"
        elif not self.config.enabled:
            reason = "disabled"
        elif not established:
            reason = "insufficient_state_history"
        elif not q2_continuous:
            reason = "q2_termination_signal"
        elif h_t <= 0.0:
            reason = "no_positive_state"

        # Non-recursive: store pre-SEE x_t and current q2, never candidate.
        self.score_history.append(x_t)
        self.score_history = self.score_history[-horizon:]
        self.q2_history.append(q2_t)
        self.q2_history = self.q2_history[-horizon:]

        prev1 = padded_scores[0] if horizon >= 1 else 0.0
        prev2 = padded_scores[1] if horizon >= 2 else 0.0
        weighted_history = float(sum((rho ** (lag + 1)) * value for lag, value in enumerate(positive_history)))
        return {
            "see_rho": rho,
            "see_horizon": horizon,
            "see_valley_ratio": tau,
            "see_q2_continuity": eta,
            "see_prev1_score": float(prev1),
            "see_prev2_score": float(prev2),
            "see_prev1_positive": float(max(prev1, 0.0)),
            "see_prev2_positive": float(max(prev2, 0.0)),
            "see_state_level": h_t,
            "see_valley_threshold": threshold,
            "see_valley_error": valley_error,
            "see_q2_prev": float(q2_prev),
            "see_q2_continuous": q2_continuous,
            "see_gate": gate,
            "see_positive_valley": int(gate and x_t >= 0.0),
            "see_negative_valley": int(gate and x_t < 0.0),
            "see_weighted_history": weighted_history,
            "see_candidate": float(candidate),
            "see_delta": float(feedback),
            "see_applied": int(feedback > 0.0),
            "see_cross_zero": int(x_t < 0.0 <= candidate),
            "see_reason": reason,
            "pre_see_score": x_t,
            "final_score": float(candidate),
        }


@dataclass
class PARSEEState:
    """Stateful paper-final PARSEE-VAD scorer."""

    config: PARSEEConfig
    correction_carry: PositiveCorrectionCarryState = field(init=False)
    see: StreamingEvidenceEscalationState = field(init=False)

    def __post_init__(self) -> None:
        self.correction_carry = PositiveCorrectionCarryState(self.config.correction_carry)
        self.see = StreamingEvidenceEscalationState(self.config.see)

    @classmethod
    def from_pipeline(cls, pipeline: dict[str, Any]) -> "PARSEEState":
        return cls(PARSEEConfig.from_pipeline(pipeline))

    def reset(self) -> None:
        self.correction_carry.reset()
        self.see.reset()

    def seed_history(
        self,
        pre_see_scores: list[float],
        previous_raw_par_delta: float = 0.0,
        q2_values: list[float] | None = None,
    ) -> None:
        """Restore temporal state for a clean interrupted-run continuation."""
        self.correction_carry.seed_previous_raw_par_delta(previous_raw_par_delta)
        self.see.seed_history(pre_see_scores, q2_values=q2_values)

    def step(
        self,
        *,
        q2: float,
        q3: float,
        p3: Any,
        p4: Any,
        p3_executed: bool,
        p4_executed: bool,
    ) -> dict[str, Any]:
        par = dimensionless_par(
            q2,
            q3,
            p3,
            p4,
            p3_executed=p3_executed,
            p4_executed=p4_executed,
            config=self.config.par,
        )
        carry = self.correction_carry.step(
            q2=float(q2),
            par_delta=float(par["par_delta"]),
            par_score=float(par["par_score"]),
        )
        see = self.see.step(float(carry["carry_score"]), float(q2))
        return {**par, **carry, **see}
