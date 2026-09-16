from PIL import Image

from src.evaluation.frame_level import expand_anchor_scores
from src.evaluation.metrics import ap
from src.workflows.full_workflow import _resize_one
from src.workflows.scoring import (
    CorrectionCarryConfig,
    PARConfig,
    PARSEEState,
    PositiveCorrectionCarryState,
    SEEConfig,
    StreamingEvidenceEscalationState,
    dimensionless_par,
)


def test_q3_negative_suppresses_positive_q2_only():
    cfg = PARConfig(alpha=0.75)
    out = dimensionless_par(
        2.0,
        -2.0,
        None,
        None,
        p3_executed=False,
        p4_executed=False,
        config=cfg,
    )
    assert out["q3_evidence"] < 0.0
    assert out["par_score"] < 2.0

    out_negative_q2 = dimensionless_par(
        -2.0,
        -2.0,
        None,
        None,
        p3_executed=False,
        p4_executed=False,
        config=cfg,
    )
    assert out_negative_q2["par_delta"] == 0.0
    assert out_negative_q2["par_score"] == -2.0


def test_signed_p3_can_reduce_current_window_fusion():
    cfg = PARConfig(alpha=0.75)
    out = dimensionless_par(
        2.0,
        1.0,
        -3.0,
        None,
        p3_executed=True,
        p4_executed=False,
        config=cfg,
    )
    assert out["p3_evidence"] < 0.0
    assert out["par_score"] < 2.0


def test_correction_carry_is_one_step_and_non_recursive():
    state = PositiveCorrectionCarryState(CorrectionCarryConfig(enabled=True, rho=0.5))

    first = state.step(q2=2.0, par_delta=1.0, par_score=3.0)
    assert first["carry_rescue_delta"] == 0.0

    second = state.step(q2=2.0, par_delta=-0.8, par_score=1.2)
    assert second["carry_rescue_delta"] == 0.5
    assert second["carry_score"] == 1.7
    assert second["carry_score"] <= 2.0

    # The inherited +0.5 rescue is not stored. The raw -0.8 correction is the
    # only state passed to the next window, so no positive carry remains.
    third = state.step(q2=2.0, par_delta=-0.8, par_score=1.2)
    assert third["carry_prev_par_delta"] == -0.8
    assert third["carry_rescue_delta"] == 0.0


def test_see_uses_finite_pre_see_history_without_recursive_rescue():
    state = StreamingEvidenceEscalationState(
        SEEConfig(enabled=True, rho=0.5, horizon=2, valley_ratio=0.2, q2_continuity=0.5)
    )

    state.step(score=2.0, q2=2.0)
    state.step(score=2.0, q2=2.0)
    rescued = state.step(score=0.1, q2=2.0)
    assert rescued["see_applied"] == 1
    assert rescued["final_score"] == 0.4

    # The state stores 0.1 (pre-SEE), not the rescued 0.4. Therefore the next
    # state level is mean([2.0, 0.1]) = 1.05.
    next_step = state.step(score=0.1, q2=2.0)
    assert abs(next_step["see_state_level"] - 1.05) < 1e-12

    # After another raw 0.1 enters the H=2 delay line, the old 2.0 evidence has
    # fully left the finite horizon.
    final_step = state.step(score=0.1, q2=2.0)
    assert abs(final_step["see_state_level"] - 0.1) < 1e-12
    assert final_step["see_applied"] == 0


def test_parsee_state_reset_clears_temporal_state():
    state = PARSEEState.from_pipeline({})
    state.step(q2=2.0, q3=1.0, p3=2.0, p4=None, p3_executed=True, p4_executed=False)
    assert state.correction_carry.previous_raw_par_delta > 0.0
    assert state.see.score_history

    state.reset()
    assert state.correction_carry.previous_raw_par_delta == 0.0
    assert state.see.score_history == []
    assert state.see.q2_history == []


def test_completed_interval_alignment_is_nonoverlapping_backward_hold():
    scores = expand_anchor_scores([0, 60, 120], [1.0, 2.0, 3.0], 125, alignment="completed")
    assert scores[0] == 1.0
    assert all(v == 2.0 for v in scores[1:61])
    assert all(v == 3.0 for v in scores[61:])


def test_availability_alignment_never_uses_score_before_anchor():
    scores = expand_anchor_scores([2, 5], [1.0, 2.0], 8, alignment="availability")
    assert scores[:2] == [0.0, 0.0]
    assert scores[2:5] == [1.0, 1.0, 1.0]
    assert scores[5:] == [2.0, 2.0, 2.0]


def test_ap_ties_are_order_invariant():
    assert ap([1, 0], [0.5, 0.5]) == ap([0, 1], [0.5, 0.5]) == 0.5


def test_pixel_budget_never_upscales():
    image = Image.new("RGB", (100, 80))
    out = _resize_one(image, 65536)
    assert out.size == image.size
