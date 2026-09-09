import math

from PIL import Image

from src.evaluation.frame_level import expand_anchor_scores
from src.evaluation.metrics import ap
from src.workflows.full_workflow import ControlledPropagationState, _resize_one, _semantic


def test_positive_only_semantic_fusion():
    out = _semantic(0.5, 1.0, -3.0)
    assert math.isclose(out["semantic_score"], 0.5 + math.tanh(1.0))
    assert out["p4_positive_delta"] == 0.0


def test_backward_hold_is_nonoverlapping():
    scores = expand_anchor_scores([0, 60, 120], [1.0, 2.0, 3.0], 125)
    assert scores[0] == 1.0
    assert all(v == 2.0 for v in scores[1:61])
    assert all(v == 3.0 for v in scores[61:])


def test_ap_ties_are_order_invariant():
    assert ap([1, 0], [0.5, 0.5]) == ap([0, 1], [0.5, 0.5]) == 0.5


def test_pixel_budget_never_upscales():
    image = Image.new("RGB", (100, 80))
    out = _resize_one(image, 65536)
    assert out.size == image.size


def test_propagation_reset():
    state = ControlledPropagationState()
    state.step(2.0)
    assert state.evidence > 0
    state.reset()
    assert state.evidence == 0.0 and state.previous_final == 0.0 and state.semantic_history == []
