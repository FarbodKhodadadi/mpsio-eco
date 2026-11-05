import torch
import pytest

from mpsio_eco.models import build_model, count_trainable_parameters


@pytest.mark.parametrize("name", ["alpha", "alpha_ce", "beta", "gamma", "delta", "epsilon"])
def test_model_forward_outputs_expected_shapes(name):
    model = build_model(name, num_classes=5)
    x = torch.randn(3, 1, 512)
    logits, embedding = model(x, return_embedding=True)
    assert logits.shape == (3, 5)
    assert embedding.shape == (3, 64)


def test_alpha_parameter_count_matches_manuscript_for_11_classes():
    model = build_model("alpha", num_classes=11)
    assert count_trainable_parameters(model) == 110_619


def test_model_accepts_channel_last_input():
    model = build_model("alpha", num_classes=2)
    x = torch.randn(4, 512, 1)
    logits = model(x)
    assert logits.shape == (4, 2)
