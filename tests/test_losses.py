import torch

from mpsio_eco.losses import (
    CenterMarginLoss,
    SeparationIndexLoss,
    SupervisedContrastiveLoss,
    legacy_normalized_squared_separation_index,
)


def test_paper_separation_index_accepts_sequence_embeddings_and_backpropagates():
    torch.manual_seed(7)
    z = torch.randn(12, 5, 8, requires_grad=True)
    y = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    loss = SeparationIndexLoss()(z, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_separation_index_returns_differentiable_zero_for_single_eligible_class():
    z = torch.randn(4, 8, requires_grad=True)
    y = torch.zeros(4, dtype=torch.long)
    loss = SeparationIndexLoss()(z, y)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None


def test_legacy_loss_is_finite_and_distinct_utility():
    z = torch.randn(12, 8, requires_grad=True)
    y = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    loss = legacy_normalized_squared_separation_index(z, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None


def test_external_losses_are_finite():
    z = torch.randn(12, 16, requires_grad=True)
    y = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    loss = SupervisedContrastiveLoss()(z, y) + CenterMarginLoss()(z, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None
