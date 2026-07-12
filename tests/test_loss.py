from __future__ import annotations

import torch

from agent_ground_up.loss import dapo_loss


def test_dapo_masks_padding_and_backpropagates() -> None:
    logps = torch.tensor([[-0.8, -0.8, -9.0], [-1.2, -1.2, -9.0]], requires_grad=True)
    old_logps = torch.full_like(logps, -1.0)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    loss, metrics = dapo_loss(
        logps,
        old_logps,
        advantages,
        mask,
        epsilon_low=0.2,
        epsilon_high=0.2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert logps.grad is not None
    assert torch.all(logps.grad[:, 2] == 0)
    assert 0 <= metrics["clip_fraction"] <= 1


def test_dapo_requires_reference_for_kl() -> None:
    values = torch.zeros((1, 2))
    try:
        dapo_loss(values, values, torch.ones(1), torch.ones_like(values), epsilon_low=0.2, epsilon_high=0.2, beta=0.1)
    except ValueError as error:
        assert "ref_logps" in str(error)
    else:
        raise AssertionError("Expected missing-reference error")
