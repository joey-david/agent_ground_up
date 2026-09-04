from __future__ import annotations

import math

import torch

from agent_ground_up.loss import completion_logps_from_logits, dapo_loss


def test_completion_logps_use_causal_shift_and_sampled_tokens() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[0, 1, 2, 1]])

    logps = completion_logps_from_logits(logits, input_ids, completion_length=2)
    expected = logits[:, 1:3, :].log_softmax(dim=-1).gather(
        -1, input_ids[:, 2:].unsqueeze(-1)
    ).squeeze(-1)

    assert logps.shape == (1, 2)
    assert torch.allclose(logps, expected)


def test_dapo_asymmetric_clipping_for_positive_and_negative_advantages() -> None:
    # Sequence 0 moved too far upward: positive advantage is capped at 1 + eps_high.
    # Sequence 1 moved too far downward: negative advantage uses the lower trust-region bound.
    logps = torch.tensor([[math.log(2.0)], [math.log(0.5)]])
    old_logps = torch.zeros_like(logps)
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.ones_like(logps)

    loss, metrics = dapo_loss(
        logps,
        old_logps,
        advantages,
        mask,
        epsilon_low=0.2,
        epsilon_high=0.2,
    )

    # losses are -1.2 and +0.8, globally normalized over two active tokens
    assert torch.allclose(loss, torch.tensor(-0.2), atol=1e-6)
    assert torch.allclose(metrics["clip_fraction"], torch.tensor(1.0))


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


def test_dapo_gradient_flows_only_through_active_sampled_token_logps() -> None:
    logps = torch.zeros((1, 2), requires_grad=True)
    old_logps = torch.zeros_like(logps).detach()
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[1.0, 0.0]])

    loss, _ = dapo_loss(
        logps,
        old_logps,
        advantages,
        mask,
        epsilon_low=0.2,
        epsilon_high=0.28,
    )
    loss.backward()

    assert logps.grad is not None
    assert logps.grad[0, 0] < 0
    assert logps.grad[0, 1] == 0


def test_dapo_requires_reference_for_kl() -> None:
    values = torch.zeros((1, 2))
    try:
        dapo_loss(
            values,
            values,
            torch.ones(1),
            torch.ones_like(values),
            epsilon_low=0.2,
            epsilon_high=0.2,
            beta=0.1,
        )
    except ValueError as error:
        assert "ref_logps" in str(error)
    else:
        raise AssertionError("Expected missing-reference error")
