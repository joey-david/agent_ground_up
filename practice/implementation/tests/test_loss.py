import torch

from kernel.loss import completion_logps_from_logits, dapo_loss


def test_causal_logprob_gather_and_masked_policy_gradient() -> None:
    logits = torch.zeros((1, 4, 5), dtype=torch.float32)
    input_ids = torch.tensor([[0, 1, 2, 3]])
    logits[0, 1, 2] = 4.0
    logits[0, 2, 3] = 4.0
    logps = completion_logps_from_logits(logits, input_ids, completion_length=2)
    assert logps.shape == (1, 2)
    assert torch.all(logps > -0.2)

    current = torch.tensor([[-0.8, -0.8, -9.0]], requires_grad=True)
    old = torch.tensor([[-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1, 1, 0]])
    loss, metrics = dapo_loss(current, old, torch.tensor([1.0]), mask, epsilon_low=0.2, epsilon_high=0.28)
    loss.backward()
    assert current.grad is not None
    assert current.grad[0, 2] == 0
    assert 0 <= metrics["clip_fraction"] <= 1
