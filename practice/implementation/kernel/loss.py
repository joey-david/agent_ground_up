from __future__ import annotations

import torch


def completion_logps_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_length: int,
) -> torch.Tensor:
    raise NotImplementedError


def dapo_loss(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float,
    normalizer: torch.Tensor | float | None = None,
    ref_logps: torch.Tensor | None = None,
    beta: float = 0.0,
    sampling_ratio: torch.Tensor | None = None,
    ratio_level: str = "token",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raise NotImplementedError
