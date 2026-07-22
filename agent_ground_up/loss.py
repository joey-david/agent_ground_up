from __future__ import annotations

from typing import Any

import torch


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Clipped sequence-ratio policy loss with DAPO token normalization."""
    mask = mask.to(logps.dtype)
    advantages = advantages.reshape(-1, 1)
    token_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_ratio = ((logps - old_logps) * mask).sum(dim=1, keepdim=True) / token_counts
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    per_token_loss = -surrogate.expand_as(logps)

    if sampling_ratio is not None:
        if sampling_ratio.dim() == 1:
            sampling_ratio = sampling_ratio.unsqueeze(1)
        per_token_loss = per_token_loss * sampling_ratio

    if beta:
        if ref_logps is None:
            raise ValueError("ref_logps are required when beta is non-zero")
        reverse_log_ratio = ref_logps - logps
        per_token_kl = reverse_log_ratio.exp() - reverse_log_ratio - 1.0
        per_token_loss = per_token_loss + beta * per_token_kl
    else:
        per_token_kl = torch.zeros_like(logps)

    denominator = (
        mask.sum() if normalizer is None else torch.as_tensor(normalizer, device=logps.device)
    )
    loss = (per_token_loss * mask).sum() / denominator.clamp_min(1.0)
    clipped = ((ratio < 1.0 - epsilon_low) | (ratio > 1.0 + epsilon_high)).to(logps.dtype)
    metrics = {
        "ratio": ratio.detach().mean(),
        "clip_fraction": clipped.detach().mean(),
        "kl": ((per_token_kl * mask).sum() / mask.sum().clamp_min(1.0)).detach(),
    }
    return loss, metrics


def build_dapo_trainer() -> type[Any]:
    from trl import GRPOTrainer

    class DAPOTrainer(GRPOTrainer):
        def _compute_loss(self, model: Any, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            mask = (
                completion_mask
                if "tool_mask" not in inputs
                else completion_mask * inputs["tool_mask"]
            )
            logps, _, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                completion_ids.size(1),
                compute_entropy=False,
                compute_aux_loss=False,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                spatial_shapes=inputs.get("spatial_shapes"),
                num_tiles=inputs.get("num_tiles"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                image_position_ids=inputs.get("image_position_ids"),
            )
            old_logps = inputs.get("old_per_token_logps")
            if old_logps is None:
                old_logps = logps.detach()
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss, metrics = dapo_loss(
                logps,
                old_logps,
                inputs["advantages"],
                mask,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                normalizer=normalizer,
                ref_logps=inputs.get("ref_per_token_logps"),
                beta=self.beta,
                sampling_ratio=inputs.get("importance_sampling_ratio"),
            )
            mode = "train" if model.training else "eval"
            for name, value in metrics.items():
                self._metrics[mode][f"custom_dapo/{name}"].append(
                    self.accelerator.gather(value).mean().item()
                )
            return loss

    return DAPOTrainer
