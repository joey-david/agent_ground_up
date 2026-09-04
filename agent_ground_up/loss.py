from __future__ import annotations

from typing import Any

import torch


def completion_logps_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    completion_length: int,
) -> torch.Tensor:
    """Turn causal-LM logits into log p(token_t | tokens_<t>) for completion tokens.

    This is intentionally explicit for the teaching path: shift the causal predictions by one,
    take log-softmax over the vocabulary, then gather the probability assigned to the token that
    was actually sampled.
    """
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [B,T,V] and input_ids [B,T]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share batch/sequence dimensions")
    if completion_length < 1 or completion_length >= input_ids.shape[1]:
        raise ValueError("completion_length must leave at least one conditioning token")

    # Logit at position t predicts token t+1. The final C completion tokens are therefore
    # predicted by the C logits immediately preceding them.
    completion_logits = logits[:, -completion_length - 1 : -1, :].float()
    completion_ids = input_ids[:, -completion_length:]
    log_probs = completion_logits.log_softmax(dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


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
    """Manual clipped DAPO/GRPO policy objective with global active-token normalization.

    Gradient path:
        model logits -> chosen-token log p -> pi_theta/pi_old -> clipped advantage -> loss.

    `ratio_level="token"` is the direct DAPO-style teaching path. `"sequence"` remains available
    for experiments that average the log-ratio over a whole completion before clipping.
    """
    if logps.shape != old_logps.shape or logps.shape != mask.shape:
        raise ValueError("logps, old_logps, and mask must have the same [B,T] shape")
    if ratio_level not in {"token", "sequence"}:
        raise ValueError("ratio_level must be 'token' or 'sequence'")

    mask = mask.to(logps.dtype)
    advantages = advantages.reshape(-1, 1).to(logps.dtype)
    log_ratio_per_token = logps - old_logps

    if ratio_level == "token":
        log_ratio = log_ratio_per_token
    else:
        token_counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        log_ratio = (log_ratio_per_token * mask).sum(dim=1, keepdim=True) / token_counts

    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    per_token_loss = -surrogate

    if sampling_ratio is not None:
        if sampling_ratio.dim() == 1:
            sampling_ratio = sampling_ratio.unsqueeze(1)
        per_token_loss = per_token_loss * sampling_ratio

    if beta:
        if ref_logps is None:
            raise ValueError("ref_logps are required when beta is non-zero")
        # Sampled reverse-KL estimator used by GRPO-family objectives.
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
    """Thin TRL bridge: TRL handles rollout/batching; our code owns the LM forward and loss."""
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

            model_inputs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            for key in (
                "pixel_values",
                "image_grid_thw",
                "num_images",
                "pixel_attention_mask",
                "spatial_shapes",
                "num_tiles",
                "image_sizes",
                "token_type_ids",
                "mm_token_type_ids",
                "image_position_ids",
            ):
                if key in inputs:
                    model_inputs[key] = inputs[key]

            # No hidden trainer helper here: run the causal LM, then explicitly recover the
            # log-probability assigned to every sampled completion token.
            outputs = model(**model_inputs)
            logps = completion_logps_from_logits(outputs.logits, input_ids, completion_ids.size(1))
            old_logps = inputs.get("old_per_token_logps")
            if old_logps is None:
                old_logps = logps.detach()

            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            ratio_level = getattr(
                self,
                "importance_sampling_level",
                getattr(self.args, "importance_sampling_level", "token"),
            )
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
                ratio_level=ratio_level,
            )
            mode = "train" if model.training else "eval"
            for name, value in metrics.items():
                self._metrics[mode][f"custom_dapo/{name}"].append(
                    self.accelerator.gather(value).mean().item()
                )
            return loss

    return DAPOTrainer
