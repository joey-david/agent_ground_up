from __future__ import annotations

import argparse
import os
from typing import Any

from agent_ground_up.config import load_config, path, section

LORA_TARGETS = (
    r"^model\.language_model\.layers\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|"
    r"in_proj_z|in_proj_b|in_proj_a|out_proj)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and package the coding agent")
    parser.add_argument("stage", choices=("sft", "rl", "merge", "quantize"))
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", "config.yaml"))
    return parser.parse_args()


def quantization_config() -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def lora_config() -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )


def common_args(config: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": str(path(config, values["output"])),
        "learning_rate": values["learning_rate"],
        "max_steps": values["max_steps"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": values["gradient_accumulation_steps"],
        "gradient_checkpointing": True,
        "bf16": True,
        "logging_steps": 1,
        "save_steps": 25,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
    }


def run_sft(config: dict[str, Any]) -> None:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    values = section(config, "sft")
    dataset = load_dataset("json", data_files=str(path(config, values["data"])), split="train")
    trainer_config = SFTConfig(
        **common_args(config, values),
        num_train_epochs=values["epochs"],
        assistant_only_loss=True,
        max_length=None,
        group_by_length=True,
        model_init_kwargs={
            "quantization_config": quantization_config(),
            "dtype": "bfloat16",
            "trust_remote_code": False,
        },
    )
    trainer = SFTTrainer(
        model=values["model"],
        args=trainer_config,
        train_dataset=dataset,
        peft_config=lora_config(),
    )
    trainer.train()
    trainer.save_model(str(path(config, values["output"]) / "final"))


def run_rl(config: dict[str, Any]) -> None:
    from datasets import load_dataset
    from trl import GRPOConfig

    from agent_ground_up.loss import build_dapo_trainer
    from agent_ground_up.remote_env import RemoteCodingEnv, remote_reward

    values = section(config, "rl")
    os.environ["AGENT_CONFIG"] = str(config["_path"])
    dataset = load_dataset("json", data_files=str(path(config, values["data"])), split="train")
    trainer_config = GRPOConfig(
        **common_args(config, values),
        model_init_kwargs={
            "quantization_config": quantization_config(),
            "dtype": "bfloat16",
            "trust_remote_code": False,
        },
        use_vllm=True,
        vllm_mode="server",
        vllm_server_base_url=values["vllm_url"],
        loss_type="dapo",
        importance_sampling_level="sequence",
        scale_rewards="batch",
        mask_truncated_completions=True,
        beta=values["beta"],
        epsilon=values["epsilon_low"],
        epsilon_high=values["epsilon_high"],
        num_generations=values["num_generations"],
        generation_batch_size=values["num_generations"],
        max_completion_length=values["max_completion_length"],
        temperature=values["temperature"],
        top_p=values["top_p"],
        top_k=values["top_k"],
        log_completions=False,
    )
    trainer_class = build_dapo_trainer()
    trainer = trainer_class(
        model=values["model"],
        args=trainer_config,
        train_dataset=dataset,
        reward_funcs=remote_reward,
        environment_factory=RemoteCodingEnv,
        peft_config=lora_config(),
    )
    trainer.train()
    trainer.save_model(str(path(config, values["output"]) / "final"))


def merge_adapter(config: dict[str, Any]) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    values = section(config, "merge")
    output = path(config, values["output"])
    model = AutoModelForImageTextToText.from_pretrained(
        values["base_model"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    merged = PeftModel.from_pretrained(model, path(config, values["adapter"])).merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(values["base_model"], trust_remote_code=False).save_pretrained(
        output
    )


def quantize_w4a16(config: dict[str, Any]) -> None:
    """Package the merged model as Ampere-compatible GPTQ INT4 weights."""
    from datasets import load_dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier

    values = section(config, "quantize")
    dataset = load_dataset("json", data_files=str(path(config, values["data"])), split="train")
    dataset = dataset.select(range(min(len(dataset), values["samples"])))
    model = str(path(config, values["model"]))
    recipe = GPTQModifier(
        targets="Linear",
        scheme=values["scheme"],
        ignore=[r"re:.*lm_head", r"re:.*visual.*", r"re:.*vision.*"],
    )
    oneshot(
        model=model,
        processor=model,
        dataset=dataset,
        recipe=recipe,
        output_dir=str(path(config, values["output"])),
        num_calibration_samples=values["samples"],
        max_seq_length=values["max_seq_length"],
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stages = {"sft": run_sft, "rl": run_rl, "merge": merge_adapter, "quantize": quantize_w4a16}
    stages[args.stage](config)


if __name__ == "__main__":
    main()
