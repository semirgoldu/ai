#!/usr/bin/env python3
"""
Step 4: Fine-tune Qwen2.5-Coder-32B on Odoo data
Uses LoRA + DeepSpeed ZeRO-2 for 2x RTX Pro 6000 96GB.

Provides real-time progress tracking, ETA, GPU monitoring,
and detailed training metrics.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_environment(config: dict):
    """Configure environment variables for training."""
    num_gpus = config["hardware"]["num_gpus"]

    # Set visible GPUs
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(i) for i in range(num_gpus)
        )

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:512"
    )

    # TF32 for faster matmul on Ampere+
    if config["training"].get("tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    console.print("[green]Environment configured[/]")
    console.print(
        f"  CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}"
    )
    console.print(f"  PyTorch: {torch.__version__}")
    console.print(f"  CUDA: {torch.version.cuda}")
    console.print(
        f"  GPUs available: {torch.cuda.device_count()}"
    )
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        console.print(f"    GPU {i}: {name} ({mem:.0f} GB)")


def load_datasets(config: dict):
    """Load training and validation datasets."""
    from datasets import load_dataset

    data_config = config["data"]
    train_path = str(PROJECT_ROOT / data_config["train_file"])
    val_path = str(PROJECT_ROOT / data_config["val_file"])

    console.print(f"\n[cyan]Loading datasets...[/]")

    train_dataset = load_dataset(
        "json", data_files=train_path, split="train"
    )
    val_dataset = load_dataset(
        "json", data_files=val_path, split="train"
    )

    console.print(f"  Train: {len(train_dataset):,} examples")
    console.print(f"  Val:   {len(val_dataset):,} examples")

    return train_dataset, val_dataset


def load_model_and_tokenizer(config: dict):
    """Load the Qwen model with LoRA configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model_config = config["model"]
    lora_config = config["lora"]

    model_name = model_config["name"]
    console.print(f"\n[cyan]Loading model: {model_name}[/]")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine attention implementation
    attn_impl = model_config.get("attn_implementation", "sdpa")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        console.print("  [green]Flash Attention 2 available[/]")
    except ImportError:
        attn_impl = "sdpa"
        console.print(
            "  [yellow]Flash Attention not found, using SDPA[/]"
        )

    # Load model
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(
        model_config.get("torch_dtype", "bfloat16"), torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=model_config.get("trust_remote_code", True),
        use_cache=False,
        device_map=None,  # Let DeepSpeed handle distribution
    )

    console.print(
        f"  Model loaded: {model.num_parameters() / 1e9:.1f}B parameters"
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=lora_config["lora_dropout"],
        target_modules=lora_config["target_modules"],
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
        modules_to_save=lora_config.get("modules_to_save"),
    )

    model = get_peft_model(model, peft_config)

    trainable, total = model.get_nb_trainable_parameters()
    console.print(
        f"  LoRA applied: {trainable:,} trainable / "
        f"{total:,} total ({trainable / total * 100:.2f}%)"
    )

    if config["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=config["training"].get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}
            )
        )
        console.print("  [green]Gradient checkpointing enabled[/]")

    return model, tokenizer


class TrainingCallbacks:
    """Custom callbacks for progress tracking during training."""

    def __init__(self, config: dict):
        from utils.progress_tracker import TrainingProgressTracker
        from utils.gpu_monitor import GPUMonitor

        self.gpu_monitor = GPUMonitor(
            log_dir=str(PROJECT_ROOT / "outputs" / "logs")
        )
        self.config = config
        self.tracker = None
        self.phase_start = None

    def on_train_begin(self, args, state, **kwargs):
        from utils.progress_tracker import TrainingProgressTracker

        steps_per_epoch = state.max_steps // args.num_train_epochs
        self.tracker = TrainingProgressTracker(
            total_epochs=int(args.num_train_epochs),
            steps_per_epoch=steps_per_epoch,
            log_dir=str(PROJECT_ROOT / "outputs" / "logs"),
            log_interval=args.logging_steps,
        )
        self.tracker.start_training()
        self.gpu_monitor.start_background_logging()
        self.phase_start = time.time()

    def on_epoch_begin(self, args, state, **kwargs):
        if self.tracker:
            self.tracker.start_epoch(int(state.epoch))

    def on_log(self, args, state, logs=None, **kwargs):
        if self.tracker and logs:
            loss = logs.get("loss", logs.get("train_loss", 0))
            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm")

            # Get GPU memory
            gpu_mem = None
            if torch.cuda.is_available():
                gpu_mem = sum(
                    torch.cuda.memory_allocated(i) / (1024**3)
                    for i in range(torch.cuda.device_count())
                )

            self.tracker.log_step(
                step=int(state.global_step % self.tracker.steps_per_epoch)
                or self.tracker.steps_per_epoch,
                loss=loss,
                learning_rate=lr,
                grad_norm=grad_norm,
                gpu_mem_used=gpu_mem,
                extra_metrics={
                    k: v
                    for k, v in logs.items()
                    if k not in ("loss", "learning_rate", "grad_norm")
                },
            )

    def on_evaluate(self, args, state, metrics=None, **kwargs):
        if metrics:
            val_loss = metrics.get("eval_loss")
            console.print(
                f"\n  [cyan]Eval @ step {state.global_step}: "
                f"loss = {val_loss:.4f}[/]\n"
            )

    def on_train_end(self, args, state, **kwargs):
        if self.tracker:
            self.tracker.end_training()
        self.gpu_monitor.stop_background_logging()


def create_trainer(model, tokenizer, train_dataset, val_dataset, config):
    """Create the SFTTrainer with all configurations."""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from transformers.trainer_callback import TrainerCallback

    training_config = config["training"]
    output_config = config["output"]

    # Build training arguments
    training_args_dict = {
        "output_dir": str(PROJECT_ROOT / output_config["output_dir"]),
        "logging_dir": str(PROJECT_ROOT / output_config["logging_dir"]),
        "per_device_train_batch_size": training_config[
            "per_device_train_batch_size"
        ],
        "per_device_eval_batch_size": training_config[
            "per_device_eval_batch_size"
        ],
        "gradient_accumulation_steps": training_config[
            "gradient_accumulation_steps"
        ],
        "num_train_epochs": training_config["num_train_epochs"],
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "warmup_ratio": training_config["warmup_ratio"],
        "weight_decay": training_config["weight_decay"],
        "max_grad_norm": training_config["max_grad_norm"],
        "bf16": training_config.get("bf16", True),
        "fp16": training_config.get("fp16", False),
        "tf32": training_config.get("tf32", True),
        "gradient_checkpointing": training_config.get(
            "gradient_checkpointing", True
        ),
        "gradient_checkpointing_kwargs": training_config.get(
            "gradient_checkpointing_kwargs", {"use_reentrant": False}
        ),
        "logging_steps": training_config["logging_steps"],
        "logging_first_step": training_config.get(
            "logging_first_step", True
        ),
        "eval_strategy": training_config.get("eval_strategy", "steps"),
        "eval_steps": training_config.get("eval_steps", 200),
        "save_strategy": training_config.get("save_strategy", "steps"),
        "save_steps": training_config.get("save_steps", 500),
        "save_total_limit": training_config.get("save_total_limit", 3),
        "load_best_model_at_end": training_config.get(
            "load_best_model_at_end", True
        ),
        "metric_for_best_model": training_config.get(
            "metric_for_best_model", "eval_loss"
        ),
        "greater_is_better": training_config.get(
            "greater_is_better", False
        ),
        "seed": training_config.get("seed", 42),
        "dataloader_num_workers": training_config.get(
            "dataloader_num_workers", 4
        ),
        "dataloader_pin_memory": training_config.get(
            "dataloader_pin_memory", True
        ),
        "remove_unused_columns": False,
        "report_to": training_config.get("report_to", ["tensorboard"]),
        "run_name": training_config.get(
            "run_name", "qwen-32b-odoo-lora"
        ),
    }

    # Add DeepSpeed if enabled
    if config.get("deepspeed", {}).get("enabled", False):
        ds_config = str(
            PROJECT_ROOT / config["deepspeed"]["config_file"]
        )
        training_args_dict["deepspeed"] = ds_config
        console.print(
            f"  [green]DeepSpeed enabled: {ds_config}[/]"
        )

    if training_config.get("max_steps", -1) > 0:
        training_args_dict["max_steps"] = training_config["max_steps"]

    training_args = TrainingArguments(**training_args_dict)

    # Custom callback wrapper
    callbacks_handler = TrainingCallbacks(config)

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            callbacks_handler.on_train_begin(args, state, **kwargs)

        def on_epoch_begin(self, args, state, control, **kwargs):
            callbacks_handler.on_epoch_begin(args, state, **kwargs)

        def on_log(self, args, state, control, logs=None, **kwargs):
            callbacks_handler.on_log(args, state, logs=logs, **kwargs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            callbacks_handler.on_evaluate(
                args, state, metrics=metrics, **kwargs
            )

        def on_train_end(self, args, state, control, **kwargs):
            callbacks_handler.on_train_end(args, state, **kwargs)

    # Create SFTTrainer
    max_seq_length = config["data"].get("max_seq_length", 4096)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        packing=config["data"].get("packing", False),
        callbacks=[ProgressCallback()],
    )

    return trainer


def estimate_training_time(config: dict, num_train_examples: int):
    """Estimate training time based on configuration and hardware."""
    tc = config["training"]
    hw = config["hardware"]

    batch_size = (
        tc["per_device_train_batch_size"]
        * hw["num_gpus"]
        * tc["gradient_accumulation_steps"]
    )
    steps_per_epoch = num_train_examples // batch_size
    total_steps = steps_per_epoch * tc["num_train_epochs"]

    # Rough estimate: ~3-5 seconds per step for 32B model with LoRA on 2x96GB
    # with gradient checkpointing and bf16
    sec_per_step_low = 3.0
    sec_per_step_high = 6.0

    time_low = total_steps * sec_per_step_low
    time_high = total_steps * sec_per_step_high

    def fmt(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        if h > 24:
            d = h // 24
            h = h % 24
            return f"{d}d {h}h {m}m"
        return f"{h}h {m}m"

    console.print(
        Panel(
            f"[bold]Training Time Estimate[/]\n\n"
            f"Training examples:    {num_train_examples:,}\n"
            f"Effective batch size: {batch_size}\n"
            f"Steps per epoch:      {steps_per_epoch:,}\n"
            f"Total steps:          {total_steps:,}\n"
            f"Epochs:               {tc['num_train_epochs']}\n\n"
            f"Estimated time range: [bold]{fmt(time_low)} - {fmt(time_high)}[/]\n\n"
            f"(Based on ~{sec_per_step_low}-{sec_per_step_high}s/step "
            f"for 32B LoRA on {hw['num_gpus']}x {hw['gpu_type']})",
            title="Time Estimate",
            border_style="yellow",
        )
    )

    return {
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_time_low_hours": round(time_low / 3600, 1),
        "estimated_time_high_hours": round(time_high / 3600, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-Coder-32B on Odoo data"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate training time, don't train",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load everything but only train for 5 steps",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    console.print(
        Panel(
            f"[bold]Qwen2.5-Coder-32B Odoo Fine-Tuning[/]\n"
            f"Model: {config['model']['name']}\n"
            f"Hardware: {config['hardware']['num_gpus']}x "
            f"{config['hardware']['gpu_type']} "
            f"({config['hardware']['total_vram_gb']} GB total)",
            title="Phase 4: Training",
            border_style="cyan",
        )
    )

    # Setup environment
    setup_environment(config)

    # Initialize phase tracker
    from utils.progress_tracker import PhaseTracker
    from utils.gpu_monitor import GPUMonitor

    phase_tracker = PhaseTracker(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )

    # Display GPU status
    gpu_monitor = GPUMonitor(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )
    gpu_monitor.display_gpu_status()

    # Check VRAM
    sufficient, msg = gpu_monitor.check_vram_sufficient(required_gb=120)
    if not sufficient:
        console.print(f"\n[red]WARNING: {msg}[/]")
        console.print(
            "[yellow]Training may fail due to insufficient VRAM. "
            "Consider reducing batch size or LoRA rank.[/]"
        )
    else:
        console.print(f"\n[green]{msg}[/]")

    # Estimate batch size
    batch_info = gpu_monitor.estimate_batch_size(model_size_b=32.0)
    console.print(
        f"\n[cyan]Recommended batch size per GPU: "
        f"{batch_info['recommended_batch_size_per_gpu']}[/]"
    )
    console.print(
        f"[cyan]Recommended gradient accumulation: "
        f"{batch_info['recommended_gradient_accumulation']}[/]"
    )

    # Load datasets
    train_dataset, val_dataset = load_datasets(config)

    # Estimate training time
    time_est = estimate_training_time(config, len(train_dataset))

    if args.estimate_only:
        console.print("\n[yellow]Estimate-only mode, exiting.[/]")
        return

    # Dry run adjustment
    if args.dry_run:
        config["training"]["max_steps"] = 5
        config["training"]["eval_steps"] = 3
        config["training"]["save_steps"] = 5
        config["training"]["logging_steps"] = 1
        console.print(
            "\n[yellow]DRY RUN MODE: Training for 5 steps only[/]\n"
        )

    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    gpu_monitor.display_gpu_status()

    # Create trainer
    console.print("\n[cyan]Creating trainer...[/]")
    trainer = create_trainer(
        model, tokenizer, train_dataset, val_dataset, config
    )

    # Start training
    phase_tracker.start_phase("training")
    console.print("\n[bold green]Starting training...[/]\n")

    try:
        result = trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        )

        # Save final model
        console.print("\n[cyan]Saving final model...[/]")
        final_dir = str(
            PROJECT_ROOT / config["output"]["final_model_dir"]
        )
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Save training results
        results_file = (
            PROJECT_ROOT / "outputs" / "logs" / "training_results.json"
        )
        with open(results_file, "w") as f:
            json.dump(
                {
                    "train_runtime": result.metrics.get(
                        "train_runtime"
                    ),
                    "train_samples_per_second": result.metrics.get(
                        "train_samples_per_second"
                    ),
                    "train_steps_per_second": result.metrics.get(
                        "train_steps_per_second"
                    ),
                    "total_steps": result.global_step,
                    "train_loss": result.metrics.get("train_loss"),
                    "time_estimate": time_est,
                },
                f,
                indent=2,
            )

        phase_tracker.end_phase("training")

        console.print(
            Panel(
                f"[bold green]Training Complete![/]\n\n"
                f"Final loss: {result.metrics.get('train_loss', 'N/A')}\n"
                f"Total steps: {result.global_step}\n"
                f"Runtime: {result.metrics.get('train_runtime', 0) / 3600:.1f} hours\n"
                f"Model saved to: {final_dir}",
                title="Training Complete",
                border_style="bold green",
            )
        )

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Training interrupted by user. "
            "Saving checkpoint...[/]"
        )
        trainer.save_model(
            str(
                PROJECT_ROOT
                / config["output"]["output_dir"]
                / "interrupted"
            )
        )
        phase_tracker.end_phase("training", status="interrupted")

    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/]")
        phase_tracker.end_phase("training", status="failed")
        raise


if __name__ == "__main__":
    main()
