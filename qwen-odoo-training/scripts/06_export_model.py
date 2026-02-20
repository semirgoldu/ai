#!/usr/bin/env python3
"""
Step 6: Export the fine-tuned model.
Merges LoRA weights and exports in various formats (safetensors, GGUF).
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()


def merge_lora_weights(model_dir: str, base_model_name: str, output_dir: str):
    """Merge LoRA adapter weights into the base model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    console.print("[cyan]Loading base model...[/]")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Use CPU for merging to save VRAM
        trust_remote_code=True,
    )

    console.print("[cyan]Loading LoRA adapter...[/]")
    model = PeftModel.from_pretrained(base_model, model_dir)

    console.print("[cyan]Merging weights...[/]")
    model = model.merge_and_unload()

    console.print(f"[cyan]Saving merged model to {output_dir}...[/]")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)

    console.print(f"[green]Merged model saved to {output_dir}[/]")

    # Calculate size
    total_size = sum(
        f.stat().st_size
        for f in output_path.rglob("*")
        if f.is_file()
    )
    console.print(
        f"  Total size: {total_size / (1024**3):.1f} GB"
    )

    return output_dir


def export_to_gguf(
    model_dir: str,
    output_dir: str,
    quantization: str = "q4_k_m",
):
    """Export model to GGUF format for llama.cpp / ollama."""
    import subprocess

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gguf_file = output_path / f"qwen-odoo-32b-{quantization}.gguf"

    console.print(
        f"[cyan]Converting to GGUF ({quantization})...[/]"
    )
    console.print(
        "[yellow]Note: This requires llama.cpp's convert script. "
        "Install from: https://github.com/ggerganov/llama.cpp[/]"
    )

    # Try using llama.cpp convert script
    convert_script = shutil.which("python3")
    llama_cpp_convert = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"

    if llama_cpp_convert.exists():
        try:
            cmd = [
                "python3",
                str(llama_cpp_convert),
                model_dir,
                "--outfile",
                str(gguf_file),
                "--outtype",
                quantization,
            ]
            subprocess.run(cmd, check=True)
            console.print(
                f"[green]GGUF exported to {gguf_file}[/]"
            )
        except subprocess.CalledProcessError as e:
            console.print(f"[red]GGUF conversion failed: {e}[/]")
    else:
        console.print(
            "[yellow]llama.cpp not found. To convert to GGUF:[/]\n"
            "  1. git clone https://github.com/ggerganov/llama.cpp\n"
            "  2. cd llama.cpp && pip install -r requirements.txt\n"
            f"  3. python convert_hf_to_gguf.py {model_dir} "
            f"--outfile {gguf_file} --outtype {quantization}"
        )

    return str(gguf_file)


def create_ollama_modelfile(
    model_dir: str, output_dir: str, gguf_name: str = ""
):
    """Create an Ollama Modelfile for easy deployment."""
    modelfile_content = f"""# Ollama Modelfile for Qwen-Odoo-32B
# Usage: ollama create qwen-odoo -f Modelfile

FROM {gguf_name if gguf_name else './qwen-odoo-32b-q4_k_m.gguf'}

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"

SYSTEM \"\"\"You are an expert Odoo developer with deep knowledge of the Odoo framework, its ORM, views, controllers, and module architecture. You write clean, efficient, and well-documented Odoo code following official best practices.\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER num_ctx 4096
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""

    modelfile_path = Path(output_dir) / "Modelfile"
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.write_text(modelfile_content)
    console.print(f"[green]Ollama Modelfile saved to {modelfile_path}[/]")
    return str(modelfile_path)


def create_model_card(
    model_dir: str,
    config: dict,
    eval_results: dict = None,
):
    """Create a model card (README) for the fine-tuned model."""
    card = f"""---
language:
  - en
  - python
tags:
  - odoo
  - code-generation
  - qwen2.5
  - lora
  - fine-tuned
base_model: {config['model']['name']}
datasets:
  - odoo-community-source
  - odoo-documentation
library_name: transformers
pipeline_tag: text-generation
---

# Qwen2.5-Coder-32B-Odoo

Fine-tuned version of {config['model']['name']} specialized for Odoo ERP development.

## Training Details

- **Base Model**: {config['model']['name']}
- **Method**: LoRA (rank={config['lora']['r']}, alpha={config['lora']['lora_alpha']})
- **Hardware**: {config['hardware']['num_gpus']}x {config['hardware']['gpu_type']}
- **Precision**: BF16
- **Epochs**: {config['training']['num_train_epochs']}
- **Batch Size**: {config['training']['per_device_train_batch_size']} x {config['hardware']['num_gpus']} x {config['training']['gradient_accumulation_steps']} = {config['training']['per_device_train_batch_size'] * config['hardware']['num_gpus'] * config['training']['gradient_accumulation_steps']}

## Capabilities

- Odoo model creation (Python ORM)
- View definitions (XML forms, trees, kanban)
- Business logic and workflows
- Security configuration (ACLs, record rules)
- Web controllers and API endpoints
- QWeb templates and reports
- Module architecture and best practices

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/qwen-odoo-32b")
tokenizer = AutoTokenizer.from_pretrained("path/to/qwen-odoo-32b")

messages = [
    {{"role": "system", "content": "You are an expert Odoo developer."}},
    {{"role": "user", "content": "Create a purchase approval workflow model in Odoo."}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

    card_path = Path(model_dir) / "README.md"
    card_path.write_text(card)
    console.print(f"[green]Model card saved to {card_path}[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned Qwen-Odoo model"
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "outputs" / "final_model"),
        help="Path to the fine-tuned LoRA model",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Training config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "exported"),
        help="Export output directory",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="Merge LoRA weights into base model",
    )
    parser.add_argument(
        "--gguf",
        action="store_true",
        help="Also export to GGUF format",
    )
    parser.add_argument(
        "--gguf-quantization",
        default="q4_k_m",
        choices=["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m"],
        help="GGUF quantization type (default: q4_k_m)",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Create Ollama Modelfile",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print(
        Panel(
            "[bold]Exporting Qwen-Odoo Model[/]",
            title="Phase 6: Export",
            border_style="cyan",
        )
    )

    from utils.progress_tracker import PhaseTracker

    phase_tracker = PhaseTracker(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )
    phase_tracker.start_phase("export")

    merged_dir = Path(args.output_dir) / "merged"

    # Merge LoRA weights
    if args.merge:
        merge_lora_weights(
            args.model_dir,
            config["model"]["name"],
            str(merged_dir),
        )

    # Export to GGUF
    gguf_name = ""
    if args.gguf:
        gguf_dir = Path(args.output_dir) / "gguf"
        gguf_name = export_to_gguf(
            str(merged_dir),
            str(gguf_dir),
            args.gguf_quantization,
        )

    # Create Ollama Modelfile
    if args.ollama:
        create_ollama_modelfile(
            str(merged_dir),
            args.output_dir,
            gguf_name,
        )

    # Create model card
    eval_file = PROJECT_ROOT / "outputs" / "logs" / "eval_results.json"
    eval_results = None
    if eval_file.exists():
        with open(eval_file) as f:
            eval_results = json.load(f)

    create_model_card(str(merged_dir), config, eval_results)

    phase_tracker.end_phase("export")

    console.print(
        Panel(
            f"[bold green]Export Complete[/]\n\n"
            f"Merged model: {merged_dir}\n"
            f"{'GGUF: ' + gguf_name if args.gguf else ''}\n"
            f"Export dir: {args.output_dir}",
            title="Export Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
