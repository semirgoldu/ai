#!/usr/bin/env python3
"""
Step 5: Evaluate the fine-tuned Qwen-Odoo model.
Runs inference on test set and computes metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

console = Console()


def load_model_for_inference(model_dir: str, config: dict):
    """Load the fine-tuned model for evaluation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model_config = config["model"]

    console.print(f"[cyan]Loading tokenizer...[/]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    console.print(f"[cyan]Loading model from {model_dir}...[/]")

    # Check if this is a LoRA adapter or full model
    adapter_config = Path(model_dir) / "adapter_config.json"

    if adapter_config.exists():
        # Load base model + LoRA adapter
        console.print("  Loading base model + LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()
        console.print("  [green]LoRA adapter merged[/]")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    console.print(f"  [green]Model loaded on {model.device}[/]")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Generate a response from the model."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_on_test_set(
    model,
    tokenizer,
    test_file: str,
    max_samples: int = 100,
) -> dict:
    """Evaluate model on the test set."""
    from rouge_score import rouge_scorer
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    # Load test data
    examples = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    if max_samples and len(examples) > max_samples:
        import random

        random.seed(42)
        examples = random.sample(examples, max_samples)

    console.print(f"  Evaluating on {len(examples)} test examples...")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    results = []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for ex in track(examples, description="Evaluating"):
        messages = ex["messages"]

        # Input is system + user messages; reference is assistant response
        input_msgs = [
            m for m in messages if m["role"] in ("system", "user")
        ]
        reference = next(
            (m["content"] for m in messages if m["role"] == "assistant"),
            "",
        )

        try:
            prediction = generate_response(
                model, tokenizer, input_msgs, max_new_tokens=1024
            )
        except Exception as e:
            prediction = f"ERROR: {e}"

        # Compute ROUGE
        scores = scorer.score(reference, prediction)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

        results.append(
            {
                "input": input_msgs[-1]["content"][:200],
                "reference": reference[:500],
                "prediction": prediction[:500],
                "rouge1": scores["rouge1"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )

    # Aggregate metrics
    metrics = {}
    for key in rouge_scores:
        values = rouge_scores[key]
        metrics[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return {"metrics": metrics, "results": results}


def evaluate_odoo_tasks(model, tokenizer) -> list[dict]:
    """Evaluate on specific Odoo coding tasks."""
    tasks = [
        {
            "name": "Create a basic Odoo model",
            "prompt": (
                "Create an Odoo model called 'library.book' with fields: "
                "name (Char, required), author (Char), isbn (Char, unique), "
                "date_published (Date), and price (Float). "
                "Include a computed field 'age_years' that calculates "
                "how old the book is."
            ),
            "check_keywords": [
                "models.Model",
                "_name",
                "library.book",
                "fields.Char",
                "fields.Date",
                "fields.Float",
                "@api.depends",
                "compute",
            ],
        },
        {
            "name": "Write an Odoo form view",
            "prompt": (
                "Write an Odoo XML form view for the 'library.book' model. "
                "Include a header with status bar, a sheet with groups "
                "for basic info and details, and a notebook with pages "
                "for description and reviews."
            ),
            "check_keywords": [
                "<record",
                "ir.ui.view",
                "<form",
                "<sheet>",
                "<group>",
                "<notebook>",
                "<page",
                "<header>",
            ],
        },
        {
            "name": "Implement a business method",
            "prompt": (
                "Write an Odoo method called 'action_borrow_book' for "
                "a 'library.book' model that: checks if the book is "
                "available, creates a borrowing record, updates the book "
                "status, and sends a notification. Include proper error "
                "handling with UserError."
            ),
            "check_keywords": [
                "def action_borrow_book",
                "self.ensure_one()",
                "UserError",
                "self.env[",
                "create(",
                "write(",
            ],
        },
        {
            "name": "Create access control",
            "prompt": (
                "Write the ir.model.access.csv file for a library module "
                "with models library.book and library.borrowing. "
                "Create two groups: library_user (read only) and "
                "library_manager (full access)."
            ),
            "check_keywords": [
                "id,name,model_id",
                "perm_read",
                "perm_write",
                "perm_create",
                "perm_unlink",
                "library_user",
                "library_manager",
            ],
        },
        {
            "name": "Write a controller",
            "prompt": (
                "Write an Odoo HTTP controller with a JSON endpoint "
                "'/api/books' that returns a list of available books "
                "with pagination support, and an HTTP endpoint "
                "'/library/catalog' that renders a website page."
            ),
            "check_keywords": [
                "http.Controller",
                "@http.route",
                "type='json'",
                "type='http'",
                "request.env",
                "request.render",
            ],
        },
    ]

    results = []
    for task in tasks:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Odoo developer. Write clean, "
                    "production-ready code following Odoo best practices."
                ),
            },
            {"role": "user", "content": task["prompt"]},
        ]

        console.print(f"\n  [cyan]Task: {task['name']}[/]")
        try:
            response = generate_response(
                model, tokenizer, messages, max_new_tokens=1024
            )

            # Check for keywords
            keywords_found = sum(
                1
                for kw in task["check_keywords"]
                if kw.lower() in response.lower()
            )
            score = keywords_found / len(task["check_keywords"])

            results.append(
                {
                    "task": task["name"],
                    "score": score,
                    "keywords_found": keywords_found,
                    "keywords_total": len(task["check_keywords"]),
                    "response_preview": response[:500],
                }
            )

            status = (
                "[green]PASS[/]"
                if score >= 0.6
                else "[yellow]PARTIAL[/]"
                if score >= 0.3
                else "[red]FAIL[/]"
            )
            console.print(
                f"    {status} ({keywords_found}/"
                f"{len(task['check_keywords'])} keywords)"
            )

        except Exception as e:
            results.append(
                {
                    "task": task["name"],
                    "score": 0,
                    "error": str(e),
                }
            )
            console.print(f"    [red]ERROR: {e}[/]")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen-Odoo model"
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "outputs" / "final_model"),
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "training_config.yaml"),
        help="Training config for model name reference",
    )
    parser.add_argument(
        "--test-file",
        default=str(PROJECT_ROOT / "data" / "processed" / "test.jsonl"),
        help="Path to test dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max test samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--skip-test-set",
        action="store_true",
        help="Skip test set evaluation, only run Odoo tasks",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    console.print(
        Panel(
            f"[bold]Evaluating Qwen-Odoo Model[/]\n"
            f"Model: {args.model_dir}",
            title="Phase 5: Evaluation",
            border_style="cyan",
        )
    )

    from utils.progress_tracker import PhaseTracker

    phase_tracker = PhaseTracker(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )
    phase_tracker.start_phase("evaluation")

    # Load model
    model, tokenizer = load_model_for_inference(args.model_dir, config)

    all_results = {}

    # Test set evaluation
    if not args.skip_test_set and Path(args.test_file).exists():
        console.print("\n[bold cyan]Test Set Evaluation[/]")
        test_results = evaluate_on_test_set(
            model, tokenizer, args.test_file, args.max_samples
        )
        all_results["test_set"] = test_results

        # Display metrics
        table = Table(title="ROUGE Scores")
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", width=10)
        table.add_column("Min", width=10)
        table.add_column("Max", width=10)

        for metric, values in test_results["metrics"].items():
            table.add_row(
                metric,
                f"{values['mean']:.4f}",
                f"{values['min']:.4f}",
                f"{values['max']:.4f}",
            )
        console.print(table)

    # Odoo-specific tasks
    console.print("\n[bold cyan]Odoo Task Evaluation[/]")
    odoo_results = evaluate_odoo_tasks(model, tokenizer)
    all_results["odoo_tasks"] = odoo_results

    # Summary table
    table = Table(title="Odoo Task Results")
    table.add_column("Task", style="cyan", width=30)
    table.add_column("Score", width=10)
    table.add_column("Keywords", width=12)

    avg_score = 0
    for r in odoo_results:
        score = r.get("score", 0)
        avg_score += score
        status_color = (
            "green" if score >= 0.6 else "yellow" if score >= 0.3 else "red"
        )
        table.add_row(
            r["task"],
            f"[{status_color}]{score:.0%}[/]",
            f"{r.get('keywords_found', 0)}/{r.get('keywords_total', 0)}",
        )

    avg_score /= max(len(odoo_results), 1)
    table.add_row(
        "[bold]Average[/]", f"[bold]{avg_score:.0%}[/]", ""
    )
    console.print(table)

    # Save results
    results_file = PROJECT_ROOT / "outputs" / "logs" / "eval_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    phase_tracker.end_phase("evaluation")

    console.print(
        Panel(
            f"[bold green]Evaluation Complete[/]\n\n"
            f"Odoo Task Average: {avg_score:.0%}\n"
            f"Results saved to: {results_file}",
            title="Evaluation Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
