#!/usr/bin/env python3
"""
Step 3: Preprocess and Format Training Data
Combines code and documentation into a unified training dataset.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from utils.progress_tracker import PhaseTracker
from utils.data_utils import (
    OdooDataFormatter,
    collect_source_files,
    deduplicate_examples,
)

console = Console()


def process_source_files(
    source_dir: Path, formatter: OdooDataFormatter
) -> list[dict]:
    """Process all Odoo source files into training examples."""
    examples = []

    if not source_dir.exists():
        console.print(
            f"[yellow]Source dir not found: {source_dir}[/]"
        )
        return examples

    files = collect_source_files(str(source_dir))
    console.print(f"  Processing [bold]{len(files)}[/] source files...")

    for f in track(files, description="Processing source files"):
        try:
            file_examples = formatter.format_code_completion(
                file_path=f["path"],
                code=f["content"],
                module_name=f["module"],
            )
            examples.extend(file_examples)
        except Exception as e:
            pass  # Skip problematic files silently

    return examples


def load_doc_examples(processed_dir: Path) -> list[dict]:
    """Load previously processed documentation examples."""
    docs_file = processed_dir / "documentation_examples.jsonl"
    examples = []

    if docs_file.exists():
        with open(docs_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        console.print(
            f"  Loaded [bold]{len(examples)}[/] documentation examples"
        )
    else:
        console.print(
            "[yellow]No documentation examples found. "
            "Run 02_collect_odoo_docs.py first.[/]"
        )

    return examples


def split_dataset(
    examples: list[dict],
    val_ratio: float = 0.05,
    test_ratio: float = 0.02,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split dataset into train/validation/test sets."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_set = shuffled[:n_test]
    val_set = shuffled[n_test : n_test + n_val]
    train_set = shuffled[n_test + n_val :]

    return train_set, val_set, test_set


def save_dataset(
    examples: list[dict], output_path: Path, format: str = "jsonl"
):
    """Save dataset to disk in the specified format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    elif format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    console.print(
        f"  Saved {len(examples)} examples to {output_path}"
    )


def display_dataset_stats(
    train: list[dict],
    val: list[dict],
    test: list[dict],
):
    """Display comprehensive dataset statistics."""
    table = Table(
        title="Dataset Statistics",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Split", style="cyan", width=15)
    table.add_column("Examples", width=12)
    table.add_column("Avg Msg Length", width=15)
    table.add_column("Total Chars", width=15)

    for name, data in [("Train", train), ("Validation", val), ("Test", test)]:
        if not data:
            table.add_row(name, "0", "0", "0")
            continue

        total_chars = sum(
            sum(len(m["content"]) for m in ex["messages"])
            for ex in data
        )
        avg_len = total_chars // len(data)
        table.add_row(
            name,
            f"{len(data):,}",
            f"{avg_len:,}",
            f"{total_chars:,}",
        )

    console.print(table)

    # Message role distribution
    all_data = train + val + test
    if all_data:
        sample = all_data[:5]
        console.print("\n[bold]Sample prompt (first example):[/]")
        for msg in sample[0]["messages"]:
            role = msg["role"]
            preview = msg["content"][:200]
            if len(msg["content"]) > 200:
                preview += "..."
            console.print(f"  [{role}]: {preview}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess and format training data"
    )
    parser.add_argument(
        "--source-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "odoo_source"),
        help="Odoo source code directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Max token length for examples (default: 4096)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio (default: 0.05)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.02,
        help="Test split ratio (default: 0.02)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    phase_tracker = PhaseTracker(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )
    phase_tracker.start_phase("data_preprocessing")

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel(
            "[bold]Preprocessing Odoo Training Data[/]",
            title="Phase 2: Data Preprocessing",
            border_style="cyan",
        )
    )

    formatter = OdooDataFormatter(max_length=args.max_length)

    # ---- Process source code ----
    console.print("\n[cyan]Step 1: Processing source code...[/]")
    code_examples = process_source_files(source_dir, formatter)
    console.print(
        f"  Generated [bold]{len(code_examples)}[/] code examples"
    )

    # ---- Load documentation examples ----
    console.print("\n[cyan]Step 2: Loading documentation examples...[/]")
    doc_examples = load_doc_examples(output_dir)

    # ---- Combine and deduplicate ----
    console.print("\n[cyan]Step 3: Combining and deduplicating...[/]")
    all_examples = code_examples + doc_examples
    console.print(
        f"  Total before dedup: [bold]{len(all_examples)}[/]"
    )

    all_examples = deduplicate_examples(all_examples)
    console.print(
        f"  Total after dedup:  [bold]{len(all_examples)}[/]"
    )

    # ---- Filter by length ----
    console.print("\n[cyan]Step 4: Filtering by length...[/]")
    max_chars = args.max_length * 4  # rough chars-to-tokens ratio
    filtered = [
        ex
        for ex in all_examples
        if sum(len(m["content"]) for m in ex["messages"]) <= max_chars
        and sum(len(m["content"]) for m in ex["messages"]) >= 100
    ]
    console.print(
        f"  Kept {len(filtered)} / {len(all_examples)} examples "
        f"(filtered {len(all_examples) - len(filtered)})"
    )

    # ---- Split dataset ----
    console.print("\n[cyan]Step 5: Splitting dataset...[/]")
    train, val, test = split_dataset(
        filtered,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ---- Save ----
    console.print("\n[cyan]Step 6: Saving datasets...[/]")
    save_dataset(train, output_dir / "train.jsonl")
    save_dataset(val, output_dir / "val.jsonl")
    save_dataset(test, output_dir / "test.jsonl")

    # Save a combined file for reference
    save_dataset(filtered, output_dir / "all_data.jsonl")

    # ---- Stats ----
    console.print()
    display_dataset_stats(train, val, test)

    # Save preprocessing stats
    stats = {
        "code_examples": len(code_examples),
        "doc_examples": len(doc_examples),
        "total_before_dedup": len(code_examples) + len(doc_examples),
        "total_after_dedup": len(all_examples),
        "total_after_filter": len(filtered),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "max_length": args.max_length,
        "seed": args.seed,
    }
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    phase_tracker.end_phase("data_preprocessing")

    console.print(
        Panel(
            f"[bold green]Preprocessing Complete[/]\n\n"
            f"Train: {len(train):,} examples\n"
            f"Val:   {len(val):,} examples\n"
            f"Test:  {len(test):,} examples\n\n"
            f"Files saved to: {output_dir}",
            title="Preprocessing Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
