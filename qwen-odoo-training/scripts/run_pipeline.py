#!/usr/bin/env python3
"""
Qwen-Odoo Training Pipeline Runner
====================================
Orchestrates the full training pipeline with progress tracking.

Usage:
    python run_pipeline.py                    # Run all phases
    python run_pipeline.py --start-from train # Skip to training
    python run_pipeline.py --dry-run          # Quick test run
    python run_pipeline.py --estimate         # Only estimate time
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from utils.progress_tracker import PhaseTracker
from utils.gpu_monitor import GPUMonitor

console = Console()

PHASES = [
    {
        "key": "data_collection",
        "name": "Data Collection",
        "script": "01_collect_odoo_source.py",
        "description": "Clone Odoo repos and collect source files",
    },
    {
        "key": "documentation",
        "name": "Documentation Processing",
        "script": "02_collect_odoo_docs.py",
        "description": "Process Odoo documentation into training examples",
    },
    {
        "key": "data_preprocessing",
        "name": "Data Preprocessing",
        "script": "03_preprocess_data.py",
        "description": "Format, deduplicate, and split training data",
    },
    {
        "key": "training",
        "name": "Model Training",
        "script": "04_train.py",
        "description": "Fine-tune Qwen2.5-Coder-32B with LoRA",
    },
    {
        "key": "evaluation",
        "name": "Evaluation",
        "script": "05_evaluate.py",
        "description": "Evaluate the fine-tuned model",
    },
    {
        "key": "export",
        "name": "Model Export",
        "script": "06_export_model.py",
        "description": "Merge LoRA weights and export model",
    },
]


def display_pipeline_plan():
    """Display the full pipeline plan."""
    table = Table(
        title="Qwen-Odoo Training Pipeline",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="cyan", width=3)
    table.add_column("Phase", width=25)
    table.add_column("Script", width=28)
    table.add_column("Description", width=45)

    for i, phase in enumerate(PHASES, 1):
        table.add_row(
            str(i),
            phase["name"],
            phase["script"],
            phase["description"],
        )

    console.print(table)


def run_phase(
    phase: dict,
    extra_args: list[str] = None,
    dry_run: bool = False,
) -> tuple[bool, float]:
    """Run a single pipeline phase. Returns (success, duration)."""
    script = PROJECT_ROOT / "scripts" / phase["script"]

    if not script.exists():
        console.print(
            f"[red]Script not found: {script}[/]"
        )
        return False, 0

    console.print(
        Panel(
            f"[bold]{phase['name']}[/]\n{phase['description']}",
            title=f"Phase: {phase['key']}",
            border_style="cyan",
        )
    )

    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    if dry_run and phase["key"] == "training":
        cmd.append("--dry-run")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=86400,  # 24 hour timeout
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            console.print(
                f"\n[green]Phase '{phase['name']}' completed "
                f"({duration / 60:.1f} min)[/]\n"
            )
            return True, duration
        else:
            console.print(
                f"\n[red]Phase '{phase['name']}' failed "
                f"(exit code {result.returncode})[/]\n"
            )
            return False, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        console.print(
            f"\n[red]Phase '{phase['name']}' timed out[/]\n"
        )
        return False, duration
    except KeyboardInterrupt:
        duration = time.time() - start_time
        console.print(
            f"\n[yellow]Phase '{phase['name']}' interrupted by user[/]\n"
        )
        return False, duration


def preflight_checks():
    """Run pre-flight checks before starting the pipeline."""
    console.print("[bold cyan]Running pre-flight checks...[/]\n")
    checks = []

    # Python version
    py_ver = sys.version_info
    checks.append(
        (
            "Python >= 3.10",
            py_ver >= (3, 10),
            f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        )
    )

    # PyTorch
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        checks.append(("PyTorch installed", True, torch.__version__))
        checks.append(("CUDA available", cuda_available, str(cuda_available)))
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            checks.append(("GPU count >= 2", num_gpus >= 2, str(num_gpus)))
            for i in range(num_gpus):
                name = torch.cuda.get_device_name(i)
                mem = (
                    torch.cuda.get_device_properties(i).total_mem
                    / (1024**3)
                )
                checks.append(
                    (f"GPU {i}", True, f"{name} ({mem:.0f} GB)")
                )
    except ImportError:
        checks.append(("PyTorch installed", False, "NOT FOUND"))

    # Key packages
    for pkg in [
        "transformers",
        "peft",
        "trl",
        "datasets",
        "accelerate",
        "deepspeed",
    ]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            checks.append((f"{pkg}", True, ver))
        except ImportError:
            checks.append((f"{pkg}", False, "NOT INSTALLED"))

    # Git
    try:
        import shutil

        git_path = shutil.which("git")
        checks.append(("git", git_path is not None, git_path or "NOT FOUND"))
    except Exception:
        checks.append(("git", False, "NOT FOUND"))

    # Disk space
    import shutil

    total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = free / (1024**3)
    checks.append(
        (
            "Disk space (>100GB free)",
            free_gb > 100,
            f"{free_gb:.0f} GB free",
        )
    )

    # Display results
    table = Table(title="Pre-flight Checks", show_header=True)
    table.add_column("Check", style="cyan", width=25)
    table.add_column("Status", width=8)
    table.add_column("Details", width=40)

    all_passed = True
    for name, passed, detail in checks:
        status = "[green]OK[/]" if passed else "[red]FAIL[/]"
        if not passed:
            all_passed = False
        table.add_row(name, status, detail)

    console.print(table)

    if not all_passed:
        console.print(
            "\n[yellow]Some checks failed. The pipeline may not "
            "run correctly. Install missing dependencies with:[/]\n"
            "  pip install -r requirements.txt\n"
        )

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Odoo Training Pipeline"
    )
    parser.add_argument(
        "--start-from",
        choices=[p["key"] for p in PHASES],
        help="Start from a specific phase (skip previous phases)",
    )
    parser.add_argument(
        "--only",
        choices=[p["key"] for p in PHASES],
        help="Run only a specific phase",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run (training uses only 5 steps)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Only estimate training time",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip pre-flight checks",
    )
    parser.add_argument(
        "--odoo-version",
        default="17.0",
        help="Odoo version to train on (default: 17.0)",
    )
    args = parser.parse_args()

    # Banner
    console.print(
        Panel(
            "[bold blue]"
            "╔═══════════════════════════════════════════════╗\n"
            "║     Qwen2.5-Coder-32B Odoo Fine-Tuning       ║\n"
            "║     2x RTX Pro 6000 96GB Training Pipeline    ║\n"
            "╚═══════════════════════════════════════════════╝"
            "[/]",
            border_style="blue",
        )
    )

    display_pipeline_plan()

    # Pre-flight checks
    if not args.skip_checks:
        console.print()
        checks_ok = preflight_checks()
        if not checks_ok:
            response = input(
                "\nSome checks failed. Continue anyway? [y/N]: "
            )
            if response.lower() != "y":
                console.print("[yellow]Aborted.[/]")
                return

    # Estimate mode
    if args.estimate:
        console.print("\n[cyan]Running in estimate-only mode...[/]\n")
        run_phase(
            {"key": "training", "name": "Training", "script": "04_train.py",
             "description": "Estimate training time"},
            extra_args=["--estimate-only"],
        )
        return

    # Determine which phases to run
    phases_to_run = PHASES.copy()
    if args.only:
        phases_to_run = [p for p in PHASES if p["key"] == args.only]
    elif args.start_from:
        start_idx = next(
            i for i, p in enumerate(PHASES) if p["key"] == args.start_from
        )
        phases_to_run = PHASES[start_idx:]

    console.print(
        f"\n[bold]Running {len(phases_to_run)} phase(s):[/] "
        + ", ".join(p["name"] for p in phases_to_run)
    )
    console.print()

    # GPU status
    gpu_monitor = GPUMonitor(
        log_dir=str(PROJECT_ROOT / "outputs" / "logs")
    )
    gpu_monitor.display_gpu_status()

    # Run pipeline
    results = {}
    pipeline_start = time.time()

    for phase in phases_to_run:
        extra_args = []
        if phase["key"] == "data_collection":
            extra_args.extend(["--odoo-version", args.odoo_version])

        success, duration = run_phase(
            phase,
            extra_args=extra_args,
            dry_run=args.dry_run,
        )
        results[phase["key"]] = {
            "success": success,
            "duration_minutes": round(duration / 60, 1),
        }

        if not success:
            console.print(
                f"\n[red]Pipeline stopped at '{phase['name']}'. "
                f"Fix the issue and re-run with "
                f"--start-from {phase['key']}[/]"
            )
            break

    # Final summary
    total_time = time.time() - pipeline_start
    total_minutes = total_time / 60
    total_hours = total_time / 3600

    table = Table(title="Pipeline Results", show_header=True)
    table.add_column("Phase", style="cyan", width=25)
    table.add_column("Status", width=10)
    table.add_column("Duration", width=15)

    for phase in phases_to_run:
        if phase["key"] in results:
            r = results[phase["key"]]
            status = (
                "[green]SUCCESS[/]" if r["success"] else "[red]FAILED[/]"
            )
            dur = f"{r['duration_minutes']:.1f} min"
        else:
            status = "[dim]SKIPPED[/]"
            dur = "—"
        table.add_row(phase["name"], status, dur)

    console.print()
    console.print(table)

    time_str = (
        f"{total_hours:.1f} hours"
        if total_hours >= 1
        else f"{total_minutes:.1f} minutes"
    )
    console.print(
        Panel(
            f"[bold]Total pipeline time: {time_str}[/]",
            border_style="blue",
        )
    )


if __name__ == "__main__":
    main()
