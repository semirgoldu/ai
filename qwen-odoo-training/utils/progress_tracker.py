"""
Progress tracking utilities for Qwen-Odoo training pipeline.
Provides real-time ETA, phase tracking, and detailed metrics display.
"""

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.layout import Layout
from rich.live import Live

console = Console()


class PhaseTracker:
    """Tracks the overall pipeline phases."""

    PHASES = [
        ("data_collection", "Data Collection"),
        ("data_preprocessing", "Data Preprocessing"),
        ("tokenization", "Tokenization"),
        ("training", "Model Training"),
        ("evaluation", "Evaluation"),
        ("export", "Model Export"),
    ]

    def __init__(self, log_dir: str = "outputs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.log_dir / "pipeline_state.json"
        self.current_phase_idx = 0
        self.phase_times: dict[str, dict] = {}
        self.pipeline_start = time.time()
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            with open(self.state_file) as f:
                state = json.load(f)
                self.phase_times = state.get("phase_times", {})
                self.current_phase_idx = state.get("current_phase_idx", 0)
                self.pipeline_start = state.get(
                    "pipeline_start", time.time()
                )

    def _save_state(self):
        state = {
            "phase_times": self.phase_times,
            "current_phase_idx": self.current_phase_idx,
            "pipeline_start": self.pipeline_start,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def start_phase(self, phase_key: str):
        self.phase_times[phase_key] = {
            "start": time.time(),
            "end": None,
            "status": "running",
        }
        for i, (key, _) in enumerate(self.PHASES):
            if key == phase_key:
                self.current_phase_idx = i
                break
        self._save_state()
        self.display_status()

    def end_phase(self, phase_key: str, status: str = "completed"):
        if phase_key in self.phase_times:
            self.phase_times[phase_key]["end"] = time.time()
            self.phase_times[phase_key]["status"] = status
            self._save_state()
        self.display_status()

    def display_status(self):
        table = Table(
            title="Pipeline Progress",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Phase", style="cyan", width=25)
        table.add_column("Status", width=12)
        table.add_column("Duration", width=15)
        table.add_column("", width=3)

        total_elapsed = time.time() - self.pipeline_start

        for i, (key, name) in enumerate(self.PHASES):
            if key in self.phase_times:
                info = self.phase_times[key]
                if info["status"] == "running":
                    elapsed = time.time() - info["start"]
                    duration = _format_duration(elapsed)
                    status = "[bold yellow]RUNNING[/]"
                    marker = ">>>"
                elif info["status"] == "completed":
                    elapsed = info["end"] - info["start"]
                    duration = _format_duration(elapsed)
                    status = "[bold green]DONE[/]"
                    marker = "[green]OK[/green]"
                else:
                    duration = "—"
                    status = f"[bold red]{info['status'].upper()}[/]"
                    marker = "[red]!![/red]"
            elif i == self.current_phase_idx + 1:
                duration = "—"
                status = "[dim]NEXT[/]"
                marker = "..."
            else:
                duration = "—"
                status = "[dim]PENDING[/]"
                marker = ""

            table.add_row(name, status, duration, marker)

        console.print()
        console.print(
            Panel(
                table,
                title=f"[bold]Qwen-Odoo Training Pipeline[/] | "
                f"Total: {_format_duration(total_elapsed)}",
                border_style="blue",
            )
        )
        console.print()


class TrainingProgressTracker:
    """Detailed progress tracker for the training phase."""

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        log_dir: str = "outputs/logs",
        log_interval: int = 10,
    ):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval

        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.epoch_start_time = None
        self.training_start_time = None

        self.loss_history: list[float] = []
        self.lr_history: list[float] = []
        self.step_times: list[float] = []

        self.metrics_file = self.log_dir / "training_metrics.jsonl"

    def start_training(self):
        self.training_start_time = time.time()
        console.print(
            Panel(
                f"[bold green]Training Started[/]\n"
                f"Epochs: {self.total_epochs} | "
                f"Steps/Epoch: {self.steps_per_epoch} | "
                f"Total Steps: {self.total_steps}",
                title="Training",
                border_style="green",
            )
        )

    def start_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.current_step = 0
        self.epoch_start_time = time.time()
        console.print(
            f"\n[bold cyan]{'='*60}[/]"
        )
        console.print(
            f"[bold cyan]  Epoch {epoch + 1}/{self.total_epochs}[/]"
        )
        console.print(
            f"[bold cyan]{'='*60}[/]\n"
        )

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        gpu_mem_used: Optional[float] = None,
        extra_metrics: Optional[dict] = None,
    ):
        self.current_step = step
        self.global_step += 1
        now = time.time()

        self.loss_history.append(loss)
        self.lr_history.append(learning_rate)

        if len(self.step_times) > 0:
            self.step_times.append(now - self._last_step_time)
        self._last_step_time = now

        # Log to file
        record = {
            "timestamp": datetime.now().isoformat(),
            "epoch": self.current_epoch,
            "step": step,
            "global_step": self.global_step,
            "loss": loss,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "gpu_mem_used_gb": gpu_mem_used,
        }
        if extra_metrics:
            record.update(extra_metrics)

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Display progress at intervals
        if step % self.log_interval == 0 or step == self.steps_per_epoch:
            self._display_step_progress(
                step, loss, learning_rate, grad_norm, gpu_mem_used
            )

    def _display_step_progress(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: Optional[float],
        gpu_mem: Optional[float],
    ):
        # Calculate ETAs
        epoch_progress = step / self.steps_per_epoch
        total_progress = self.global_step / self.total_steps

        if len(self.step_times) > 1:
            avg_step_time = sum(self.step_times[-100:]) / len(
                self.step_times[-100:]
            )
            remaining_steps = self.total_steps - self.global_step
            eta_seconds = remaining_steps * avg_step_time
            eta_str = _format_duration(eta_seconds)

            epoch_remaining = self.steps_per_epoch - step
            epoch_eta = _format_duration(epoch_remaining * avg_step_time)
            speed = 1.0 / avg_step_time if avg_step_time > 0 else 0
        else:
            eta_str = "calculating..."
            epoch_eta = "calculating..."
            speed = 0

        elapsed = _format_duration(time.time() - self.training_start_time)

        # Smoothed loss (last 50 steps)
        recent_loss = self.loss_history[-50:]
        avg_loss = sum(recent_loss) / len(recent_loss)

        # Build display
        bar_width = 40
        filled = int(bar_width * epoch_progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        total_filled = int(bar_width * total_progress)
        total_bar = "█" * total_filled + "░" * (bar_width - total_filled)

        lines = [
            f"  Epoch   [{bar}] {epoch_progress*100:5.1f}%  "
            f"Step {step}/{self.steps_per_epoch}  ETA: {epoch_eta}",
            f"  Overall [{total_bar}] {total_progress*100:5.1f}%  "
            f"Step {self.global_step}/{self.total_steps}  ETA: {eta_str}",
            f"",
            f"  Loss: {loss:.4f}  (avg: {avg_loss:.4f})  |  "
            f"LR: {lr:.2e}  |  Speed: {speed:.2f} steps/s",
        ]

        if grad_norm is not None:
            lines[-1] += f"  |  Grad Norm: {grad_norm:.4f}"
        if gpu_mem is not None:
            lines[-1] += f"  |  GPU Mem: {gpu_mem:.1f} GB"

        lines.append(f"  Elapsed: {elapsed}")

        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]Epoch {self.current_epoch + 1}/{self.total_epochs}[/]",
                border_style="yellow",
            )
        )

    def end_epoch(self, val_loss: Optional[float] = None):
        epoch_time = time.time() - self.epoch_start_time
        avg_loss = sum(self.loss_history[-self.steps_per_epoch :]) / min(
            self.steps_per_epoch, len(self.loss_history)
        )
        msg = (
            f"Epoch {self.current_epoch + 1} completed in "
            f"{_format_duration(epoch_time)}\n"
            f"Average Loss: {avg_loss:.4f}"
        )
        if val_loss is not None:
            msg += f"\nValidation Loss: {val_loss:.4f}"
        console.print(Panel(msg, title="Epoch Summary", border_style="green"))

    def end_training(self):
        total_time = time.time() - self.training_start_time
        final_loss = (
            sum(self.loss_history[-100:]) / len(self.loss_history[-100:])
            if self.loss_history
            else 0
        )
        console.print(
            Panel(
                f"[bold green]Training Complete![/]\n\n"
                f"Total Time:    {_format_duration(total_time)}\n"
                f"Total Steps:   {self.global_step}\n"
                f"Final Avg Loss: {final_loss:.4f}\n"
                f"Metrics saved to: {self.metrics_file}",
                title="Training Complete",
                border_style="bold green",
            )
        )

    def get_progress_bar(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
        )


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "N/A"
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)
