"""
GPU monitoring utilities for tracking VRAM usage, temperature, and utilization.
Designed for multi-GPU setups (2x RTX Pro 6000 96GB).
"""

import time
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMonitor:
    """Real-time GPU monitoring for training."""

    def __init__(self, log_dir: str = "outputs/logs", log_interval: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "gpu_metrics.jsonl"
        self.log_interval = log_interval
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self.num_gpus = self._get_gpu_count()

    def _get_gpu_count(self) -> int:
        if NVML_AVAILABLE:
            return pynvml.nvmlDeviceGetCount()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0

    def get_gpu_info(self) -> list[dict]:
        gpus = []
        if not NVML_AVAILABLE:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_total = (
                        torch.cuda.get_device_properties(i).total_mem
                        / (1024**3)
                    )
                    gpus.append(
                        {
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_used_gb": round(mem_alloc, 2),
                            "memory_total_gb": round(mem_total, 2),
                            "memory_percent": round(
                                mem_alloc / mem_total * 100, 1
                            ),
                            "temperature": None,
                            "utilization": None,
                            "power_draw_w": None,
                            "power_limit_w": None,
                        }
                    )
            return gpus

        for i in range(self.num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except Exception:
                gpu_util = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = (
                    pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                )
            except Exception:
                power = None
                power_limit = None

            gpus.append(
                {
                    "id": i,
                    "name": name,
                    "memory_used_gb": round(
                        mem_info.used / (1024**3), 2
                    ),
                    "memory_total_gb": round(
                        mem_info.total / (1024**3), 2
                    ),
                    "memory_free_gb": round(
                        mem_info.free / (1024**3), 2
                    ),
                    "memory_percent": round(
                        mem_info.used / mem_info.total * 100, 1
                    ),
                    "temperature": temp,
                    "utilization": gpu_util,
                    "power_draw_w": round(power, 1) if power else None,
                    "power_limit_w": (
                        round(power_limit, 1) if power_limit else None
                    ),
                }
            )
        return gpus

    def display_gpu_status(self):
        gpus = self.get_gpu_info()
        if not gpus:
            console.print("[yellow]No GPUs detected.[/]")
            return

        table = Table(
            title="GPU Status",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("GPU", style="cyan", width=8)
        table.add_column("Name", width=28)
        table.add_column("VRAM Used", width=14)
        table.add_column("VRAM %", width=8)
        table.add_column("Temp", width=8)
        table.add_column("Util %", width=8)
        table.add_column("Power", width=14)

        for gpu in gpus:
            mem_str = f"{gpu['memory_used_gb']:.1f} / {gpu['memory_total_gb']:.1f} GB"
            mem_pct = f"{gpu['memory_percent']:.0f}%"
            temp_str = (
                f"{gpu['temperature']}°C"
                if gpu["temperature"] is not None
                else "N/A"
            )
            util_str = (
                f"{gpu['utilization']}%"
                if gpu["utilization"] is not None
                else "N/A"
            )
            power_str = "N/A"
            if gpu["power_draw_w"] is not None:
                power_str = f"{gpu['power_draw_w']:.0f} / {gpu['power_limit_w']:.0f} W"

            # Color code memory usage
            if gpu["memory_percent"] > 90:
                mem_pct = f"[red]{mem_pct}[/]"
            elif gpu["memory_percent"] > 70:
                mem_pct = f"[yellow]{mem_pct}[/]"
            else:
                mem_pct = f"[green]{mem_pct}[/]"

            table.add_row(
                f"GPU {gpu['id']}",
                gpu["name"],
                mem_str,
                mem_pct,
                temp_str,
                util_str,
                power_str,
            )

        total_used = sum(g["memory_used_gb"] for g in gpus)
        total_mem = sum(g["memory_total_gb"] for g in gpus)

        console.print(
            Panel(
                table,
                title=f"[bold]GPU Monitor[/] | Total VRAM: "
                f"{total_used:.1f} / {total_mem:.1f} GB",
                border_style="blue",
            )
        )

    def start_background_logging(self):
        if self._monitoring:
            return
        self._monitoring = True
        self._thread = threading.Thread(
            target=self._log_loop, daemon=True
        )
        self._thread.start()
        console.print(
            f"[green]GPU monitoring started "
            f"(logging every {self.log_interval}s)[/]"
        )

    def stop_background_logging(self):
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=5)
        console.print("[yellow]GPU monitoring stopped.[/]")

    def _log_loop(self):
        while self._monitoring:
            gpus = self.get_gpu_info()
            record = {
                "timestamp": datetime.now().isoformat(),
                "gpus": gpus,
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            time.sleep(self.log_interval)

    def check_vram_sufficient(
        self, required_gb: float = 60.0
    ) -> tuple[bool, str]:
        gpus = self.get_gpu_info()
        if not gpus:
            return False, "No GPUs detected"

        total_free = sum(g.get("memory_free_gb", 0) for g in gpus)
        total_mem = sum(g["memory_total_gb"] for g in gpus)

        if total_mem < required_gb:
            return (
                False,
                f"Total VRAM ({total_mem:.0f} GB) is less than "
                f"required ({required_gb:.0f} GB)",
            )

        return (
            True,
            f"VRAM check passed: {total_mem:.0f} GB total "
            f"({total_free:.0f} GB free) across {len(gpus)} GPU(s)",
        )

    def estimate_batch_size(self, model_size_b: float = 32.0) -> dict:
        """Estimate optimal batch size for given model size and available VRAM."""
        gpus = self.get_gpu_info()
        total_vram = sum(g["memory_total_gb"] for g in gpus)
        num_gpus = len(gpus)

        # Rough estimates for bf16 LoRA training
        model_mem = model_size_b * 2  # ~2 GB per billion params in bf16
        lora_overhead = model_mem * 0.05  # LoRA adds ~5% overhead
        optimizer_mem = lora_overhead * 3  # AdamW: 3x LoRA param memory
        activation_per_sample = 2.0  # ~2 GB per sample for 32B model

        base_mem = model_mem + lora_overhead + optimizer_mem
        available_for_batch = total_vram - base_mem - 4.0  # 4GB safety margin

        max_batch = max(1, int(available_for_batch / activation_per_sample))
        recommended_batch = max(1, max_batch // 2)  # conservative

        return {
            "num_gpus": num_gpus,
            "total_vram_gb": total_vram,
            "model_memory_gb": model_mem,
            "estimated_overhead_gb": lora_overhead + optimizer_mem,
            "available_for_batch_gb": available_for_batch,
            "max_batch_size_per_gpu": max_batch // max(num_gpus, 1),
            "recommended_batch_size_per_gpu": recommended_batch
            // max(num_gpus, 1),
            "recommended_gradient_accumulation": max(
                1, 32 // (recommended_batch // max(num_gpus, 1))
            ),
        }
