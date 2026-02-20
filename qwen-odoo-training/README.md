# Qwen2.5-Coder-32B Odoo Fine-Tuning

Fine-tune **Qwen2.5-Coder-32B-Instruct** on Odoo source code and documentation using LoRA + DeepSpeed.

**Hardware**: Windows 11 with 2x NVIDIA RTX Pro 6000 96GB (192GB VRAM total)

## Project Structure

```
qwen-odoo-training/
├── configs/
│   ├── training_config.yaml       # All training hyperparameters
│   └── deepspeed_config.json      # DeepSpeed ZeRO-2 config
├── scripts/
│   ├── 01_collect_odoo_source.py  # Clone Odoo repos
│   ├── 02_collect_odoo_docs.py    # Process documentation
│   ├── 03_preprocess_data.py      # Format training data
│   ├── 04_train.py                # Main training script
│   ├── 05_evaluate.py             # Model evaluation
│   ├── 06_export_model.py         # Merge LoRA + export
│   └── run_pipeline.py            # Full pipeline orchestrator
├── utils/
│   ├── progress_tracker.py        # Training progress & ETA
│   ├── gpu_monitor.py             # GPU VRAM/temp monitoring
│   └── data_utils.py              # Data processing utilities
├── data/
│   ├── raw/odoo_source/           # Cloned Odoo repos
│   └── processed/                 # Training-ready datasets
├── outputs/
│   ├── checkpoints/               # Training checkpoints
│   ├── final_model/               # Final fine-tuned model
│   ├── logs/                      # Training metrics
│   └── tensorboard/               # TensorBoard logs
├── setup_and_train.bat            # Windows one-click setup
├── train_deepspeed.bat            # Launch training with DeepSpeed
└── requirements.txt               # Python dependencies
```

## Quick Start (Windows)

### Option 1: One-Click Setup

Double-click `setup_and_train.bat` - it will:
1. Create a virtual environment
2. Install PyTorch with CUDA
3. Install all dependencies
4. Present a menu to run the pipeline

### Option 2: Manual Setup

```powershell
# Create and activate venv
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Optional: Flash Attention 2 (requires Visual Studio Build Tools)
pip install flash-attn --no-build-isolation
```

## Running the Pipeline

### Full Pipeline
```powershell
python scripts/run_pipeline.py
```

### Estimate Training Time Only
```powershell
python scripts/run_pipeline.py --estimate
```

### Dry Run (5 training steps)
```powershell
python scripts/run_pipeline.py --dry-run
```

### Run Individual Phases
```powershell
# Step 1: Collect Odoo source code
python scripts/01_collect_odoo_source.py --odoo-version 17.0

# Step 2: Process documentation
python scripts/02_collect_odoo_docs.py

# Step 3: Preprocess into training format
python scripts/03_preprocess_data.py

# Step 4: Train (single command - DeepSpeed handles multi-GPU)
deepspeed --num_gpus=2 scripts/04_train.py --config configs/training_config.yaml

# Step 5: Evaluate
python scripts/05_evaluate.py

# Step 6: Export (merge LoRA + optional GGUF)
python scripts/06_export_model.py --merge --gguf --ollama
```

### Resume from a Specific Phase
```powershell
# Skip data collection, start from training
python scripts/run_pipeline.py --start-from training

# Run only evaluation
python scripts/run_pipeline.py --only evaluation
```

### Resume from Checkpoint
```powershell
python scripts/04_train.py --resume-from-checkpoint outputs/checkpoints/checkpoint-1000
```

## Pipeline Phases

| # | Phase | What it Does |
|---|-------|-------------|
| 1 | **Data Collection** | Clones Odoo community, documentation, and OCA repos |
| 2 | **Doc Processing** | Parses RST/MD docs into Q&A training examples |
| 3 | **Preprocessing** | Formats code into instruction pairs, deduplicates, splits train/val/test |
| 4 | **Training** | LoRA fine-tuning with DeepSpeed ZeRO-2, real-time progress & ETA |
| 5 | **Evaluation** | ROUGE metrics on test set + Odoo-specific coding tasks |
| 6 | **Export** | Merges LoRA weights, optional GGUF export for Ollama |

## Training Configuration

Key settings in `configs/training_config.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | Qwen2.5-Coder-32B-Instruct | ~64GB in bf16 |
| Method | LoRA (r=64, alpha=128) | ~2% trainable params |
| Precision | BF16 | Native on RTX Pro 6000 |
| Batch size | 2 per GPU x 2 GPUs x 8 accum = **32 effective** | |
| Learning rate | 2e-4 with cosine schedule | |
| Epochs | 3 | |
| Max seq length | 4096 tokens | |
| DeepSpeed | ZeRO Stage 2 | Splits optimizer state across GPUs |
| Gradient checkpointing | Enabled | Trades compute for memory |

## Progress Tracking

The training script provides real-time feedback:

- **Phase tracker** - shows which pipeline phase is active with duration
- **Epoch progress bar** - current epoch progress with ETA
- **Overall progress bar** - total training progress with ETA
- **Live metrics** - loss, learning rate, gradient norm, GPU memory
- **GPU monitor** - background logging of VRAM, temperature, utilization
- **TensorBoard** - detailed training curves

View TensorBoard logs:
```powershell
tensorboard --logdir outputs/tensorboard
```

## Memory Estimates (2x RTX Pro 6000 96GB)

| Component | Memory |
|-----------|--------|
| Model (bf16) | ~64 GB |
| LoRA parameters | ~3 GB |
| Optimizer states | ~9 GB |
| Gradients | ~3 GB |
| Activations (batch=2, checkpointing) | ~8 GB/GPU |
| **Total per GPU (ZeRO-2)** | **~50-55 GB** |
| **Headroom per GPU** | **~40 GB** |

With 192GB total VRAM and ZeRO-2 splitting optimizer states, there is substantial headroom. You can increase batch size or LoRA rank if needed.

## Customization

### Change Odoo Version
```powershell
python scripts/01_collect_odoo_source.py --odoo-version 16.0
```

### Adjust LoRA Rank
Edit `configs/training_config.yaml`:
```yaml
lora:
  r: 128        # Higher rank = more capacity
  lora_alpha: 256
```

### Increase Batch Size
With 96GB per GPU, you can likely increase batch size:
```yaml
training:
  per_device_train_batch_size: 4  # Try 4 instead of 2
  gradient_accumulation_steps: 4  # Reduce to keep effective batch = 32
```

### Use Full Fine-Tuning (advanced)
With 192GB VRAM, full fine-tuning may be possible with ZeRO-3 + CPU offloading.
Edit the DeepSpeed config to use stage 3 and enable CPU offload.

## Export Options

### Hugging Face Format (default)
```powershell
python scripts/06_export_model.py --merge
```

### GGUF for Ollama / llama.cpp
```powershell
python scripts/06_export_model.py --merge --gguf --gguf-quantization q4_k_m
```

### Create Ollama Model
```powershell
python scripts/06_export_model.py --merge --gguf --ollama
cd outputs/exported
ollama create qwen-odoo -f Modelfile
ollama run qwen-odoo
```

## Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Reduce LoRA `r` to 32
- Ensure `gradient_checkpointing: true`
- Enable DeepSpeed CPU offloading in `deepspeed_config.json`

### Flash Attention Build Error
- Install Visual Studio Build Tools with C++ workload
- Or just use SDPA (the script auto-detects and falls back)

### Training Interrupted
- Resume with: `python scripts/04_train.py --resume-from-checkpoint outputs/checkpoints/checkpoint-XXXX`
- The pipeline saves checkpoints every 500 steps by default

### Slow Data Collection
- Use `--depth 1` (default) for shallow git clones
- Skip OCA repos with `--include-oca false`
