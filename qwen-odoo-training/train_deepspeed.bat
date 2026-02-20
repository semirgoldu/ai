@echo off
REM ============================================================
REM Launch training with DeepSpeed (multi-GPU)
REM For 2x RTX Pro 6000 96GB
REM ============================================================

call venv\Scripts\activate.bat

echo Starting DeepSpeed training on 2 GPUs...
echo.

deepspeed --num_gpus=2 scripts/04_train.py ^
    --config configs/training_config.yaml

echo.
echo Training finished.
pause
