@echo off
REM ============================================================
REM Qwen2.5-Coder-32B Odoo Fine-Tuning - Windows Setup & Run
REM Hardware: 2x RTX Pro 6000 96GB
REM ============================================================

echo.
echo ========================================================
echo   Qwen2.5-Coder-32B Odoo Fine-Tuning Setup
echo   Windows 11 - 2x RTX Pro 6000 96GB
echo ========================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check git
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git not found. Install Git from git-scm.com
    pause
    exit /b 1
)

REM Check NVIDIA driver
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvidia-smi not found. Install NVIDIA drivers.
    pause
    exit /b 1
)

echo [OK] Python, Git, and NVIDIA drivers found.
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

REM Activate virtual environment
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated.
echo.

REM Install PyTorch with CUDA
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
echo.

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Try to install flash-attn (optional, may need Visual Studio Build Tools)
echo Attempting to install Flash Attention 2 (optional)...
pip install flash-attn --no-build-isolation 2>nul
if errorlevel 1 (
    echo [WARN] Flash Attention not installed. Will use SDPA instead.
    echo        For Flash Attention, install Visual Studio Build Tools first.
) else (
    echo [OK] Flash Attention installed.
)
echo.

REM Verify GPU setup
echo Verifying GPU setup...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.0f} GB)') for i in range(torch.cuda.device_count())]"
echo.

echo ========================================================
echo   Setup complete! Choose what to do:
echo ========================================================
echo.
echo   1. Run FULL pipeline (collect data + train + evaluate)
echo   2. Estimate training time only
echo   3. Run a dry-run (5 training steps)
echo   4. Collect data only
echo   5. Start training only (data already collected)
echo   6. Run pre-flight checks only
echo   7. Exit
echo.
set /p choice="Enter choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo Starting full pipeline...
    python scripts\run_pipeline.py
) else if "%choice%"=="2" (
    echo.
    echo Estimating training time...
    python scripts\run_pipeline.py --estimate
) else if "%choice%"=="3" (
    echo.
    echo Starting dry run...
    python scripts\run_pipeline.py --dry-run
) else if "%choice%"=="4" (
    echo.
    echo Collecting Odoo data...
    python scripts\01_collect_odoo_source.py
    python scripts\02_collect_odoo_docs.py
    python scripts\03_preprocess_data.py
) else if "%choice%"=="5" (
    echo.
    echo Starting training...
    python scripts\run_pipeline.py --start-from training
) else if "%choice%"=="6" (
    echo.
    echo Running pre-flight checks...
    python scripts\run_pipeline.py --skip-checks --estimate
) else if "%choice%"=="7" (
    echo Exiting.
) else (
    echo Invalid choice.
)

echo.
pause
