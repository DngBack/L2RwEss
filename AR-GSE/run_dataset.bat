@echo off
echo ================================
echo AR-GSE Dataset Runner
echo ================================
echo.

echo Available Commands:
echo.
echo 1. Run with CIFAR-10 (default settings)
echo 2. Run with CIFAR-100 (default settings)  
echo 3. Run with custom imbalance factor
echo 4. Run with different head ratio
echo 5. Run with custom seed
echo 6. Show help
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo Running CIFAR-10 with default settings...
    python -m src.data.splits --root ./data --name cifar10 --imb_factor 100.0 --seed 42 --save ./splits/cifar10_lt_if100_seed42.json
    goto end
)

if "%choice%"=="2" (
    echo Running CIFAR-100 with default settings...
    python -m src.data.splits --root ./data --name cifar100 --imb_factor 100.0 --seed 42 --save ./splits/cifar100_lt_if100_seed42.json
    goto end
)

if "%choice%"=="3" (
    set /p imb="Enter imbalance factor (e.g., 10.0, 50.0, 100.0): "
    set /p dataset="Enter dataset (cifar10 or cifar100): "
    echo Running %dataset% with imbalance factor %imb%...
    python -m src.data.splits --root ./data --name %dataset% --imb_factor %imb% --seed 42 --save ./splits/%dataset%_lt_if%imb%_seed42.json
    goto end
)

if "%choice%"=="4" (
    set /p ratio="Enter head ratio (e.g., 0.3, 0.5, 0.7): "
    set /p dataset="Enter dataset (cifar10 or cifar100): "
    echo Running %dataset% with head ratio %ratio%...
    python -m src.data.splits --root ./data --name %dataset% --imb_factor 50.0 --seed 42 --head_ratio %ratio% --save ./splits/%dataset%_lt_if50_hr%ratio%_seed42.json
    goto end
)

if "%choice%"=="5" (
    set /p seed="Enter seed (e.g., 42, 123, 999): "
    set /p dataset="Enter dataset (cifar10 or cifar100): "
    echo Running %dataset% with seed %seed%...
    python -m src.data.splits --root ./data --name %dataset% --imb_factor 100.0 --seed %seed% --save ./splits/%dataset%_lt_if100_seed%seed%.json
    goto end
)

if "%choice%"=="6" (
    echo.
    echo Dataset Splits Help:
    echo.
    echo Parameters:
    echo   --root: Directory to store CIFAR data (default: ./data)
    echo   --name: Dataset name (cifar10 or cifar100)
    echo   --imb_factor: Imbalance factor - higher = more imbalanced (default: 100.0)
    echo   --seed: Random seed for reproducibility (default: 42)
    echo   --save: Output file path for splits (default: ./splits/...)
    echo   --head_ratio: Ratio of classes in head group (default: 0.5)
    echo.
    echo Example: python -m src.data.splits --name cifar10 --imb_factor 50.0 --seed 123
    echo.
    python -m src.data.splits --help
    goto end
)

echo Invalid choice. Please run the script again.

:end
echo.
echo ================================
echo Check ./splits/ folder for output files
echo ================================
pause