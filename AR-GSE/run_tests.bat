@echo off
echo Installing test dependencies...
pip install pytest pytest-mock

echo.
echo Running dataset tests...
python -m pytest tests/data/test_dataset.py -v --tb=short

pause