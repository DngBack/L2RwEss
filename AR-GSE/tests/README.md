# AR-GSE Dataset Tests

This directory contains comprehensive tests for the dataset functions in `src/data/dataset.py`.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
└── data/
    ├── __init__.py
    └── test_dataset.py       # Tests for dataset.py functions
```

## Functions Tested

The test suite covers all functions in `dataset.py`:

1. **Transform Functions**
   - `cifar_train_transform()` - Tests CIFAR training transforms
   - `cifar_eval_transform()` - Tests CIFAR evaluation transforms

2. **Data Loading**
   - `load_cifar()` - Tests CIFAR-10/100 dataset loading

3. **Long-tailed Distribution**
   - `img_num_per_cls()` - Tests exponential distribution generation
   - `make_long_tailed_indices()` - Tests long-tailed sampling

4. **Data Splitting**
   - `SplitConfig` - Tests split configuration dataclass
   - `partition_indices()` - Tests index partitioning
   - `save_splits()` / `load_splits()` - Tests JSON serialization

5. **Complete Pipeline**
   - `build_cifar_lt_splits()` - Tests the full pipeline
   - `count_per_class()` - Tests class counting utilities

## Requirements

Install test dependencies:

```bash
pip install pytest pytest-mock
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Running Tests

### Option 1: Using pytest directly
```bash
pytest tests/data/test_dataset.py -v
```

### Option 2: Using the test runner script
```bash
python run_tests.py
```

### Option 3: Run all tests
```bash
pytest tests/ -v
```

## Test Features

- **Comprehensive Coverage**: Tests all functions with various edge cases
- **Mocking**: Uses unittest.mock to avoid downloading actual CIFAR datasets
- **Reproducibility**: Tests random seed functionality
- **Error Handling**: Tests invalid inputs and edge cases
- **Integration**: Tests the complete pipeline end-to-end

## Example Test Output

```
tests/data/test_dataset.py::TestTransformFunctions::test_cifar_train_transform PASSED
tests/data/test_dataset.py::TestTransformFunctions::test_cifar_eval_transform PASSED
tests/data/test_dataset.py::TestLoadCifar::test_load_cifar10_train PASSED
tests/data/test_dataset.py::TestLoadCifar::test_load_cifar100_test PASSED
tests/data/test_dataset.py::TestImgNumPerCls::test_balanced_distribution PASSED
tests/data/test_dataset.py::TestImgNumPerCls::test_imbalanced_distribution PASSED
...
```