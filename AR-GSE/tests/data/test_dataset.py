import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from torch.utils.data import Subset
from torchvision import transforms

# Import the functions we want to test
from data.dataset import (
    cifar_train_transform,
    cifar_eval_transform,
    load_cifar,
    img_num_per_cls,
    make_long_tailed_indices,
    SplitConfig,
    partition_indices,
    save_splits,
    load_splits,
    build_cifar_lt_splits,
    count_per_class
)


class TestTransformFunctions:
    """Test the CIFAR transform functions."""
    
    def test_cifar_train_transform(self):
        """Test that train transform returns correct composition."""
        transform = cifar_train_transform()
        
        # Check that it's a Compose object
        assert isinstance(transform, transforms.Compose)
        
        # Check that it contains the expected transforms
        transform_types = [type(t) for t in transform.transforms]
        expected_types = [
            transforms.RandomCrop,
            transforms.RandomHorizontalFlip,
            transforms.ToTensor,
            transforms.Normalize
        ]
        
        assert len(transform_types) == len(expected_types)
        for actual, expected in zip(transform_types, expected_types):
            assert actual == expected
            
        # Check RandomCrop parameters
        random_crop = transform.transforms[0]
        assert random_crop.size == (32, 32)
        assert random_crop.padding == 4
        
        # Check Normalize parameters
        normalize = transform.transforms[3]
        expected_mean = (0.4914, 0.4822, 0.4465)
        expected_std = (0.2023, 0.1994, 0.2010)
        assert normalize.mean == expected_mean
        assert normalize.std == expected_std
    
    def test_cifar_eval_transform(self):
        """Test that eval transform returns correct composition."""
        transform = cifar_eval_transform()
        
        # Check that it's a Compose object
        assert isinstance(transform, transforms.Compose)
        
        # Check that it contains the expected transforms
        transform_types = [type(t) for t in transform.transforms]
        expected_types = [
            transforms.ToTensor,
            transforms.Normalize
        ]
        
        assert len(transform_types) == len(expected_types)
        for actual, expected in zip(transform_types, expected_types):
            assert actual == expected
            
        # Check Normalize parameters
        normalize = transform.transforms[1]
        expected_mean = (0.4914, 0.4822, 0.4465)
        expected_std = (0.2023, 0.1994, 0.2010)
        assert normalize.mean == expected_mean
        assert normalize.std == expected_std


class TestLoadCifar:
    """Test the load_cifar function."""
    
    @patch('data.dataset.datasets.CIFAR10')
    def test_load_cifar10_train(self, mock_cifar10):
        """Test loading CIFAR-10 train dataset."""
        # Mock the CIFAR10 dataset
        mock_dataset = MagicMock()
        mock_dataset.targets = [0, 1, 2, 3, 4] * 1000  # 5000 samples
        mock_cifar10.return_value = mock_dataset
        
        ds, targets, num_classes = load_cifar("/fake/root", "cifar10", train=True)
        
        # Check that CIFAR10 was called with correct parameters
        mock_cifar10.assert_called_once()
        call_args = mock_cifar10.call_args
        assert call_args[0][0] == "/fake/root"
        assert call_args[1]['train'] is True
        assert call_args[1]['download'] is True
        assert call_args[1]['transform'] is not None
        
        # Check return values
        assert ds == mock_dataset
        assert targets == mock_dataset.targets
        assert num_classes == 10
    
    @patch('data.dataset.datasets.CIFAR100')
    def test_load_cifar100_test(self, mock_cifar100):
        """Test loading CIFAR-100 test dataset."""
        # Mock the CIFAR100 dataset
        mock_dataset = MagicMock()
        mock_dataset.targets = [0, 1, 2] * 3333  # ~10000 samples
        mock_cifar100.return_value = mock_dataset
        
        ds, targets, num_classes = load_cifar("/fake/root", "cifar100", train=False)
        
        # Check that CIFAR100 was called with correct parameters
        mock_cifar100.assert_called_once()
        call_args = mock_cifar100.call_args
        assert call_args[0][0] == "/fake/root"
        assert call_args[1]['train'] is False
        assert call_args[1]['download'] is True
        assert call_args[1]['transform'] is not None
        
        # Check return values
        assert ds == mock_dataset
        assert targets == mock_dataset.targets
        assert num_classes == 100
    
    def test_load_cifar_invalid_name(self):
        """Test that invalid dataset name raises assertion error."""
        with pytest.raises(AssertionError):
            load_cifar("/fake/root", "invalid_dataset", train=True)


class TestImgNumPerCls:
    """Test the img_num_per_cls function."""
    
    def test_balanced_distribution(self):
        """Test balanced distribution when imb_factor <= 1."""
        result = img_num_per_cls(base_per_cls=100, c_num=10, imb_factor=1.0)
        expected = [100] * 10
        assert result == expected
        
        result = img_num_per_cls(base_per_cls=50, c_num=5, imb_factor=0.5)
        expected = [50] * 5
        assert result == expected
    
    def test_imbalanced_distribution(self):
        """Test imbalanced distribution with exponential decay."""
        result = img_num_per_cls(base_per_cls=100, c_num=3, imb_factor=10.0)
        
        # Check that distribution is decreasing
        assert result[0] >= result[1] >= result[2]
        
        # Check that first class has maximum samples
        assert result[0] == 100
        
        # Check that all classes have at least 1 sample
        assert all(n >= 1 for n in result)
        
        # Check length
        assert len(result) == 3
    
    def test_minimum_samples_per_class(self):
        """Test that each class gets at least 1 sample."""
        result = img_num_per_cls(base_per_cls=10, c_num=100, imb_factor=1000.0)
        
        assert len(result) == 100
        assert all(n >= 1 for n in result)
        assert result[0] == 10  # Head class should have base_per_cls samples


class TestMakeLongTailedIndices:
    """Test the make_long_tailed_indices function."""
    
    def test_balanced_case(self):
        """Test with imb_factor = 1 (balanced)."""
        # Create fake targets: 100 samples per class for 3 classes
        targets = [0] * 100 + [1] * 100 + [2] * 100
        
        indices = make_long_tailed_indices(targets, num_classes=3, imb_factor=1.0, seed=42)
        
        # Check that we get all indices back (balanced case)
        assert len(indices) == 300
        assert set(indices) == set(range(300))
    
    def test_imbalanced_case(self):
        """Test with imb_factor > 1 (imbalanced)."""
        # Create fake targets: 100 samples per class for 3 classes
        targets = [0] * 100 + [1] * 100 + [2] * 100
        
        indices = make_long_tailed_indices(targets, num_classes=3, imb_factor=10.0, seed=42)
        
        # Should have fewer samples than original
        assert len(indices) < 300
        
        # Check that indices are valid
        assert all(0 <= idx < 300 for idx in indices)
        
        # Check that class distribution follows long-tail
        targets_array = np.array(targets)
        selected_targets = targets_array[indices]
        class_counts = np.bincount(selected_targets, minlength=3)
        
        # Class 0 should have most samples, class 2 should have least
        assert class_counts[0] >= class_counts[1] >= class_counts[2]
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        targets = [0] * 50 + [1] * 50 + [2] * 50
        
        indices1 = make_long_tailed_indices(targets, num_classes=3, imb_factor=5.0, seed=123)
        indices2 = make_long_tailed_indices(targets, num_classes=3, imb_factor=5.0, seed=123)
        
        assert indices1 == indices2


class TestSplitConfig:
    """Test the SplitConfig dataclass."""
    
    def test_default_values(self):
        """Test default split configuration values."""
        config = SplitConfig()
        
        assert config.train == 0.80
        assert config.tuneV == 0.08
        assert config.val_small == 0.06
        assert config.calib == 0.06
        
        # Check that splits sum to 1.0
        total = config.train + config.tuneV + config.val_small + config.calib
        assert abs(total - 1.0) < 1e-6
    
    def test_custom_values(self):
        """Test custom split configuration values."""
        config = SplitConfig(train=0.7, tuneV=0.1, val_small=0.1, calib=0.1)
        
        assert config.train == 0.7
        assert config.tuneV == 0.1
        assert config.val_small == 0.1
        assert config.calib == 0.1


class TestPartitionIndices:
    """Test the partition_indices function."""
    
    def test_basic_partitioning(self):
        """Test basic index partitioning."""
        indices = list(range(1000))
        config = SplitConfig(train=0.8, tuneV=0.1, val_small=0.05, calib=0.05)
        
        splits = partition_indices(indices, config, seed=42)
        
        # Check that all expected keys are present
        expected_keys = {"train", "tuneV", "val_small", "calib"}
        assert set(splits.keys()) == expected_keys
        
        # Check approximate sizes
        assert len(splits["train"]) == 800
        assert len(splits["tuneV"]) == 100
        assert len(splits["val_small"]) == 50
        assert len(splits["calib"]) == 50
        
        # Check that all indices are used exactly once
        all_split_indices = []
        for split_indices in splits.values():
            all_split_indices.extend(split_indices)
        
        assert len(all_split_indices) == 1000
        assert set(all_split_indices) == set(indices)
    
    def test_reproducibility(self):
        """Test that same seed produces same splits."""
        indices = list(range(100))
        config = SplitConfig()
        
        splits1 = partition_indices(indices, config, seed=123)
        splits2 = partition_indices(indices, config, seed=123)
        
        for key in splits1:
            assert splits1[key] == splits2[key]
    
    def test_small_dataset(self):
        """Test partitioning with very small dataset."""
        indices = list(range(10))
        config = SplitConfig()
        
        splits = partition_indices(indices, config, seed=42)
        
        # Check that all indices are used
        all_split_indices = []
        for split_indices in splits.values():
            all_split_indices.extend(split_indices)
        
        assert len(all_split_indices) == 10
        assert set(all_split_indices) == set(indices)


class TestSaveLoadSplits:
    """Test save_splits and load_splits functions."""
    
    def test_save_and_load_splits(self):
        """Test saving and loading splits to/from JSON."""
        splits = {
            "train": [1, 2, 3, 4, 5],
            "val": [6, 7, 8],
            "test": [9, 10]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Test saving
            save_splits(temp_path, splits)
            assert os.path.exists(temp_path)
            
            # Test loading
            loaded_splits = load_splits(temp_path)
            
            # Check that loaded splits match original
            assert loaded_splits == splits
            
            # Check that values are integers (not floats from JSON)
            for split_name, indices in loaded_splits.items():
                assert all(isinstance(idx, int) for idx in indices)
                
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_creates_directory(self):
        """Test that save_splits creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "nested", "subdir", "splits.json")
            splits = {"train": [1, 2, 3]}
            
            save_splits(nested_path, splits)
            
            assert os.path.exists(nested_path)
            loaded = load_splits(nested_path)
            assert loaded == splits


class TestBuildCifarLtSplits:
    """Test the build_cifar_lt_splits function."""
    
    @patch('data.dataset.load_cifar')
    def test_build_cifar_lt_splits(self, mock_load_cifar):
        """Test the complete pipeline for building long-tailed CIFAR splits."""
        # Mock the load_cifar function
        mock_train_dataset = MagicMock()
        mock_train_dataset.targets = [0] * 100 + [1] * 100 + [2] * 100  # 3 classes, 100 each
        
        mock_test_dataset = MagicMock()
        
        mock_load_cifar.side_effect = [
            (mock_train_dataset, mock_train_dataset.targets, 3),  # train call
            (mock_test_dataset, [], 3)  # test call
        ]
        
        split_cfg = SplitConfig(train=0.8, tuneV=0.1, val_small=0.05, calib=0.05)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            datasets, splits, num_classes = build_cifar_lt_splits(
                root="/fake/root",
                name="cifar10",
                imb_factor=2.0,
                seed=42,
                split_cfg=split_cfg,
                save_path=temp_path
            )
            
            # Check that load_cifar was called twice (train and test)
            assert mock_load_cifar.call_count == 2
            
            # Check return values
            assert num_classes == 3
            assert isinstance(datasets, dict)
            assert isinstance(splits, dict)
            
            # Check dataset keys
            expected_keys = {"train", "tuneV", "val_small", "calib", "test"}
            assert set(datasets.keys()) == expected_keys
            
            # Check that train/tuneV/val_small/calib are Subset objects
            for key in ["train", "tuneV", "val_small", "calib"]:
                assert isinstance(datasets[key], Subset)
                assert datasets[key].dataset == mock_train_dataset
            
            # Check that test is the original test dataset
            assert datasets["test"] == mock_test_dataset
            
            # Check that splits file was created
            assert os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestCountPerClass:
    """Test the count_per_class function."""
    
    def test_count_per_class(self):
        """Test counting samples per class in a dataset subset."""
        # Create a mock base dataset
        mock_base_dataset = MagicMock()
        mock_base_dataset.targets = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3]  # 10 samples
        
        # Create a subset with specific indices
        subset_indices = [0, 2, 3, 5, 6, 8]  # classes: [0, 1, 1, 2, 2, 2]
        mock_subset = MagicMock()
        mock_subset.dataset = mock_base_dataset
        mock_subset.indices = subset_indices
        
        counts = count_per_class(mock_subset, num_classes=4)
        
        expected_counts = np.array([1, 2, 3, 0])  # class 0: 1, class 1: 2, class 2: 3, class 3: 0
        np.testing.assert_array_equal(counts, expected_counts)
    
    def test_count_per_class_empty_subset(self):
        """Test counting with empty subset."""
        mock_base_dataset = MagicMock()
        mock_base_dataset.targets = [0, 1, 2]
        
        mock_subset = MagicMock()
        mock_subset.dataset = mock_base_dataset
        mock_subset.indices = []
        
        counts = count_per_class(mock_subset, num_classes=3)
        
        expected_counts = np.array([0, 0, 0])
        np.testing.assert_array_equal(counts, expected_counts)
    
    def test_count_per_class_all_same_class(self):
        """Test counting when all samples belong to same class."""
        mock_base_dataset = MagicMock()
        mock_base_dataset.targets = [0, 1, 1, 1, 2]
        
        subset_indices = [1, 2, 3]  # all class 1
        mock_subset = MagicMock()
        mock_subset.dataset = mock_base_dataset
        mock_subset.indices = subset_indices
        
        counts = count_per_class(mock_subset, num_classes=3)
        
        expected_counts = np.array([0, 3, 0])
        np.testing.assert_array_equal(counts, expected_counts)


if __name__ == "__main__":
    pytest.main([__file__])