"""Tests for SWE-bench integration."""

import pytest
import tempfile
import os
import json
from pathlib import Path

from openbench.datasets.swebench import get_swebench_dataset, swebench_record_to_sample
from openbench.scorers.swebench import swebench_scorer, extract_code_from_response, calculate_patch_similarity
from openbench.evals.swebench import swebench_lite


def test_swebench_record_to_sample():
    """Test converting SWE-bench record to Sample."""
    record = {
        "instance_id": "test_1",
        "repo": "test/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug in the add function.",
        "patch": "def add(a, b):\n    return a + b",
        "test_patch": "def test_add():\n    assert add(-1, -2) == -3",
        "created_at": "2024-01-01",
        "FAIL_TO_PASS": ["test_add"],
        "PASS_TO_PASS": []
    }
    
    sample = swebench_record_to_sample(record)
    
    assert sample.id == "test_1"
    assert "Fix the bug in the add function." in sample.input
    assert sample.target == "def add(a, b):\n    return a + b"
    assert sample.metadata["repo"] == "test/repo"
    assert sample.metadata["base_commit"] == "abc123"


def test_extract_code_from_response():
    """Test extracting code from model responses."""
    # Test with code blocks
    response_with_block = """Here's the fix:
```python
def add(a, b):
    return a + b
```
This should work."""
    
    extracted = extract_code_from_response(response_with_block)
    assert "def add(a, b):" in extracted
    assert "return a + b" in extracted
    
    # Test with diff patterns
    response_with_diff = """The fix is:
+ def add(a, b):
+     return a + b
- def add(a, b):
-     return a - b"""
    
    extracted = extract_code_from_response(response_with_diff)
    assert "+ def add(a, b):" in extracted


def test_calculate_patch_similarity():
    """Test patch similarity calculation."""
    # Test exact match
    patch1 = "def add(a, b):\n    return a + b"
    patch2 = "def add(a, b):\n    return a + b"
    similarity = calculate_patch_similarity(patch1, patch2)
    assert similarity == 1.0
    
    # Test partial match
    patch1 = "def add(a, b):\n    return a + b"
    patch2 = "def add(a, b):\n    return a - b"
    similarity = calculate_patch_similarity(patch1, patch2)
    assert 0 < similarity < 1
    
    # Test no match
    patch1 = "def add(a, b):\n    return a + b"
    patch2 = "def multiply(a, b):\n    return a * b"
    similarity = calculate_patch_similarity(patch1, patch2)
    assert similarity >= 0


def test_swebench_dataset_loading():
    """Test SWE-bench dataset loading with temporary data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data file
        test_data = [
            {
                "instance_id": "test_1",
                "repo": "test/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug in the add function.",
                "patch": "def add(a, b):\n    return a + b",
                "test_patch": "def test_add():\n    assert add(-1, -2) == -3",
                "created_at": "2024-01-01",
                "FAIL_TO_PASS": ["test_add"],
                "PASS_TO_PASS": []
            }
        ]
        
        test_file = Path(tmpdir) / "swe-bench-lite.jsonl"
        with open(test_file, "w") as f:
            for record in test_data:
                f.write(json.dumps(record) + "\n")
        
        # Test loading
        dataset = get_swebench_dataset(variant="lite", data_dir=tmpdir)
        samples = list(dataset)
        
        assert len(samples) == 1
        assert samples[0].id == "test_1"


def test_swebench_dataset_file_not_found():
    """Test SWE-bench dataset loading with missing file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError) as exc_info:
            get_swebench_dataset(variant="lite", data_dir=tmpdir)
        
        assert "SWE-bench data file not found" in str(exc_info.value)


def test_swebench_dataset_invalid_variant():
    """Test SWE-bench dataset loading with invalid variant."""
    with pytest.raises(ValueError) as exc_info:
        get_swebench_dataset(variant="invalid")
    
    assert "Unknown variant: invalid" in str(exc_info.value)


def test_swebench_scorer_creation():
    """Test that SWE-bench scorer can be created."""
    scorer = swebench_scorer()
    assert scorer is not None


def test_swebench_task_creation():
    """Test that SWE-bench task can be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal test data
        test_file = Path(tmpdir) / "swe-bench-lite.jsonl"
        with open(test_file, "w") as f:
            record = {
                "instance_id": "test_1",
                "repo": "test/repo",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug.",
                "patch": "def fix():\n    pass",
                "test_patch": "",
                "created_at": "2024-01-01",
                "FAIL_TO_PASS": [],
                "PASS_TO_PASS": []
            }
            f.write(json.dumps(record) + "\n")
        
        task = swebench_lite(data_dir=tmpdir)
        
        assert task is not None
        assert task.name == "swebench_lite"
        assert task.dataset is not None
        assert task.solver is not None
        assert task.scorer is not None