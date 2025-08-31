#!/usr/bin/env python3
"""Test script for SWE-bench implementation."""

import sys
import os
sys.path.insert(0, 'src')

from openbench.datasets.swebench import get_swebench_dataset
from openbench.scorers.swebench import swebench_scorer
from openbench.evals.swebench import swebench_lite


def test_dataset_loading():
    """Test that SWE-bench dataset loads correctly."""
    print("Testing SWE-bench dataset loading...")
    
    try:
        dataset = get_swebench_dataset(variant="lite", data_dir="./swebench_data")
        print(f"✓ Dataset loaded successfully")
        
        # Check dataset contents
        samples = list(dataset)
        print(f"✓ Dataset contains {len(samples)} samples")
        
        if samples:
            sample = samples[0]
            print(f"✓ First sample ID: {sample.id}")
            print(f"✓ First sample input length: {len(sample.input)} characters")
            print(f"✓ First sample target length: {len(sample.target)} characters")
            print(f"✓ Sample metadata keys: {list(sample.metadata.keys())}")
            
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_scorer():
    """Test the SWE-bench scorer."""
    print("\nTesting SWE-bench scorer...")
    
    try:
        scorer = swebench_scorer()
        print("✓ Scorer created successfully")
        return True
    except Exception as e:
        print(f"✗ Scorer creation failed: {e}")
        return False


def test_task_creation():
    """Test that SWE-bench task can be created."""
    print("\nTesting SWE-bench task creation...")
    
    try:
        task = swebench_lite(data_dir="./swebench_data")
        print("✓ Task created successfully")
        print(f"✓ Task name: {task.name}")
        print(f"✓ Task has dataset: {task.dataset is not None}")
        print(f"✓ Task has solver: {task.solver is not None}")
        print(f"✓ Task has scorer: {task.scorer is not None}")
        return True
    except Exception as e:
        print(f"✗ Task creation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("SWE-bench Implementation Test Suite")
    print("=" * 40)
    
    tests = [
        test_dataset_loading,
        test_scorer,
        test_task_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SWE-bench implementation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())