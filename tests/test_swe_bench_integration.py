"""Integration test for SWE-bench evaluation."""

import pytest
from unittest.mock import patch, MagicMock
from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput
from openbench.evals.swebench import swe_bench_lite
from openbench.datasets.swebench import record_to_sample


class TestSWEBenchIntegration:
    """Integration tests for SWE-bench evaluation."""

    def test_record_to_sample_conversion(self):
        """Test conversion of SWE-bench record to Sample."""
        # Mock SWE-bench record
        record = {
            "instance_id": "test_123",
            "problem_statement": "Fix the bug in function X",
            "repo": "test/repo",
            "version": "1.0.0",
            "base_commit": "abc123",
            "environment_setup_commit": "def456",
            "test_patch": "test code here",
            "patch": "- old_code\n+ new_code",
        }

        sample = record_to_sample(record)

        assert sample.id == "test_123"
        assert "Fix the bug in function X" in sample.input
        assert "test/repo" in sample.input
        assert sample.target == "- old_code\n+ new_code"
        assert sample.metadata["repo"] == "test/repo"
        assert sample.metadata["version"] == "1.0.0"

    @patch("openbench.datasets.swebench.hf_dataset")
    def test_task_creation(self, mock_hf_dataset):
        """Test that SWE-bench task can be created."""
        # Mock the dataset with samples
        from inspect_ai.dataset import MemoryDataset, Sample

        mock_dataset = MemoryDataset(
            [Sample(id="test", input="test input", target="test target")]
        )
        mock_hf_dataset.return_value = mock_dataset

        task = swe_bench_lite()

        assert task.name == "swe-bench-lite"
        assert task.dataset == mock_dataset
        assert task.scorer is not None
        assert task.solver is not None
        assert task.config is not None

        # Verify dataset was called with correct parameters
        mock_hf_dataset.assert_called_once_with(
            path="princeton-nlp/SWE-bench_Lite",
            split="test",
            sample_fields=record_to_sample,
            name="swe-bench-lite",
        )

    @pytest.mark.asyncio
    async def test_evaluation_pipeline(self):
        """Test the complete evaluation pipeline with mocked components."""
        # Create a mock sample
        sample = Sample(
            id="test_instance",
            input="Fix this bug: def broken_func(): return None",
            target="def fixed_func(): return 42",
            metadata={
                "repo": "test/repo",
                "problem_statement": "Function should return 42",
            },
        )

        # Create mock task state
        state = MagicMock(spec=TaskState)
        state.output = MagicMock(spec=ModelOutput)
        state.output.completion = """
        Here's the fix:
        
        ```python
        def fixed_func():
            return 42
        ```
        """
        state.metadata = sample.metadata

        # Test the scorer
        from openbench.scorers.swe_bench import swe_bench_scorer

        scorer = swe_bench_scorer()

        score = await scorer(state, sample.target)

        assert score.value in ["C", "I"]  # CORRECT or INCORRECT
        assert "def fixed_func():" in score.answer
        assert "similarity score" in score.explanation.lower()

    def test_dataset_variant_validation(self):
        """Test that dataset validates variants correctly."""
        from openbench.datasets.swebench import get_swe_bench_dataset

        # Valid variants should not raise
        valid_variants = ["lite", "verified", "full"]
        for variant in valid_variants:
            try:
                # This will fail due to network, but should not fail validation
                get_swe_bench_dataset(variant)
            except Exception as e:
                # Should fail on network/HF issues, not validation
                assert "Unsupported SWE-bench variant" not in str(e)

        # Invalid variant should raise ValueError
        with pytest.raises(ValueError, match="Unsupported SWE-bench variant"):
            get_swe_bench_dataset("invalid")

    def test_cli_integration(self):
        """Test that benchmarks are properly registered and discoverable."""
        from openbench.config import get_benchmark_metadata

        # Test all three variants are registered
        for variant in ["swe_bench_lite", "swe_bench_verified", "swe_bench_full"]:
            metadata = get_benchmark_metadata(variant)
            assert metadata is not None
            assert metadata.name.startswith("SWE-bench")
            assert "software-engineering" in metadata.tags
            assert metadata.module_path == "openbench.evals.swebench"

    @patch("openbench.datasets.swebench.hf_dataset")
    def test_different_variants(self, mock_hf_dataset):
        """Test that different SWE-bench variants load correctly."""
        from openbench.evals.swebench import swe_bench_verified, swe_bench_full
        from inspect_ai.dataset import MemoryDataset, Sample

        mock_dataset = MemoryDataset(
            [Sample(id="test", input="test input", target="test target")]
        )
        mock_hf_dataset.return_value = mock_dataset

        # Test verified variant
        task_verified = swe_bench_verified()
        assert task_verified.name == "swe-bench-verified"

        # Test full variant
        task_full = swe_bench_full()
        assert task_full.name == "swe-bench-full"

        # Verify correct dataset paths were used
        calls = mock_hf_dataset.call_args_list
        paths = [call[1]["path"] for call in calls]
        assert "princeton-nlp/SWE-bench_Verified" in paths
        assert "princeton-nlp/SWE-bench" in paths
