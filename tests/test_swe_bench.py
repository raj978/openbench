"""Tests for SWE-bench implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from inspect_ai.solver import TaskState
from inspect_ai.model import ModelOutput
from openbench.scorers.swe_bench import (
    swe_bench_scorer,
    extract_code_from_completion,
    calculate_patch_similarity,
    normalize_code,
    extract_names,
)


class TestSWEBenchScorer:
    """Test the SWE-bench scorer functionality."""

    @pytest.mark.asyncio
    async def test_swe_bench_scorer_with_code(self):
        """Test scorer with a completion containing code."""
        scorer = swe_bench_scorer()
        
        # Mock state with code in completion
        state = MagicMock(spec=TaskState)
        state.output = MagicMock(spec=ModelOutput)
        state.output.completion = """
        Here's the fix for the issue:
        
        ```python
        def fix_function():
            return "fixed"
        ```
        """
        
        target = "def fix_function():\n    return 'fixed'"
        
        score = await scorer(state, target)
        
        assert score.value in ["C", "I"]  # CORRECT or INCORRECT
        assert "fix_function" in score.answer

    @pytest.mark.asyncio
    async def test_swe_bench_scorer_no_code(self):
        """Test scorer with completion containing no code."""
        scorer = swe_bench_scorer()
        
        state = MagicMock(spec=TaskState)
        state.output = MagicMock(spec=ModelOutput)
        state.output.completion = "This is just text with no code."
        
        target = "def some_function(): pass"
        
        score = await scorer(state, target)
        
        assert score.value == "I"  # INCORRECT
        assert "No code changes found" in score.explanation


class TestCodeExtraction:
    """Test code extraction utilities."""

    def test_extract_code_from_python_block(self):
        """Test extracting code from python code block."""
        completion = """
        Here's the solution:
        
        ```python
        def hello():
            print("world")
        ```
        """
        
        result = extract_code_from_completion(completion)
        assert "def hello():" in result
        assert "print" in result

    def test_extract_code_from_diff_block(self):
        """Test extracting code from diff block."""
        completion = """
        ```diff
        + def new_function():
        +     return True
        - def old_function():
        -     return False
        ```
        """
        
        result = extract_code_from_completion(completion)
        assert "+ def new_function():" in result
        assert "- def old_function():" in result

    def test_extract_code_no_blocks(self):
        """Test extracting code when no code blocks present."""
        completion = "This is just text without any code blocks."
        
        result = extract_code_from_completion(completion)
        assert result == ""

    def test_extract_code_from_diff_patterns(self):
        """Test extracting code from diff-like patterns."""
        completion = """
        Here are the changes:
        
        + def added_function():
        +     return "new"
        
        - def removed_function():
        -     return "old"
        """
        
        result = extract_code_from_completion(completion)
        assert "+ def added_function():" in result


class TestPatchSimilarity:
    """Test patch similarity calculation."""

    def test_identical_patches(self):
        """Test similarity of identical patches."""
        patch1 = "def hello(): print('world')"
        patch2 = "def hello(): print('world')"
        
        similarity = calculate_patch_similarity(patch1, patch2)
        assert similarity == 1.0

    def test_different_patches(self):
        """Test similarity of completely different patches."""
        patch1 = "def hello(): print('world')"
        patch2 = "class MyClass: pass"
        
        similarity = calculate_patch_similarity(patch1, patch2)
        assert 0.0 <= similarity <= 1.0

    def test_similar_patches(self):
        """Test similarity of similar patches."""
        patch1 = "def hello(): print('world')"
        patch2 = "def hello(): print('universe')"
        
        similarity = calculate_patch_similarity(patch1, patch2)
        assert 0.3 <= similarity <= 0.9  # Should be similar but not identical

    def test_empty_patches(self):
        """Test similarity with empty patches."""
        similarity = calculate_patch_similarity("", "")
        assert similarity == 0.0
        
        similarity = calculate_patch_similarity("def hello(): pass", "")
        assert similarity == 0.0


class TestCodeNormalization:
    """Test code normalization utilities."""

    def test_normalize_code(self):
        """Test code normalization."""
        code = """
        def hello():
            print("world")
        """
        
        normalized = normalize_code(code)
        assert normalized == "def hello(): print(\"world\")"

    def test_extract_names(self):
        """Test name extraction from code."""
        code = """
        def my_function():
            var = 42
            return var
        
        class MyClass:
            pass
        """
        
        names = extract_names(code)
        assert "my_function" in names
        assert "MyClass" in names
        assert "var" in names