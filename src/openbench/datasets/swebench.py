"""SWE-bench dataset loaders for software engineering benchmarks."""

import os
from typing import Any, Dict, Optional
from inspect_ai.dataset import Dataset, Sample, json_dataset


def swebench_record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert SWE-bench record to Sample format."""
    # Extract the problem statement and requirements
    problem_statement = record.get("problem_statement", "")

    # Build a comprehensive input prompt for the model
    input_text = f"""You are a software engineer tasked with fixing a bug in a codebase.

**Problem Statement:**
{problem_statement}

**Repository:** {record.get("repo", "")}
**Base Commit:** {record.get("base_commit", "")}

**Instructions:**
1. Analyze the problem described above
2. Understand the codebase structure and identify the root cause
3. Propose a fix by providing the modified code
4. Ensure your solution addresses the specific issue mentioned

Please provide your solution as a code patch that can be applied to fix the issue."""

    return Sample(
        id=record.get("instance_id", ""),
        input=input_text,
        target=record.get("patch", ""),
        metadata={
            "repo": record.get("repo", ""),
            "base_commit": record.get("base_commit", ""),
            "test_patch": record.get("test_patch", ""),
            "created_at": record.get("created_at", ""),
            "FAIL_TO_PASS": record.get("FAIL_TO_PASS", []),
            "PASS_TO_PASS": record.get("PASS_TO_PASS", []),
        },
    )


def get_swebench_dataset(variant: str = "lite", data_dir: Optional[str] = None) -> Dataset:
    """Load SWE-bench dataset for evaluation.

    Args:
        variant: Which SWE-bench variant to load ("full", "verified", "lite")
        data_dir: Directory containing the SWE-bench data files

    Returns:
        Dataset: The SWE-bench dataset

    Raises:
        FileNotFoundError: If the data file is not found
        ValueError: If variant is not supported
    """
    if variant not in {"full", "verified", "lite"}:
        raise ValueError(f"Unknown variant: {variant}. Supported: full, verified, lite")

    # Default data directory
    if data_dir is None:
        data_dir = os.environ.get("SWEBENCH_DATA_DIR", "./swebench_data")

    # Map variant to filename
    filename_map = {
        "full": "swe-bench.jsonl",
        "verified": "swe-bench-verified.jsonl",
        "lite": "swe-bench-lite.jsonl",
    }

    filepath = os.path.join(data_dir, filename_map[variant])

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"SWE-bench data file not found: {filepath}\n"
            f"Please download the {variant} dataset and place it in {data_dir}\n"
            f"You can download from: https://github.com/SWE-bench/SWE-bench"
        )

    return json_dataset(
        json_file=filepath,
        sample_fields=swebench_record_to_sample,
    )
