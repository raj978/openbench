"""SWE-bench dataset loaders for OpenBench.

SWE-bench is a benchmark for evaluating Language Models on Software Engineering tasks.
It consists of real-world GitHub issues and their corresponding fixes.

Reference: https://github.com/princeton-nlp/SWE-bench
"""

from typing import Any, Dict
from inspect_ai.dataset import Dataset, Sample, hf_dataset

SWE_BENCH_INSTRUCTION = """
You are a software engineer tasked with fixing a bug in a codebase. You will be given:
1. A problem statement describing the issue
2. Relevant code context and files
3. Test cases that should pass after your fix

Please provide a solution that fixes the issue described. Your response should contain the exact code changes needed to resolve the problem.
""".strip()


def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert a SWE-bench record to a Sample for evaluation.

    Args:
        record: A SWE-bench record dictionary containing issue details.

    Returns:
        Sample: An Inspect AI Sample object.
    """
    # Create input prompt combining the problem statement and environment details
    input_text = f"""
Problem Statement:
{record.get("problem_statement", "")}

Repository: {record.get("repo", "")}
Version: {record.get("version", "")}
Environment Setup Commands:
{record.get("environment_setup_commit", "")}

Instance ID: {record.get("instance_id", "")}

{SWE_BENCH_INSTRUCTION}
""".strip()

    return Sample(
        id=record.get("instance_id", ""),
        input=input_text,
        target=record.get("patch", ""),
        metadata={
            "repo": record.get("repo", ""),
            "version": record.get("version", ""),
            "base_commit": record.get("base_commit", ""),
            "environment_setup_commit": record.get("environment_setup_commit", ""),
            "test_patch": record.get("test_patch", ""),
            "problem_statement": record.get("problem_statement", ""),
        },
    )


def get_swe_bench_dataset(variant: str = "lite") -> Dataset:
    """Get SWE-bench dataset for evaluation.

    Args:
        variant: Which SWE-bench variant to use. Options: "lite", "verified", "full"

    Returns:
        Dataset: The SWE-bench dataset.

    Raises:
        ValueError: If variant is not supported.
    """
    if variant not in ["lite", "verified", "full"]:
        raise ValueError(
            f"Unsupported SWE-bench variant: {variant}. Choose from: lite, verified, full"
        )

    # Map variant names to HuggingFace dataset paths
    dataset_paths = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }

    return hf_dataset(
        path=dataset_paths[variant],
        split="test",
        sample_fields=record_to_sample,
        name=f"swe-bench-{variant}",
    )
