"""SWE-bench evaluation tasks for software engineering benchmarks."""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from typing import Optional

from openbench.datasets.swebench import get_swebench_dataset
from openbench.scorers.swebench import swebench_scorer


@task
def swebench_full(data_dir: Optional[str] = None) -> Task:
    """SWE-bench Full: Complete dataset with 2,294 tasks."""
    return Task(
        dataset=get_swebench_dataset(variant="full", data_dir=data_dir),
        solver=[generate()],
        scorer=swebench_scorer(),
        name="swebench_full",
        config=GenerateConfig(
            temperature=0.0,  # Use deterministic generation for code
            max_tokens=4096,  # Allow longer responses for code patches
        ),
    )


@task
def swebench_verified(data_dir: Optional[str] = None) -> Task:
    """SWE-bench Verified: Human-validated subset of SWE-bench."""
    return Task(
        dataset=get_swebench_dataset(variant="verified", data_dir=data_dir),
        solver=[generate()],
        scorer=swebench_scorer(),
        name="swebench_verified",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=4096,
        ),
    )


@task
def swebench_lite(data_dir: Optional[str] = None) -> Task:
    """SWE-bench Lite: Smaller subset with 300 tasks for faster evaluation."""
    return Task(
        dataset=get_swebench_dataset(variant="lite", data_dir=data_dir),
        solver=[generate()],
        scorer=swebench_scorer(),
        name="swebench_lite",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=4096,
        ),
    )


# Alias for backward compatibility and convenience
@task
def swebench(variant: str = "lite", data_dir: Optional[str] = None) -> Task:
    """SWE-bench: Software engineering benchmark with configurable variant.

    Args:
        variant: Which SWE-bench variant to use ("full", "verified", "lite")
        data_dir: Directory containing the SWE-bench data files
    """
    return Task(
        dataset=get_swebench_dataset(variant=variant, data_dir=data_dir),
        solver=[generate()],
        scorer=swebench_scorer(),
        name=f"swebench_{variant}",
        config=GenerateConfig(
            temperature=0.0,
            max_tokens=4096,
        ),
    )
