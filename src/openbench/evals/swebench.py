"""SWE-bench evaluations for OpenBench.

SWE-bench is a benchmark for evaluating Language Models on Software Engineering tasks.
It consists of real-world GitHub issues and their corresponding fixes.

Reference: https://github.com/princeton-nlp/SWE-bench
"""

from inspect_ai import task, Task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.swebench import get_swe_bench_dataset
from openbench.scorers.swe_bench import swe_bench_scorer


@task
def swe_bench_lite() -> Task:
    """SWE-bench Lite evaluation task.

    A smaller, curated subset of 300 instances from SWE-bench for faster evaluation.
    """
    return Task(
        dataset=get_swe_bench_dataset(variant="lite"),
        solver=[generate()],
        scorer=swe_bench_scorer(),
        name="swe-bench-lite",
        config=GenerateConfig(
            temperature=0.2,
            max_tokens=2048,
        ),
    )


@task
def swe_bench_verified() -> Task:
    """SWE-bench Verified evaluation task.

    A subset of SWE-bench instances that have been human-verified for quality.
    """
    return Task(
        dataset=get_swe_bench_dataset(variant="verified"),
        solver=[generate()],
        scorer=swe_bench_scorer(),
        name="swe-bench-verified",
        config=GenerateConfig(
            temperature=0.2,
            max_tokens=2048,
        ),
    )


@task
def swe_bench_full() -> Task:
    """SWE-bench Full evaluation task.

    The complete SWE-bench dataset with over 2,000 instances.
    This is computationally intensive and may take a long time to run.
    """
    return Task(
        dataset=get_swe_bench_dataset(variant="full"),
        solver=[generate()],
        scorer=swe_bench_scorer(),
        name="swe-bench-full",
        config=GenerateConfig(
            temperature=0.2,
            max_tokens=2048,
        ),
    )
