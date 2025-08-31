"""SWE-bench scorer for OpenBench.

This scorer evaluates code fixes generated for SWE-bench tasks.
For now, it uses a simplified approach that compares the generated solution
against the target patch using text similarity and code structure analysis.

A full implementation would involve setting up the actual repository environment
and running the test cases, which is beyond the scope of this basic integration.
"""

import re
from typing import Set

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    CORRECT,
    INCORRECT,
    scorer,
)
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy()])
def swe_bench_scorer() -> Scorer:
    """Scorer for SWE-bench tasks.

    This is a simplified scorer that evaluates the quality of generated code fixes.
    It checks for:
    1. Presence of code changes
    2. Structural similarity to target patch
    3. Relevant keywords and patterns

    Returns:
        Scorer: The SWE-bench scorer function.
    """

    async def score(state: TaskState, target: Target) -> Score:
        """Score a model's response for a SWE-bench task.

        Args:
            state: The current task state containing model output and metadata.
            target: The target patch/solution.

        Returns:
            Score: The evaluation result.
        """
        completion = state.output.completion
        target_patch = str(target) if target else ""

        # Extract code from the completion
        extracted_code = extract_code_from_completion(completion)

        if not extracted_code:
            return Score(
                value=INCORRECT,
                answer=completion,
                explanation="No code changes found in the response.",
            )

        # Calculate similarity score based on multiple factors
        similarity_score = calculate_patch_similarity(extracted_code, target_patch)

        # Consider it correct if similarity is above threshold
        is_correct = similarity_score > 0.3  # Adjustable threshold

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=extracted_code,
            explanation=f"Patch similarity score: {similarity_score:.3f}. "
            f"Threshold for correctness: 0.3. "
            f"{'Passed' if is_correct else 'Failed'} similarity check.",
        )

    return score


def extract_code_from_completion(completion: str) -> str:
    """Extract code changes from the model's completion.

    Args:
        completion: The model's text response.

    Returns:
        str: Extracted code or empty string if none found.
    """
    # Look for code blocks
    code_block_patterns = [
        r"```(?:python|diff|patch)?\n(.*?)```",
        r"```\n(.*?)```",
        r"<code>\n(.*?)</code>",
        r"<patch>\n(.*?)</patch>",
    ]

    for pattern in code_block_patterns:
        matches = re.findall(pattern, completion, re.DOTALL)
        if matches:
            return matches[0].strip()

    # If no code blocks found, look for diff-like patterns
    lines = completion.split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        # Common indicators of code changes
        if any(
            line.strip().startswith(prefix)
            for prefix in ["+", "-", "@@", "def ", "class ", "import ", "from "]
        ):
            in_code = True
            code_lines.append(line)
        elif in_code and line.strip() == "":
            code_lines.append(line)
        elif in_code and not re.match(r"^[a-zA-Z\s]*$", line.strip()):
            code_lines.append(line)
        elif in_code:
            break

    return "\n".join(code_lines).strip()


def calculate_patch_similarity(generated: str, target: str) -> float:
    """Calculate similarity between generated and target patches.

    Args:
        generated: Generated code/patch.
        target: Target code/patch.

    Returns:
        float: Similarity score between 0 and 1.
    """
    if not generated or not target:
        return 0.0

    # Normalize whitespace and convert to lowercase for comparison
    gen_normalized = normalize_code(generated)
    target_normalized = normalize_code(target)

    # Calculate various similarity metrics
    scores = []

    # 1. Token overlap
    gen_tokens = set(gen_normalized.split())
    target_tokens = set(target_normalized.split())
    if target_tokens:
        token_overlap = len(gen_tokens & target_tokens) / len(target_tokens)
        scores.append(token_overlap)

    # 2. Line overlap (for diff-style patches)
    gen_lines = set(line.strip() for line in generated.split("\n") if line.strip())
    target_lines = set(line.strip() for line in target.split("\n") if line.strip())
    if target_lines:
        line_overlap = len(gen_lines & target_lines) / len(target_lines)
        scores.append(line_overlap)

    # 3. Function/class name overlap
    gen_names = extract_names(generated)
    target_names = extract_names(target)
    if target_names:
        name_overlap = len(gen_names & target_names) / len(target_names)
        scores.append(name_overlap)

    # Return weighted average
    return sum(scores) / len(scores) if scores else 0.0


def normalize_code(code: str) -> str:
    """Normalize code for comparison.

    Args:
        code: Code string to normalize.

    Returns:
        str: Normalized code.
    """
    # Remove extra whitespace and normalize
    lines = [line.strip() for line in code.split("\n")]
    normalized = " ".join(line for line in lines if line)
    return normalized.lower()


def extract_names(code: str) -> Set[str]:
    """Extract function and class names from code.

    Args:
        code: Code string to analyze.

    Returns:
        Set[str]: Set of extracted names.
    """
    names = set()

    # Extract function names
    func_pattern = r"def\s+(\w+)\s*\("
    names.update(re.findall(func_pattern, code))

    # Extract class names
    class_pattern = r"class\s+(\w+)\s*[:\(]"
    names.update(re.findall(class_pattern, code))

    # Extract variable assignments (simple cases)
    var_pattern = r"(\w+)\s*="
    names.update(re.findall(var_pattern, code))

    return names
