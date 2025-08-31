"""Scorer for SWE-bench software engineering tasks."""

import re
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    CORRECT,
    INCORRECT,
    stderr,
    scorer,
)
from inspect_ai.solver import TaskState


def extract_code_from_response(response: str) -> str:
    """Extract code content from model response."""
    # Try to find code blocks first
    code_block_pattern = r"```(?:python|diff|patch)?\n?(.*?)\n?```"
    code_blocks = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)

    if code_blocks:
        # Return the largest code block
        return max(code_blocks, key=len).strip()

    # If no code blocks, try to find diff-like patterns
    diff_pattern = r"(?:^|\n)([\+\-\@].*?)(?=\n(?:[^\+\-\@]|$)|$)"
    diff_lines = re.findall(diff_pattern, response, re.MULTILINE)

    if diff_lines:
        return "\n".join(diff_lines).strip()

    # Fallback: return the whole response cleaned up
    return response.strip()


def normalize_code(code: str) -> str:
    """Normalize code for comparison."""
    # Remove leading/trailing whitespace
    code = code.strip()

    # Normalize line endings
    code = code.replace("\r\n", "\n").replace("\r", "\n")

    # Remove empty lines at start and end
    lines = code.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def calculate_patch_similarity(predicted: str, target: str) -> float:
    """Calculate similarity between predicted and target patches."""
    pred_normalized = normalize_code(extract_code_from_response(predicted))
    target_normalized = normalize_code(target)

    if not target_normalized:
        # If target is empty, score based on whether prediction is also empty
        return 1.0 if not pred_normalized else 0.0

    if not pred_normalized:
        return 0.0

    # Simple similarity based on common lines
    pred_lines = set(pred_normalized.split("\n"))
    target_lines = set(target_normalized.split("\n"))

    if not target_lines:
        return 1.0 if not pred_lines else 0.0

    intersection = pred_lines.intersection(target_lines)
    union = pred_lines.union(target_lines)

    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 1.0

    # Also consider exact match bonus
    exact_match = 1.0 if pred_normalized == target_normalized else 0.0

    # Weighted combination: 70% Jaccard + 30% exact match bonus
    return 0.7 * jaccard + 0.3 * exact_match


@scorer(metrics=[accuracy(), stderr()])
def swebench_scorer() -> Scorer:
    """
    Scorer for SWE-bench tasks. Evaluates generated code patches against target patches.

    Returns:
        Scorer: The SWE-bench scoring function.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Extract the predicted code/patch from the model output
        predicted = state.output.completion
        target_patch = target.text

        # Calculate patch similarity
        similarity = calculate_patch_similarity(predicted, target_patch)

        # Determine if this is correct (using a threshold)
        # For SWE-bench, we're generous - if there's significant similarity, consider it correct
        is_correct = similarity > 0.5

        # Create score with metadata
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=extract_code_from_response(predicted),
            explanation=f"Patch similarity: {similarity:.3f}",
            metadata={
                "patch_similarity": similarity,
                "extracted_code": extract_code_from_response(predicted),
                "target_patch": target_patch,
            },
        )

    return score
