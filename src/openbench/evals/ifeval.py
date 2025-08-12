"""
Instruction Following Evaluation (IFEval) for Large Language Models

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma,
Sujoy Basu, Yi Luan, Denny Zhou, Le Hou
https://arxiv.org/abs/2311.07911

Based on: https://github.com/google-research/google-research/tree/master/instruction_following_eval
with code adapted from this repository.

Evaluates a model's ability to follow specific instructions and constraints
in text generation tasks.

# eval all constraint types with a strict scorer type:
bench eval ifeval --model "groq/llama-3.1-8b-instant" -T scorer_type=strict
"""

from colorama import Fore, Style
from typing import Any
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, stderr
from inspect_ai.solver import solver, Solver
from inspect_ai.model import get_model
from inspect_ai.model import ChatMessageUser


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert IFEval dataset record to inspect Sample."""

    def create_list_of_not_none_dicts(
        dict_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            only_keep_not_none_values_in_dict(d) for d in dict_list if d is not None
        ]

    def only_keep_not_none_values_in_dict(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if v is not None}

    return Sample(
        input=record["prompt"],
        target=record.get("expected_output", ""),
        metadata={
            "key": record.get("key", ""),
            "prompt": record.get("prompt", ""),
            "instruction_id_list": record.get("instruction_id_list", ""),
            "kwargs": create_list_of_not_none_dicts(record.get("kwargs", [])),
        },
    )


# IFEval dataset from Hugging Face
dataset = hf_dataset(
    path="google/IFEval",
    split="train",
    sample_fields=record_to_sample,
)


@scorer(metrics=[accuracy(), stderr()])
def custom_scorer(scorer_type: str) -> Scorer:
    """Custom scorer for IFEval."""

    async def score(state, target: Target) -> Score:
        from openbench.scorers.ifeval.evaluation_lib import (
            test_instruction_following_strict,
            test_instruction_following_loose,
            read_single_prompt_dict,
        )

        inp = read_single_prompt_dict(state.metadata)

        prompt_to_response = {inp.prompt: state.messages[-1].content}

        if scorer_type == "strict":
            strict_result = test_instruction_following_strict(inp, prompt_to_response)
            score_value = 1.0 if strict_result.follow_all_instructions else 0.0
            state.metadata["scorer"] = "strict"
            state.metadata["explanation"] = str(strict_result.follow_instruction_list)
        elif scorer_type == "loose":
            loose_result = test_instruction_following_loose(inp, prompt_to_response)
            score_value = 1.0 if loose_result.follow_all_instructions else 0.0
            state.metadata["scorer"] = "loose"
            state.metadata["explanation"] = str(loose_result.follow_instruction_list)
        else:
            raise ValueError(f"Invalid scorer type: {scorer_type}")

        # TODO: the following was Claude generated code that was quite close to
        # the reported results for strict prompt scoring for GPT-4 on the
        # based on the original IFEval paper's reported scores.

        # TODO: the following code is commented out since the paper implies
        # that strict and loose scoring should be evaluated separately,
        # though keeping the code here for future refernece.

        # TODO: start of code
        # Calculate score using all() over both strict and loose results
        # score_value = (
        #     1.0
        #     if all(
        #         [
        #             strict_result.follow_all_instructions,
        #             loose_result.follow_all_instructions,
        #         ]
        #     )
        #     else 0.0
        # )
        # TODO: end of code

        return Score(
            value=score_value,
            explanation=f"using {state.metadata['scorer']} scorer result: {state.metadata['explanation']}",
        )

    return score


@solver
def custom_solver() -> Solver:
    """Custom solver for IFEval."""

    model = get_model()

    async def solve(state, generate):
        resp = await model.generate(
            input=[
                ChatMessageUser(content=state.input),
            ],
        )

        state.messages.append(resp.choices[0].message)
        return state

    return solve


@task
def ifeval(scorer_type: str | None = None) -> Task:
    """Inspect Task implementation for IFEval"""
    if scorer_type is None:
        print(
            f"{Fore.YELLOW}⚠️  No scorer type provided for IFEval through OpenBench. Defaulting to 'strict'.{Style.RESET_ALL}"
        )
        scorer_type = "strict"
    return Task(
        dataset=dataset,
        solver=custom_solver(),
        scorer=custom_scorer(scorer_type),
        config=GenerateConfig(),
    )
