# from inspect_ai.task import Task, TaskState, TaskTarget
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.scorer import scorer, choice, Target, Score
from inspect_ai.solver import solver, multiple_choice


def collie():
    return Task(
        dataset=hf_dataset("Collie/collie", split="train"),
        solver=solver(multiple_choice()),
        scorer=scorer(choice()),
    )
