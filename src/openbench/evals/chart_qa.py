from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice
from inspect_ai.dataset import hf_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage


def generate_user_prompt(query: str, image: str) -> str:
    """
    Attribution: EleutherAI/lm-evaluation-harness prompt

    Source:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/chartqa/chartqa.yaml
    """
    PROMPT = f"""
    {query}
      Analyze the image and question carefully, using step-by-step reasoning.
      First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
      Final Answer: <answer>
      where <answer> follows the following instructions:
      - <answer> should should be a single phrase or number.
      - <answer> should not paraphrase or reformat the text in the image.
      - If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
      - If the question is a Yes/No question, <answer> should be Yes/No.
      - If <answer> is a number, it should not contain any units.
      - If <answer> is a percentage, it should include a % sign.
      - If <answer> is an entity, it should include the full label from the graph.
      IMPORTANT: Remember, to end your answer with Final Answer: <answer>.
    """
    return PROMPT

def record_to_sample(record: dict) -> Sample:
    input_list = [
        ChatMessageUser(content=[ContentText(text=record["query"])]),
        ChatMessageUser(content=[ContentImage(image=record["image"])]),
    ]
    return Sample(
        input=record["input"],
        metadata=record["metadata"],
        choices=record["choices"],
        target=record["target"],
    )

@task
def chart_qa():
    return Task(
        dataset=hf_dataset(
            "HuggingFaceM4/ChartQA", split="test", sample_fields=record_to_sample
        ),
        solver=None,
        scorer=None,
    )
