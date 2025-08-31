# SWE-bench evaluation integration for OpenBench
# Reference: https://github.com/SWE-bench/SWE-bench

from openbench.evals import Eval
from openbench.datasets.swebench import SWEbenchDataset

class SWEbenchEval(Eval):
    """
    SWE-bench evaluation logic for OpenBench.
    """
    def __init__(self, variant: str = "full", data_dir: str = None):
        self.dataset = SWEbenchDataset(variant=variant, data_dir=data_dir)
        self.variant = variant

    def evaluate(self, model, *args, **kwargs):
        results = []
        for task in self.dataset:
            # Each task contains: repo, commit, patch, test, etc.
            # The model is expected to generate a patch or code change.
            # Here, we just pass the task to the model and collect the result.
            result = model.solve_swebench_task(task)
            results.append({
                "task_id": task.get("instance_id"),
                "success": result.get("success", False),
                "model_output": result.get("output"),
                "expected": task.get("patch")
            })
        return results

    @property
    def name(self):
        return f"swe-bench-{self.variant}"
