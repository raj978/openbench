# SWE-bench dataset integration for OpenBench
# Reference: https://github.com/SWE-bench/SWE-bench

import os
import json
from typing import List, Dict, Any
from openbench.datasets import Dataset

class SWEbenchDataset(Dataset):
    """
    SWE-bench dataset loader for OpenBench.
    Supports: Full, Verified, and Lite variants.
    """
    def __init__(self, variant: str = "full", data_dir: str = None):
        assert variant in {"full", "verified", "lite"}, f"Unknown variant: {variant}"
        self.variant = variant
        self.data_dir = data_dir or os.environ.get("SWEBENCH_DATA_DIR", "./swebench_data")
        self.data = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        filename = {
            "full": "swe-bench.jsonl",
            "verified": "swe-bench-verified.jsonl",
            "lite": "swe-bench-lite.jsonl"
        }[self.variant]
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SWE-bench data file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def name(self):
        return f"swe-bench-{self.variant}"
