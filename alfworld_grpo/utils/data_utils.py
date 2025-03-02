import random
import json
from typing import List, Dict

from datasets import Dataset, load_dataset # type: ignore

from verifiers.utils.data_utils import format_prompt

def preprocess_dataset(dataset_name: str = "alfworld", 
                       split: str = "train",
                       system_prompt: str | None = None,
                       few_shot: List[Dict[str, str]] | None = None,
                       fewshot_prob: float = 1.0) -> Dataset:
    if dataset_name == "alfworld":
        dataset: Dataset = load_dataset("crislmfroes/alfworld")[split] # type: ignore
        dataset = dataset.map(lambda x: {
            "prompt": format_prompt(x["prompt"], system_prompt, few_shot, fewshot_prob),
        })
        return dataset
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for preprocess_dataset.")