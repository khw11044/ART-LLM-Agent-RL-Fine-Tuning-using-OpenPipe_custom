import random

from typing import List, Literal, Optional
from datasets import Dataset, load_dataset

from tasks.email.model import Scenario

def load_training_scenarios(
    split: Literal["train", "test"] = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
    SCENARIO_DATASET_REPO_ID: Optional[str] = "corbt/enron_emails_sample_questions",
) -> List[Scenario]:
    """Load training scenarios from Hugging Face dataset"""
    
    print(f"Loading {split} scenarios from Hugging Face...")
    dataset: Dataset = load_dataset(SCENARIO_DATASET_REPO_ID, split=split)

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if shuffle or (seed is not None):
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()

    # Convert each row to a Scenario object
    scenarios = [Scenario(**row, split=split) for row in dataset]

    if max_messages is not None:
        scenarios = [s for s in scenarios if len(s.message_ids) <= max_messages]

    if shuffle:
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(scenarios)
        else:
            random.shuffle(scenarios)

    if limit is not None:
        scenarios = scenarios[:limit]

    print(f"Loaded {len(scenarios)} scenarios.")
    return scenarios

if __name__=="__main__":

    # Load training scenarios
    training_scenarios = load_training_scenarios(
        split="train", limit=50, max_messages=1, shuffle=True, seed=42,
        SCENARIO_DATASET_REPO_ID="corbt/enron_emails_sample_questions"
    )

    print("Email search environment created with full Enron dataset!")
    print(
        f"Database contains the complete email dataset, loaded {len(training_scenarios)} training scenarios."
    )

    # print first scenario
    print("\nSample scenario")
    print("id:", training_scenarios[0].id)
    print("question:", training_scenarios[0].question)
    print("answer:", training_scenarios[0].answer)
    print("message_ids:", training_scenarios[0].message_ids)
    print("how_realistic:", training_scenarios[0].how_realistic)
    print("inbox_address:", training_scenarios[0].inbox_address)
    print("query_date:", training_scenarios[0].query_date)
    print("split:", training_scenarios[0].split)