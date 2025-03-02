from datasets import Dataset

prompts = ["Reset the alfworld environment, and execute the task given in the initial observation.",]*10000

dataset = Dataset.from_dict(mapping=dict(prompt=prompts))

dataset.push_to_hub(repo_id='crislmfroes/alfworld', split='train', private=False)