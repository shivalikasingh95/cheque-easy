from datasets import load_dataset
from zenml.steps import step
from params import DataParams

@step(enable_cache=False)
def push_dataset_to_hf_hub(params: DataParams, train_md: bool, val_md: bool, test_md: bool) -> None:
    dataset = load_dataset("imagefolder", data_dir=params.hf_data_dir)
    dataset.push_to_hub(params.hf_dataset_repo)