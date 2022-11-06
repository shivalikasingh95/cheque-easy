from datasets import load_dataset, DatasetDict
from params import DonutTrainParams
from zenml.steps import step

@step(enable_cache=False)
def load_data(params: DonutTrainParams) -> DatasetDict:
    """
    Load the dataset that will be used for fine-tuning donut
    Args:
        dataset_name (str): Dataset to be loaded for model training 
    Returns:
        dataset (DatasetDict): cheques dataset that will be used for modeling
    """
    print("params:",params)
    dataset = load_dataset(params.dataset)
    return dataset 