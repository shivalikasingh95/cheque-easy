from zenml.steps import step
from datasets import Dataset,load_dataset
from params import InferenceParams
from PIL.Image import Image
from zenml.integrations.pillow.materializers import PillowImageMaterializer

@step(enable_cache=False,output_materializers=PillowImageMaterializer)
def import_inference_data(params: InferenceParams) -> Image:
    score_data = load_dataset(params.test_data_repo, split="test[0:10]")
    sample_input_image = score_data[2]['image']
    return sample_input_image
