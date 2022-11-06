from transformers import DonutProcessor , VisionEncoderDecoderModel
from ..params import InferenceParams
from zenml.steps import step

@step(enable_cache=False)
def load_model_and_processor(params: InferenceParams) -> Output(
    processor=DonutProcessor,
    model=VisionEncoderDecoderModel
):
    model = VisionEncoderDecoderModel.from_pretrained(params.cheques_model)
    processor = DonutProcessor.from_pretrained(params.cheques_donut_processor)

    return processor, model
