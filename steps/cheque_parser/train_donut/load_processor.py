from materializers.donut_processor_materializer import HFDonutMaterializer
from transformers import DonutProcessor
from params import DonutTrainParams
from zenml.steps import step

@step(enable_cache=False,output_materializers=HFDonutMaterializer)
def load_donut_processor(params: DonutTrainParams) -> DonutProcessor:
    """
    Load the processor consisting of feature_extractor & tokenizer for Donut Model
    Args:
        params (DonutTrainParams): Donut model specific parameters
    Returns:
        donut_processor (DonutProcessor): processor containing feature_extractor of encoder & tokenizer of decoder for Donut Model
    """

    donut_processor = DonutProcessor.from_pretrained(params.pretrained_ckpt)
    donut_processor.feature_extractor.size = params.image_size[::-1]
    donut_processor.feature_extractor.do_align_long_axis = False
    return donut_processor