from transformers import VisionEncoderDecoderConfig
from params import DonutTrainParams
from materializers.config_materializer import DonutConfigMaterializer
from zenml.steps import step

@step(enable_cache=False,output_materializers=DonutConfigMaterializer)
def load_model_config(params: DonutTrainParams) -> VisionEncoderDecoderConfig:
    """
    Load the config for VisionEncoderDecoderModel since `Donut` is a type of VisionEncoderDecoderModel
    Args:
        ckpt_name (str): name of checkpoint to be loaded from the HuggingFace Hub (or local if processor exists locally)
    Returns:
        config (VisionEncoderDecoderConfig:): configuration to be used to instantiate `VisionEncoderDecoderModel`
    """
    config = VisionEncoderDecoderConfig.from_pretrained(params.pretrained_ckpt)
    config.encoder.image_size = params.image_size
    config.decoder.max_length = params.max_length

    return config