from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    PreTrainedModel
)
from params import DonutTrainParams
from zenml.steps import step

@step(enable_cache=True)
def load_vision_encoder_decoder_model(
    params: DonutTrainParams,
    vis_enc_dec_config: VisionEncoderDecoderConfig) -> PreTrainedModel:
    """
    Donut falls under VisionEncoderDecoder class of models of Hugging Face Transformers.
    So we will load VisionEncoderDecoderModel
    Args:
        params (DonutTrainParams): Donut model specific parameters
        vis_enc_dec_config (VisionEncoderDecoderConfig): configuration for instantiating pre-trained Donut model  
    Returns:
        model (PreTrainedModel): pretrained model which will be used for fine-tuning 
    """

    model = VisionEncoderDecoderModel.from_pretrained(params.pretrained_ckpt, config=vis_enc_dec_config)
    
    return model

