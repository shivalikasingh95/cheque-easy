from transformers import (
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
    DonutProcessor,
    PreTrainedModel
)
from params import DonutTrainParams
from zenml.steps import step

@step(enable_cache=False)
def load_vision_encoder_decoder_model(params: DonutTrainParams,
    vis_enc_dec_config: VisionEncoderDecoderConfig, donut_processor: DonutProcessor) -> PreTrainedModel:
    """
    Load the processor consisting of feature_extractor & tokenizer for Donut Model
    Args:
        params (DonutTrainParams): Donut model specific parameters
        vis_enc_dec_config (VisionEncoderDecoderConfig): configuration for instantiating pre-trained Donut model
        donut_processor (DonutProcessor): Donut model specific processor   
    Returns:
        model (PreTrainedModel): pretrained model which will be used for fine-tuning 
    """

    model = VisionEncoderDecoderModel.from_pretrained(params.pretrained_ckpt)
    model.config.pad_token_id = donut_processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = donut_processor.tokenizer.convert_tokens_to_ids([params.task_start_token])[0]
    return model

