from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def load_donut_model_and_processor(trained_model_repo):
    donut_processor = DonutProcessor.from_pretrained(trained_model_repo)
    model = VisionEncoderDecoderModel.from_pretrained(trained_model_repo)
    model.to(device)
    return donut_processor, model

def prepare_data_using_processor(donut_processor,image,task_prompt):
    ## Pass image through donut processor's feature extractor and retrieve image tensor
    pixel_values = donut_processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    ## Pass task prompt for document (cheque) parsing task to donut processor's tokenizer and retrieve the input_ids
    decoder_input_ids = donut_processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    decoder_input_ids = decoder_input_ids.to(device)

    return pixel_values, decoder_input_ids