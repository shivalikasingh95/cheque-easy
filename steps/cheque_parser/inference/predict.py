from typing import Dict
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from PIL.Image import Image
from PIL import PngImagePlugin
from zenml.steps import step
# from transformers import DonutProcessor, VisionEncoderDecoderModel
from typing import Dict
import base64
from datasets import Dataset
import io
import json
from io import BytesIO

def image_to_byte_array(image: Image) -> bytes:
      imgByteArr = io.BytesIO()
      image.save(imgByteArr, format=image.format)
      imgByteArr = imgByteArr.getvalue()
      return imgByteArr

def encode_pil_to_base64(pil_image):
    with BytesIO() as output_bytes:

        # Copy any text-only metadata
        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in pil_image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True

        pil_image.save(
            output_bytes, "PNG", pnginfo=(metadata if use_metadata else None)
        )
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), "utf-8")
    return "data:image/png;base64," + base64_str

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    data: Image,
) -> Dict:
    """Run an inference request against a prediction service"""    
    # encoded_image = base64.encodebytes(image_to_byte_array(data))
    print("type image data :",data)
    encoded_image = encode_pil_to_base64(data)
    
    #img_utf = encoded_image.decode('utf-8')
    #img_base64_str = json.dumps(img_utf)
    print("encoded_image:", type(encoded_image))
    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict(encoded_image)
    print("prediction:", prediction)
    return prediction


# def predict(model: VisionEncoderDecoderModel, 
#                 processor: DonutProcessor,
#                 image,
#                 task_prompt: str) -> Dict:

#     pixel_values = processor(image, return_tensors="pt").pixel_values
#     pixel_values = pixel_values.to(device)

#     decoder_input_ids = processor.tokenizer(task_prompt, 
#                     add_special_tokens=False, return_tensors="pt")["input_ids"]
#     decoder_input_ids = decoder_input_ids.to(device)


#     outputs = model.generate(pixel_values,
#                     decoder_input_ids=decoder_input_ids,
#                     max_length=model.decoder.config.max_position_embeddings,
#                     early_stopping=True,
#                     pad_token_id=processor.tokenizer.pad_token_id,
#                     eos_token_id=processor.tokenizer.eos_token_id,
#                     use_cache=True,
#                     num_beams=1,
#                     bad_word_ids=[[processor.tokenizer.unk_token_id]],
#                     return_dict_id_in_generate=True,
#                     output_scores=True,)

#     sequence = processor.batch_decoder(outputs.sequences)[0]
#     sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
#     sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

#     processed_cheque_details = processor.token2json(sequence)

#     return processed_cheque_details

    
