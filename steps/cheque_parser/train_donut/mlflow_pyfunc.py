from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict
from PIL import Image
import base64
from io import BytesIO
import json
import re

TASK_PROMPT = "<s_cord-v2>"

class DonutModel(PythonModel):
    def load_context(self, context: PythonModelContext):
        import os
        import torch
        import re
        from PIL import Image
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('context.artifacts["sentencepiece.bpe"]:' , context.artifacts["donut_processor"])
        print('context.artifacts["pytorch_model"]:', context.artifacts["donut_model"])

        self.processor = DonutProcessor.from_pretrained(context.artifacts["donut_processor"])
        self.model = VisionEncoderDecoderModel.from_pretrained(context.artifacts["donut_model"])
        
        self.model.eval().to(self.device)

    def decode_base64_to_image(self,encoding):
        content = encoding.split(";")[1]
        image_encoded = content.split(",")[1]
        return Image.open(BytesIO(base64.b64decode(image_encoded)))


    def predict(self, context: PythonModelContext, data) -> str:
        
        image_data = (data[data.columns[0]]).apply(self.decode_base64_to_image).to_list()[0]
        
        # prepare encoder inputs
        pixel_values = self.processor(image_data, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        # prepare decoder inputs
        task_prompt = TASK_PROMPT
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)

        outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        decoded_output_sequence = self.processor.batch_decode(outputs.sequences)[0]
    
        extracted_cheque_details = decoded_output_sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        ## remove task prompt from token sequence
        cleaned_cheque_details = re.sub(r"<.*?>", "", extracted_cheque_details, count=1).strip()  
        ## generate ordered json sequence from output token sequence
        cheque_details_json = self.processor.token2json(cleaned_cheque_details)
        
        # cheque_details_str = json.dumps(cheque_details_json)

        return cheque_details_json