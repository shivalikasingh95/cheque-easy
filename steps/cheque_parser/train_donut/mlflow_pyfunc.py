from mlflow.pyfunc import PythonModel, PythonModelContext
from typing import Dict
from PIL import Image
import base64
from io import BytesIO
import json

TASK_PROMPT = "<s_cord-v2>"

class DonutModel(PythonModel):
    def load_context(self, context: PythonModelContext):
        import os
        import torch
        from PIL import Image
        from transformers import DonutProcessor, VisionEncoderDecoderModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('context.artifacts["sentencepiece.bpe"]:' , context.artifacts["donut_processor"])
        print('context.artifacts["pytorch_model"]:', context.artifacts["donut_model"])

        self.processor = DonutProcessor.from_pretrained(context.artifacts["donut_processor"])
        self.model = VisionEncoderDecoderModel.from_pretrained(context.artifacts["donut_model"])
        
        _ = self.model.eval().to(self.device)

    def decode_base64_to_image(encoding):
        content = encoding.split(";")[1]
        image_encoded = content.split(",")[1]
        return Image.open(BytesIO(base64.b64decode(image_encoded)))


    def predict(self, context: PythonModelContext, data) -> Dict:
        # raw_image_bytes = base64.decodebytes(bytearray(json.loads(list(data['images'].values)[0]), encoding="utf8"))
        # print("list(data['images'].values)[0]:",type(list(data['images'].values)[0]))
        # print("list(data['images'].values)[0]:",list(data['images'].values)[0])
        print("incoming data:", data)
        print("data column:",data.columns[0])
        image_data = (data[data.columns[0]]).apply(self.decode_base64_to_image).to_list()[0]
        print("image_data:",image_data)
        # print("raw_image_bytes:",raw_image_bytes)
        # print("raw_image_bytes type:", type(raw_image_bytes))
        # # image_data = Image.open(BytesIO(raw_image_bytes))
        # print("image_data type:", type(image_data))
        # prepare encoder inputs
        pixel_values = self.processor(image_data, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        # prepare decoder inputs
        task_prompt = TASK_PROMPT
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(self.device)
        
        # autoregressively generate sequence
        outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

        # turn into JSON
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        seq = processor.token2json(seq)
        
        seq = json.dumps(seq)
        return seq  
