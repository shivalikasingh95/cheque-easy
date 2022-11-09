from zenml.integrations.mlflow.services import MLFlowDeploymentService
from PIL.Image import Image
from PIL import PngImagePlugin
from zenml.steps import step
from typing import Dict
import base64
# import io
from io import BytesIO
import pandas as pd

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
    
    encoded_image = encode_pil_to_base64(data)
    sample = pd.DataFrame({"images": [encoded_image]}).to_json(orient='split')
    service.start(timeout=10)  
    prediction = service.predict(sample)
    return prediction