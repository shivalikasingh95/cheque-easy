import os
from tempfile import TemporaryDirectory
from typing import Any, Type
from transformers import DonutProcessor
from zenml.artifacts import ModelArtifact
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.utils import io_utils

DEFAULT_PROCESSOR_DIR = "donut_processor"

class HFDonutMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (DonutProcessor,)
    ASSOCIATED_ARTIFACT_TYPES = (ModelArtifact,)
    
    def handle_input(self, data_type: Type[Any]) -> DonutProcessor:
        super().handle_input(data_type)
        # temp_dir = TemporaryDirectory()
        # io_utils.copy_dir(
        #   os.path.join(self.artifact.uri, DEFAULT_PROCESSOR_DIR),
        #   temp_dir.name
        # )
        return DonutProcessor.from_pretrained(
          # temp_dir.name
          os.path.join(self.artifact.uri, DEFAULT_PROCESSOR_DIR)
        )

    def handle_return(self, processor: DonutProcessor) -> None:
        super().handle_return(processor)
        temp_dir = TemporaryDirectory()
        processor.save_pretrained(temp_dir.name)
        io_utils.copy_dir(
          temp_dir.name,
          os.path.join(self.artifact.uri, DEFAULT_PROCESSOR_DIR)
        )