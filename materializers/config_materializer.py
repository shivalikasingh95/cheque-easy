from typing import Type, Any
from tempfile import TemporaryDirectory
from transformers import VisionEncoderDecoderConfig
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.io import fileio
from zenml.artifacts import ModelArtifact
from zenml.utils import io_utils
import os

DEFAULT_CONFIG_DIR = "hf_config"

class DonutConfigMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (VisionEncoderDecoderConfig,)
    ASSOCIATED_ARTIFACT_TYPES = (ModelArtifact,)
    
    def handle_input(self, data_type: Type[VisionEncoderDecoderConfig]) -> VisionEncoderDecoderConfig:
        """Read from artifact store"""
        super().handle_input(data_type)
        # temp_dir = TemporaryDirectory()
        # io_utils.copy_dir(
        #   os.path.join(self.artifact.uri, DEFAULT_CONFIG_DIR),
        #   temp_dir.name
        # )
        return VisionEncoderDecoderConfig.from_pretrained(
          os.path.join(self.artifact.uri, DEFAULT_CONFIG_DIR)
        )

    def handle_return(self, model_config: VisionEncoderDecoderConfig) -> None:
        """Write to artifact store"""
        super().handle_return(model_config)
        temp_dir = TemporaryDirectory()
        model_config.save_pretrained(temp_dir.name)
        
        io_utils.copy_dir(
          temp_dir.name,
          os.path.join(self.artifact.uri, DEFAULT_CONFIG_DIR), 
        )