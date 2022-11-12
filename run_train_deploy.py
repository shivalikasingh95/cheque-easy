from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters
from zenml.services import load_last_service_from_step
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

## import pipelines for cheque parser module
from pipelines.cheque_parser.data_postprocess import data_postprocess
from pipelines.cheque_parser.train_deploy import train_donut_pipeline
from pipelines.cheque_parser.inference_pipeline import inference_pipeline

## import steps for inference pipeline
from steps.cheque_parser.inference.load_data import import_inference_data
from steps.cheque_parser.inference.load_prediction_service import (
	prediction_service_loader,
	MLFlowDeploymentLoaderStepParameters
)
from steps.cheque_parser.inference.predict import predictor

## import steps for training pipeline
from steps.cheque_parser.train_donut.load_config import load_model_config 
from steps.cheque_parser.train_donut.load_processor import load_donut_processor
from steps.cheque_parser.train_donut.load_model import load_vision_encoder_decoder_model
from steps.cheque_parser.train_donut.train_model import train_evaluate_donut
from steps.cheque_parser.train_donut.evaluate import evaluate
from steps.cheque_parser.train_donut.deployment_trigger import deployment_trigger, DeploymentTriggerConfig

## import steps for dataset processing pipeline
from steps.cheque_parser.data_postprocess.import_and_clean import import_clean_data
from steps.cheque_parser.data_postprocess.split_data import split_data
from steps.cheque_parser.data_postprocess.create_metadata import (
    create_train_metadata, 
    create_test_metadata, 
    create_val_metadata
    )
from steps.cheque_parser.data_postprocess.store_data import push_dataset_to_hf_hub

from params import DonutTrainParams, DonutTrainParams, DataParams, InferenceParams, ModelSaveDeployParams
from transformers import VisionEncoderDecoderModel
from argparse import ArgumentParser

TRAIN_CHOICE = "train"
DATA_PREP_CHOICE = "data_process"
INFERENCE_CHOICE = "inference"

## Load custom parameters for our pipelines
data_params = DataParams()
donut_train_params = DonutTrainParams()
inference_params = InferenceParams()
model_params = ModelSaveDeployParams()

postprocess_pipeline = data_postprocess(
    importer=import_clean_data(data_params),
    data_splitter=split_data(),
    create_train_metadata_json=create_train_metadata(data_params),
    create_val_metadata_json=create_val_metadata(data_params),
    create_test_metadata_json=create_test_metadata(data_params),
    push_data_to_hf_hub=push_dataset_to_hf_hub(data_params)
)


train_pipeline = train_donut_pipeline(
    
    load_config=load_model_config(donut_train_params),
    
    load_processor=load_donut_processor(donut_train_params),
    
    load_model=load_vision_encoder_decoder_model(donut_train_params),
    
    train_donut_model=train_evaluate_donut(donut_train_params),

    evaluator=evaluate(donut_train_params),

    deployment_trigger=deployment_trigger(
            config=DeploymentTriggerConfig(
                min_accuracy=model_params.min_accuracy,
            )
        ),
    model_deployer=mlflow_model_deployer_step(
        params=MLFlowDeployerParameters(workers=model_params.workers,
                experiment_name=model_params.mlflow_experiment_name,
                model_name=model_params.mlflow_model_name,
                timeout=model_params.timeout))

)

inference_pipeline = inference_pipeline(
        dynamic_importer=import_inference_data(inference_params),
        prediction_service_loader=prediction_service_loader(
            MLFlowDeploymentLoaderStepParameters(
                pipeline_name="train_donut_pipeline",
                step_name="mlflow_model_deployer_step",
                experiment_name=model_params.mlflow_experiment_name
            )
        ),
        predictor=predictor(),
    )



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pipeline_type", choices=[DATA_PREP_CHOICE,TRAIN_CHOICE, INFERENCE_CHOICE])
    args = parser.parse_args()
    if args.pipeline_type == DATA_PREP_CHOICE:
        postprocess_pipeline.run()
    elif args.pipeline_type == TRAIN_CHOICE:
        train_pipeline.run()
    elif args.pipeline_type == INFERENCE_CHOICE:
        inference_pipeline.run()
