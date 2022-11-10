from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.steps import MLFlowDeployerParameters
from zenml.services import load_last_service_from_step
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

@pipeline
def train_donut_pipeline(
    load_config, 
    load_processor,
    load_model, 
    train_donut_model,
    evaluator,
    deployment_trigger,
    model_deployer
):
    config = load_config()
    processor = load_processor()
    model = load_model(vis_enc_dec_config=config)
    artifacts = train_donut_model(processor=processor, model=model)
    accuracy = evaluator(trained_model)
    deployment_decision = deployment_trigger(accuracy)
    model_deployer(deployment_decision, artifacts)
