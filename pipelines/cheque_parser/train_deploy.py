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
    # importer, 
    load_config, 
    load_processor,
    load_model, 
    # create_pytorch_dataset,
    train_donut_model,
    evaluator
    # deployment_trigger,
    # model_deployer
):
    # dataset = importer()
    config = load_config()
    processor = load_processor()
    model = load_model(vis_enc_dec_config=config, donut_processor=processor)
    # train_dataloader, val_dataloader = create_pytorch_dataset(processor=processor, model=model)
    trained_model, donut_processor = train_donut_model(processor=processor, model=model
                                    # train_dataloader=train_dataloader,
                                    # val_dataloader=val_dataloader
    )

    accuracy = evaluator(trained_model, donut_processor)
    # deployment_decision = deployment_trigger(accuracy)
    # model_deployer(deployment_decision, artifacts)
