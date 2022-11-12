from zenml.pipelines import pipeline


@pipeline
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(service=model_deployment_service, data=batch_data)
