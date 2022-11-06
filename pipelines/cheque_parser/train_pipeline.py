# from zenml.config import DockerSettings
# from zenml.integrations.constants import WANDB
from zenml.pipelines import pipeline

#docker_settings = DockerSettings(required_integrations=[WANDB])

#pipeline(enable_cache=True, settings={"docker": docker_settings})

@pipeline
def train_donut_pipeline(
    importer, load_config, load_processor,load_model, create_pytorch_dataset,train_donut_model
):
  dataset = importer()
  config = load_config()
  processor = load_processor()
  model = load_model(config=config, processor=processor)
  train_dataloader, val_dataloader = create_pytorch_dataset(processor=processor, model=model)
  train_donut_model(processor=processor, model=model,
                                    train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader)
