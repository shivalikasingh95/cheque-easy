from zenml.steps import step
from typing import Dict
from torch.utils.data import DataLoader
from transformers import DonutProcessor, VisionEncoderDecoderModel
from .utils.pl_module import DonutModelPLModule
from pytorch_lightning.callbacks import Callback
from params import DonutTrainParams, ModelSaveDeployParams
from zenml.client import Client
from pytorch_lightning.loggers import MLFlowLogger
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
from .mlflow_pyfunc import DonutModel
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import mlflow
import os, pathlib
import pytorch_lightning as pl
from .create_pt_dataset import DonutDataset

experiment_tracker = Client().active_stack.experiment_tracker

model_params = ModelSaveDeployParams()
EXPERIMENT_NAME = model_params.mlflow_experiment_name

mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, tracking_uri=get_tracking_uri())


MODEL_REPO = model_params.hf_trained_model_save_repo
TRAIN_END_COMMIT_MSG_PROCESSOR = "Training complete!Pushing Processor to Hub"
TRAIN_END_COMMIT_MSG_MODEL = "Training complete! Pushing model to Hub"
PROCESSOR_SAVE_PATH = model_params.processor_save_path
MODEL_SAVE_PATH = model_params.model_save_path


conda_env = {
    'channels': ['conda-forge'],
    'dependencies': [
        'python=3.9.13',
        'pip<=22.3.1'],
    'pip': [
        'mlflow',
        'cffi==1.15.1',
        'cloudpickle==2.2.0',
        'defusedxml==0.7.1',
        'dill==0.3.5.1',
        'ipython==8.6.0',
        'pillow==9.3.0',
        'sentencepiece==0.1.97',
        'torch==1.13.0',
        'transformers==git+https://github.com/shivalikasingh95/transformers.git@image_utils_fix'
    ],
    'name': 'mlflow-env'
}

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(MODEL_REPO,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(MODEL_REPO,
                                    commit_message=TRAIN_END_COMMIT_MSG_PROCESSOR)
        pl_module.processor.save_pretrained(MODEL_SAVE_PATH)

        pl_module.model.save_pretrained(MODEL_SAVE_PATH)
        pl_module.model.push_to_hub(MODEL_REPO,
                                    commit_message=TRAIN_END_COMMIT_MSG_MODEL)

        artifacts={'donut_processor': PROCESSOR_SAVE_PATH, 'donut_model': MODEL_SAVE_PATH}
        print("artifacts:",artifacts)

        input_schema = Schema([ColSpec("string", "images")])
        output_schema = Schema([ColSpec("string")])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        mlflow.pyfunc.log_model(MODEL_SAVE_PATH, 
                            python_model=DonutModel(), 
                            artifacts=artifacts, 
                            signature=signature,
                            registered_model_name='donut-cheques-model',
                            conda_env=conda_env)

# @step(experiment_tracker="wandb_tracker")
# @step(experiment_tracker=experiment_tracker.name)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name,
    settings={
        "experiment_tracker.mlflow": MLFlowExperimentTrackerSettings(
            experiment_name=EXPERIMENT_NAME,
        )
    }
)
def train_evaluate_donut(params: DonutTrainParams,
                processor: DonutProcessor,
                model: VisionEncoderDecoderModel,
                # train_dataloader: DataLoader,
                # val_dataloader: DataLoader
                ) --> Output(trained_model=VisionEncoderDecoderModel,
                processor_donut=DonutProcessor): #> Dict:

    train_dataset = load_dataset(params.dataset, split='train[0:30]')
    val_dataset = load_dataset(params.dataset, split='validation[0:20]')

    train_dataset = DonutDataset(train_dataset, model=model, processor=processor,
                               max_length=params.max_length, 
                               split="train", task_start_token=params.task_start_token,
                               prompt_end_token=params.task_end_token,
                               sort_json_key=False,
                               )
    val_dataset = DonutDataset(val_dataset, model=model, processor=processor,
                                max_length=params.max_length, 
                                split="validation", task_start_token=params.task_start_token,
                                prompt_end_token=params.task_end_token,
                                sort_json_key=False,
                                )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s_cord-v2>"])[0]

    print("Pad token ID:", processor.decode([model.config.pad_token_id]))
    print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

    model_module = DonutModelPLModule(params, processor, model)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers)
  
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True,
                              num_workers=params.num_workers)

    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.device_num,
        max_epochs=params.max_epochs,
        val_check_interval=params.val_check_interval,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        gradient_clip_val=params.gradient_clip_val,
        precision=params.precision, 
        num_sanity_val_steps=0,
        logger=mlf_logger,
        callbacks=[PushToHubCallback()],
    )

    trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trained_model = model_module.model
    processor_donut = model_module.processor

    return trained_model, processor_donut #{"message": "training_complete"}
