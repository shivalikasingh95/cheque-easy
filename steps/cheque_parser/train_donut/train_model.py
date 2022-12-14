## zenml imports
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import MLFlowExperimentTrackerSettings
from zenml.steps import step, Output
from zenml.client import Client

## hugging face imports
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

## torch / pytorch-lightning imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

## mlflow imports
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import mlflow

## other imports
from params import DonutTrainParams, ModelSaveDeployParams
from .mlflow_pyfunc import DonutModel
from utils.create_pt_dataset import DonutDataset
from utils.donut_pl_module import DonutModelPLModule

from typing import Dict

experiment_tracker = Client().active_stack.experiment_tracker

model_params = ModelSaveDeployParams()
EXPERIMENT_NAME = model_params.mlflow_experiment_name

mlf_logger = MLFlowLogger(experiment_name=EXPERIMENT_NAME, tracking_uri=get_tracking_uri())


MODEL_REPO = model_params.hf_trained_model_save_repo
TRAIN_END_COMMIT_MSG_PROCESSOR = "Training complete!Pushing Processor to Hub"
TRAIN_END_COMMIT_MSG_MODEL = "Training complete! Pushing model to Hub"
PROCESSOR_SAVE_PATH = model_params.processor_save_path
MODEL_SAVE_PATH = model_params.model_save_path
MLFLOW_REGISTERED_MODEL_NAME = model_params.mlflow_registered_model_name


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

class LoggingArtifactsCallback(Callback):
    def on_train_epoch_end(self, trainer, donut_pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        donut_pl_module.model.push_to_hub(MODEL_REPO,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, donut_pl_module):
        print(f"Pushing model to the hub after training")

        ## save processor and push to Hugging Face Hub
        donut_pl_module.processor.push_to_hub(MODEL_REPO,
                                    commit_message=TRAIN_END_COMMIT_MSG_PROCESSOR)
        donut_pl_module.processor.save_pretrained(PROCESSOR_SAVE_PATH)

        # save trained model and push to Hugging Face Hub
        donut_pl_module.model.save_pretrained(MODEL_SAVE_PATH)
        donut_pl_module.model.push_to_hub(MODEL_REPO,
                                    commit_message=TRAIN_END_COMMIT_MSG_MODEL)

        # specify path of artifacts to saved in MLflow Artifact store
        artifacts={'donut_processor': PROCESSOR_SAVE_PATH, 'donut_model': MODEL_SAVE_PATH}

        # specify expected types for input & output of the model i.e. Model Signature
        input_schema = Schema([ColSpec("string", "images")])
        output_schema = Schema([ColSpec("string")])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Log model to MLflow Model Registry
        mlflow.pyfunc.log_model(MODEL_SAVE_PATH, 
                            python_model=DonutModel(), 
                            artifacts=artifacts, 
                            signature=signature,
                            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
                            conda_env=conda_env)


@step(enable_cache=False,experiment_tracker=experiment_tracker.name,
    settings={
        "experiment_tracker.mlflow": MLFlowExperimentTrackerSettings(
            experiment_name=EXPERIMENT_NAME,
        )
    }
)
def train_evaluate_donut(params: DonutTrainParams,
                processor: DonutProcessor,
                model: VisionEncoderDecoderModel
                ) -> Dict:

    ## load dataset from HF Hub containing cheque images and corresponding ground_truth (GT)
    train_dataset = load_dataset(params.dataset, split='train').shuffle()
    val_dataset = load_dataset(params.dataset, split='validation').shuffle()
    
    ## create training and validation dataset converting images to tensor form and GT to input_ids
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

    ## set `pad_token_id` and `decoder_start_token_id` so that the `decoder_input_ids` can be created automatically
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([params.task_start_token])[0]

    # `pad_token_id` should be <PAD>
    print("Pad token ID:", processor.decode([model.config.pad_token_id]))

    # `decoder_start_token_id` value should be equal to prompt for starting IE task i.e. value of `task_start_token` defined 
    # under DonutTrainParams() in params.py file 
    print("Decoder start token ID:", processor.decode([model.config.decoder_start_token_id]))

    ## Initialize Pytorch-lightning training module
    model_module = DonutModelPLModule(params, processor, model)

    ## create train and val dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers)
  
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True,
                              num_workers=params.num_workers)

    ## create Pytorch-Lightning Trainer 
    trainer = pl.Trainer(
        accelerator=params.accelerator,
        devices=params.device_num,
        max_epochs=params.max_epochs,
        val_check_interval=params.val_check_interval,
        check_val_every_n_epoch=params.check_val_every_n_epoch,
        gradient_clip_val=params.gradient_clip_val,
        precision=params.precision, 
        num_sanity_val_steps=0,
        # logger=mlf_logger,
        callbacks=[LoggingArtifactsCallback()],
    )

    # start training model
    trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return {"message": "training_complete"}
