from utils.donut_utils import (
    prepare_data_using_processor
)
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings)
from zenml.client import Client
from zenml.steps import step
from transformers import VisionEncoderDecoderModel, DonutProcessor
from donut import JSONParseEvaluator   
from params import DonutTrainParams, ModelSaveDeployParams
from typing import Dict
from datasets import load_dataset
import numpy as np
import mlflow
import torch
import re
import json

experiment_tracker = Client().active_stack.experiment_tracker
model_params = ModelSaveDeployParams()
EXPERIMENT_NAME = model_params.mlflow_experiment_name

@step(enable_cache=False, experiment_tracker=experiment_tracker.name,
    settings={
        "experiment_tracker.mlflow": MLFlowExperimentTrackerSettings(
            experiment_name=EXPERIMENT_NAME,
        )
    }
)
def evaluate(params: DonutTrainParams,
        trained_model_artifacts: Dict) -> float: 
   output_list = []
   accs = []

   dataset = load_dataset(params.dataset, split="test")

   model = VisionEncoderDecoderModel.from_pretrained(model_params.hf_trained_model_save_repo)
   donut_processor = DonutProcessor.from_pretrained(model_params.hf_trained_model_save_repo)

   device = "cuda" if torch.cuda.is_available() else "cpu"

   model.eval()
   model.to(device)


   for idx, sample in enumerate(dataset):

    sample = dataset[idx]
    image = sample["image"].convert("RGB")

    cheque_image_tensor, input_for_decoder = prepare_data_using_processor(donut_processor,image, params.task_start_token)
    
    outputs = model.generate(cheque_image_tensor,
                                decoder_input_ids=input_for_decoder,
                                max_length=model.decoder.config.max_position_embeddings,
                                early_stopping=True,
                                pad_token_id=donut_processor.tokenizer.pad_token_id,
                                eos_token_id=donut_processor.tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1,
                                bad_words_ids=[[donut_processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True,
                                output_scores=True,)

    decoded_output_sequence = donut_processor.batch_decode(outputs.sequences)[0]
    
    extracted_cheque_details = decoded_output_sequence.replace(donut_processor.tokenizer.eos_token, "").replace(donut_processor.tokenizer.pad_token, "")
    ## remove task prompt from token sequence
    cleaned_cheque_details = re.sub(r"<.*?>", "", extracted_cheque_details, count=1).strip()  
    ## generate ordered json sequence from output token sequence
    cheque_details_json = donut_processor.token2json(cleaned_cheque_details)

    ground_truth = json.loads(sample["ground_truth"])
    ground_truth = ground_truth["gt_parse"]
    print("ground_truth:",ground_truth)
    evaluator = JSONParseEvaluator()
    score = evaluator.cal_acc(cheque_details_json, ground_truth)

    accs.append(score)
    output_list.append(cheque_details_json)
    print("accs:",accs)
    
    mean_acc = np.mean(accs)
    print("mean_acc:",mean_acc)
    mlflow.log_metric("mean_val_acc", mean_acc)
    return mean_acc

