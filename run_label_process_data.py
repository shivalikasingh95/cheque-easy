from zenml.steps import step, BaseParameters, Output
from zenml.client import Client
from typing import Any, Dict, List
from argparse import ArgumentParser
import pandas as pd

## load pipelines for cheque parser module
from pipelines.cheque_parser.data_postprocess import data_postprocess
from pipelines.cheque_parser.labelling import cheque_parser_labelling, process_labelled_data

## load steps for dataset processing for cheque parser module
from steps.cheque_parser.labelling.convert_annotations import convert_label_studio_annot_to_train_format
from steps.cheque_parser.data_postprocess.import_and_clean import import_clean_data
from steps.cheque_parser.data_postprocess.split_data import split_data
from steps.cheque_parser.data_postprocess.create_metadata import (
    create_train_metadata, 
    create_test_metadata, 
    create_val_metadata
    )
from steps.cheque_parser.data_postprocess.store_data import push_dataset_to_hf_hub

## create dataset imports
from zenml.integrations.label_studio.label_config_generators import (
    generate_basic_object_detection_bounding_boxes_label_config, 
    generate_basic_ocr_label_config
)
from zenml.integrations.label_studio.steps import (
    LabelStudioDatasetRegistrationParameters,
    get_or_create_dataset,
)

## get labelled data imports
from zenml.integrations.label_studio.steps.label_studio_standard_steps import (
    get_labeled_data,
)

## sync create storage related imports
from zenml.integrations.label_studio.steps import (
    LabelStudioDatasetSyncParameters,
    sync_new_data_to_label_studio,
)

from params import DataParams, AnnotationParams
import os

LABEL_CHOICE = "label"
GET_LABELLED_DATA_CHOICE = 'get_labelled_data'
DATA_PROCESS_CHOICE = "data_process"

## Load custom parameters for our pipelines
data_params = DataParams()
annot_params = AnnotationParams()

CHEQUE_PARSER_LABELS = annot_params.cheque_parser_labels 

## fetch active stack's annotator component (label studio)
annotator = Client().active_stack.annotator

## get labelling config for OCR labelling task
bbox_label_config, label_config_type = generate_basic_ocr_label_config(CHEQUE_PARSER_LABELS) 

## get label studio project creation registration parameters
label_studio_registration_params = LabelStudioDatasetRegistrationParameters(
    label_config=bbox_label_config,
    dataset_name=annot_params.cheque_parser_label_project,
)


IMAGE_REGEX_FILTER = ".*(jpe?g|png)"

# AZURE CREATE/SYNC STORAGE SET UP
zenml_azure_artifact_store_sync_params = LabelStudioDatasetSyncParameters(
    storage_type="azure",
    label_config_type=label_config_type,
    regex_filter=IMAGE_REGEX_FILTER,
)
azure_data_sync = sync_new_data_to_label_studio(
    params=zenml_azure_artifact_store_sync_params,
)

@step
def get_azure_images_uri(params: AnnotationParams) -> Output(
    azure_images_uri=str,
    predictions=List[Dict[str, Any]]):
    images_uri = os.environ['LABEL_DATA_STORAGE_BUCKET_NAME']
    return images_uri, []


## define pipelines to run

cheque_label_pipeline = cheque_parser_labelling(
            get_or_create_ls_project=get_or_create_dataset(
                        label_studio_registration_params),
            get_azure_image_uri=get_azure_images_uri(annot_params),
            storage_create_sync=azure_data_sync,
    )


process_labelled_data_pipeline = process_labelled_data(
    get_or_create_ls_project=get_or_create_dataset(
                        label_studio_registration_params),
    fetch_labeled_data=get_labeled_data(),
    convert_label_studio_annotation=convert_label_studio_annot_to_train_format(annot_params)
)


## Modify internal logic of the steps of below pipeline in accordance with the dataset you're using.
## At present, below pipeline is written keeping the following kaggle dataset in mind -
## https://www.kaggle.com/datasets/medali1992/cheque-images
postprocess_pipeline = data_postprocess(
    importer=import_clean_data(data_params),
    data_splitter=split_data(),
    create_train_metadata_json=create_train_metadata(data_params),
    create_val_metadata_json=create_val_metadata(data_params),
    create_test_metadata_json=create_test_metadata(data_params),
    push_data_to_hf_hub=push_dataset_to_hf_hub(data_params)
)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pipeline_type", choices=[LABEL_CHOICE,GET_LABELLED_DATA_CHOICE,DATA_PROCESS_CHOICE])
    args = parser.parse_args()
    if args.pipeline_type == LABEL_CHOICE:
        cheque_label_pipeline.run()
    elif args.pipeline_type == GET_LABELLED_DATA_CHOICE:
        process_labelled_data_pipeline.run()
    elif args.pipeline_type == DATA_PROCESS_CHOICE:
        postprocess_pipeline.run()
