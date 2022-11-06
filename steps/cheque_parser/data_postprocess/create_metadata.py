import os
import json
import shutil
import pandas as pd
from pathlib import Path
from params import DataParams
from zenml.steps import step

def create_metadata_json(dest_dir, data_dir, input_data):
    metadata = []
    for idx,row in input_data.iterrows():
        image_data = f"{row.cheque_no}.jpg"
        img_src_path = data_dir.joinpath(image_data)
        img_dest_path = dest_dir.joinpath(image_data)
        if img_src_path.is_file():
            shutil.copyfile(img_src_path, img_dest_path)
        gt_json = {"gt_parse": {"cheque_details": 
                        [{"amt_in_words": row.amt_in_words}, 
                        {"amt_in_figures": row.amt_in_figures}, 
                        {"payee_name": row.payee_name},
                        {"bank_name": row.bank_name},
                        {"cheque_date": row.cheque_date}]}}
        metadata.append({'file_name': image_data , "ground_truth": json.dumps(gt_json)})
    return metadata

def generate_json(dest_dir, json_metadata):
    with open(dest_dir.joinpath('metadata.jsonl'), 'w') as jsonfile:
        for entry in json_metadata:
            json.dump(entry, jsonfile)
            jsonfile.write('\n')
            
def create_metadata(dest_dir, data_dir, input_data):
    os.makedirs(dest_dir,exist_ok=True)
    dest_dir = Path(dest_dir)
    metadata = create_metadata_json(dest_dir, data_dir, input_data)
    generate_json(dest_dir, metadata)

@step(enable_cache=False)
def create_train_metadata(params: DataParams,input_data: pd.DataFrame) -> bool:

    dest_dir = Path(params.train_data_path)
    data_dir = Path(params.cheques_dataset_path)
    create_metadata(dest_dir,data_dir,input_data)

    return True

@step(enable_cache=False)
def create_val_metadata(params: DataParams,input_data: pd.DataFrame) -> bool:

    dest_dir = Path(params.val_data_path)
    data_dir = Path(params.cheques_dataset_path)
    create_metadata(dest_dir,data_dir,input_data)
    
    return True

@step(enable_cache=False)
def create_test_metadata(params: DataParams,input_data: pd.DataFrame) -> bool:

    dest_dir = Path(params.test_data_path)
    data_dir = Path(params.cheques_dataset_path)
    create_metadata(dest_dir,data_dir,input_data)
    
    return True