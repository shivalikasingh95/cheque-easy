from zenml.steps import BaseParameters

class InferenceParams(BaseParameters):
    test_data_repo = "shivi/cheques_sample_data"

class AnnotationParams(BaseParameters):
    cheque_parser_label_project = "cheque_parser"
    cheque_parser_labels = ["payee_name","bank_name","amt_in_words","amt_in_figures","cheque_date"]
    export_format = "JSON"
    images_uri = "az://zenml3-zenmlartifactstore/data/batch1"

class DataParams(BaseParameters):
    annotation_file_path = "../cheques_dataset/cheques_label_file.csv"
    train_data_path = "../hf_cheques_data/train"
    val_data_path = "../hf_cheques_data/val"
    test_data_path = "../hf_cheques_data/test"
    cheques_dataset_path = '../cheques_dataset/cheque_images'
    hf_dataset_repo = "shivi/cheques_sample_data"
    hf_data_dir = "../hf_cheques_data"

    
class DonutTrainParams(BaseParameters):
    pretrained_ckpt = "nielsr/donut-base" #"naver-clova-ix/donut-base"
    dataset = "shivi/cheques_sample_data"
    image_size = [960, 720] # image size for encoder
    max_length = 768 # decoder seq length
    task_start_token="<parse-cheque>"
    task_end_token="<parse-cheque>"
    
    batch_size = 1
    num_workers = 4
    epochs = 1
    max_epochs = 1
    val_check_interval = 0.2 # how many times we want to validate during an epoch
    check_val_every_n_epoch = 1
    gradient_clip_val = 1.0
    num_training_samples_per_epoch = 800
    lr = 3e-5
    train_batch_sizes = [8]
    val_batch_sizes = [1]
    # "seed":2022,
    num_nodes = 1
    warmup_steps = 300 # 800/8*30/10, 10%
    result_path= "./result"
    verbose = True
    accelerator = "gpu"
    device_num = 1
    precision = 16 # use mixed precision 

class ModelSaveDeployParams(BaseParameters):
    workers = 3
    mlflow_model_name = 'ChequeParserDonutModel'
    mlflow_experiment_name = 'donut_training'
    timeout = 60
    model_save_path = "ChequeParserDonutModel"
    processor_save_path = "ChequeParserDonutProcessor"
    hf_trained_model_save_repo = "shivi/cheques_model"
    min_accuracy = 0.6