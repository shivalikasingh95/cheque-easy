from zenml.steps import BaseParameters


class AnnotationParams(BaseParameters):
    
    ## name of annotation project to create in label studio
    cheque_parser_label_project = "cheque_parser"
    
    ## name of labels that need to be annotated
    cheque_parser_labels = ["payee_name","bank_name","amt_in_words","amt_in_figures","cheque_date"]
    
    ## the format in which annotations must be exported
    export_format = "JSON"

class DataParams(BaseParameters):

    ## path to file containing  
    annotation_file_path = "../cheques_dataset/cheques_label_file.csv"

    ## Path to dataset containing images that need to be used for training
    cheques_dataset_path = '../cheques_dataset/cheque_images'
    
    ## Path to folder where all training set images will be saved
    train_data_path = "../hf_cheques_data/train"
    
    ## Path to folder where all validation set images will be saved
    val_data_path = "../hf_cheques_data/val"

    ## Path to folder where all test set images will be saved
    test_data_path = "../hf_cheques_data/test"

    ## Name of repository on Hugging Face where we need to push the final dataset created for model training
    hf_dataset_repo = "shivi/cheques_sample_data"
    
    ## Path to root directory for saving the Hugging Face Dataset corresponding to input raw data
    hf_data_dir = "../hf_cheques_data"

    
class DonutTrainParams(BaseParameters):

    ## checkpoint of pretrained donut model which will be used for fine-tuning 
    pretrained_ckpt = "nielsr/donut-base" #"naver-clova-ix/donut-base"
    
    ## Name of Hugging Face Dataset that will be used for training the model
    dataset = "shivi/cheques_sample_data"
    
    ## input size for images that will be fed to the visual encoder of Donut 
    image_size = [960, 720] # image size for encoder
    
    ## sequence length for text decoder of Donut
    max_length = 768
    
    ## task start prompt for information extraction task 
    task_start_token="<parse-cheque>"
    
    ## task end token for information extraction task
    task_end_token="<parse-cheque>"
    
    ## batch size to be used during training
    batch_size = 1

    ## num workers to be used with torch DataLoader to enable multi-process data loading with the 
    ## specified number of loader worker processes
    num_workers = 4

    ## no. of epochs after which training should stop
    max_epochs = 30

    ## how often within one training epoch to check the validation set.
    val_check_interval = 0.2 
    
    ## run validation loop after every N training epochs
    check_val_every_n_epoch = 1
    
    ## the value at which to clip gradients. Passing `None` disables gradient clipping.
    gradient_clip_val = 1.0
    
    ## learning rate to be used for optimizer 
    lr = 3e-5
    
    ## if true, prints validation results
    verbose = True

    ## type of device to be used for training e.g. `cpu`, `gpu`
    accelerator = "gpu"
    
    ## no. of devices to be used for training
    device_num = 1
    
    ## use mixed precision 
    precision = 16 

class ModelSaveDeployParams(BaseParameters):
    
    ## number of workers to use for the prediction service
    workers = 3

    ## name using which trained model should be saved in Mlflow artifact store
    mlflow_model_name = 'ChequeParserDonutModel'
    
    ## name of training experiment 
    mlflow_experiment_name = 'donut_training'
    
    ## time duration to wait for MLFlow model prediction service to start/stop
    timeout = 60
    
    ## Path in which trained model should be saved using `save_pretrained()`
    model_save_path = "ChequeParserDonutModel"

    ## Path in which donut processor used during training should be saved using `save_pretrained()`
    processor_save_path = "ChequeParserDonutProcessor"
    
    ## The repository name on Hugging Face Hub where the trained model should be pushed
    hf_trained_model_save_repo = "shivi/cheques_model"
    
    ## min accuracy threshold that should be met to trigger model deployment
    min_accuracy = 0.8
    
    ## name with which trained model should be registered in MLflow model registry
    mlflow_registered_model_name = 'donut-cheques-model'


class InferenceParams(BaseParameters):

    ## Name of repository from which to load data for inference
    test_data_repo = "shivi/cheques_sample_data"