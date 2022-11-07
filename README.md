# ChequeEasy: Banking with Transformers

ChequeEasy is a project that aims to simply the process of approval of cheques. Leveraging recent advances in Visual Document Understanding (VDU) domain to extract relevant data from cheques and make the whole process quicker and easier for both bank officials and customers. 

This project leverages Donut model proposed in the paper [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) for the parsing of the required data from cheques.Donut is based on a very simple transformer encoder and decoder architecture. It's main USP is that it is an OCR-free approach to information extraction from documents. OCR based techniques come with several limitations such as use of additional downstream models, lack of understanding about document structure, use of hand crafted rules,etc. Donut helps you get rid of all of these OCR specific limitations. 

The model for the project has been trained using this [dataset](https://huggingface.co/datasets/shivi/cheques_sample_data) which is available for use on Hugging Face Hub. This Hugging Face dataset is actually a filtered version of this [Kaggle dataset](https://www.kaggle.com/datasets/medali1992/cheque-images).


## Prerequisites:

1. Create your python virtual environment and install `zenml` and `zenml[server]`. Note that ZenML is compatible with Python 3.7, 3.8, and 3.9. This project uses some custom changes that are not available as part of official zenml release yet so please install zenml as shown below.

```shell
    pip install -q git+https://github.com/shivalikasingh95/zenml.git@label_studio_ocr_config
    pip install "zenml[server]"
```

2. All the dependencies of this project are mentioned in the `requirements.txt` file. 
However, I would recommend installing all integrations of zenml using the `zenml integration install` command to ensure full compatibility with zenml.

```shell
    zenml integration install label_studio azure mlflow torch huggingface pytorch-lightning pillow
```
However, this project has a few additional dependencies such as `mysqlclient`, `nltk` and `donut-python` would have to be installed separately as these are not covered by the zenml integration command.

Also, transformers must be installed from this git branch as it also contains some minor fixes not available as part of official transformers library yet:

```shell
    pip install -q git+https://github.com/shivalikasingh95/transformers.git@image_utils_fix
```

The following dependencies must be installed if you want to run the gradio demo app for the project: `word2number`, `gradio`, `textblob`

2. Install all system level dependencies. 

This is for being able to connect to the MySQL server which will be used by mlflow as a backend store to keep track of all experiments runs, metadata, etc.

```shell
sudo apt-get update

sudo apt-get install python3.9-dev default-libmysqlclient-dev build-essential`
```
If you don't want to run your mlflow server with a MySQL backend, you can skip this step.

2. Cloud resources:
At the moment, zenml supports only cloud based artifact stores for use with label-studio as `annotator` component so you if you wish to use the annotation component of this project then you need to have an AWS/GCP/Azure account which will be used for storing the artifacts generated as part of pipelines run using the annotator stack. 
The below setup has been described for use with Azure but similar set up can be done for AWS/GCP. To see how to setup label studio with zenml using AWS/GCP refer this [link](https://github.com/zenml-io/zenml/tree/main/examples/label_studio_annotation).

    For using label studio with Azure, make sure you have an azure storage account and an azure key vault. You can leverage, Zenml's [MLOps stack recipes](https://github.com/zenml-io/mlops-stacks) to do this for you in case you don't have one. For Azure, you can take a look at the [azure-minimal](https://github.com/zenml-io/mlops-stacks/tree/main/azure-minimal) stack. Although this creates a few additional resources apart from Azure Blob Storage & Key vault so you might want to disable that.

3. If you want to set up the stack for labelling using label studio, you need to setup the following environment variables:
    - **ANNOT_STACK_NAME:** Name to assign to the zenml stack that will be used for labelling.
    - **AZURE_KEY_VAULT:** Name of the key vault that will be used as secrets-manager for your stack.
    - **STORAGE_ACCOUNT:** This is the name of the Azure storage account which contains the bucket that can be used by ZenML as an artifact store.
    - **BUCKET_NAME:** The path of the Azure Blob storage that will be used as an artifact store for this stack. It would something like - `az://<storage_bucket_or_container_name>`
    - **STORAGE_ACCOUNT_KEY:** This refers to the access token value for the azure storage account. 
    - **LABEL_STUDIO_API_KEY:** This refers to the `Access Token` of your label studio instance. You'll to have first start your label studio instance using the command - `label studio start -p 8094` and go to Account page to retrieve your Access Token value to set this environment variable.

4. If you want to set up the stack for training and inference, you need to setup the following environment variables:
    - **TRAIN_STACK_NAME:** Name to assign to the zenml stack that will be used for labelling.
    - **MLFLOW_TRACKING_URI:** Name of the key vault that will be used as secrets-manager for your stack.
    - **MLFLOW_USERNAME:** This is the name of the Azure storage account which contains the bucket that can be used by ZenML as an artifact store.
    - **MLFLOW_PASSWORD:** This refers to the access token value for the azure storage account. 
