# ChequeEasy: Banking with Transformers

ChequeEasy is a project that aims to simply the process of approval of cheques. Leveraging recent advances in Visual Document Understanding (VDU) domain to extract relevant data from cheques and make the whole process quicker and easier for both bank officials and customers. 

This project leverages Donut model proposed in this paper for the parsing of the required data from cheques.Donut is based on a very simple transformer encoder and decoder architecture. It's main USP is that it is an OCR-free approach to information extraction from documents. OCR based techniques come with several limitations such as use of additional downstream models, lack of understanding about document structure, use of hand crafted rules,etc. Donut helps you get rid of all of these OCR specific limitations. 

The model for the project has been trained using this dataset . This HF dataset is actually a filtered version of this kaggle dataset .


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

- This is for being able to connect to the MySQL server which will be used by mlflow as a backend store to keep track of all experiments runs, metadata, etc.

```shell
sudo apt-get update

sudo apt-get install python3.9-dev default-libmysqlclient-dev build-essential`
```
- If you don't want to run your mlflow server with a MySQL backend, you can skip this step.

2. If you want to set up the stack for labelling using label studio, you need to setup the following environment variables:
    - ANNOT_STACK_NAME: Name to assign to the zenml stack that will be used for labelling.
    - AZURE_KEY_VAULT: Name of the key vault that will be used as secrets-manager for your stack.
    - STORAGE_ACCOUNT: This is the name of the Azure storage account which contains the bucket that can be used by ZenML as an artifact store.
    - BUCKET_NAME: The path of the Azure Blob storage that will be used as an artifact store for this stack. It would something like - `az://<storage_bucket_or_container_name>`
    - STORAGE_ACCOUNT_KEY: This refers to the access token value for the azure storage account. 
    - LABEL_STUDIO_API_KEY: This refers to the `Access Token` of your label studio instance. You'll to have first start your label studio instance using the command - `label studio start -p 8094` and go to Account page to retrieve your Access Token value to set this environment variable.

3. If you want to set up the stack for training and inference, you need to setup the following environment variables:
    - TRAIN_STACK_NAME: Name to assign to the zenml stack that will be used for labelling.
    - AZURE_KEY_VAULT: Name of the key vault that will be used as secrets-manager for your stack.
    - STORAGE_ACCOUNT: This is the name of the Azure storage account which contains the bucket that can be used by ZenML as an artifact store.
    - STORAGE_ACCOUNT_KEY: This refers to the access token value for the azure storage account. 
    - LABEL_STUDIO_API_KEY: This refers to the `Access Token` of your label studio instance. You'll to have first start your label studio instance using the command - `label studio start -p 8094` and go to Account page to retrieve your Access Token value to set this environment variable.



2. Make sure your MySQL server is accessible from whichever IP you're using to connect to it.
3. Start


You must run the following steps to set up your ZenML stack:

0. Clone the repository & run a `zenml init` to initialize this code repo as a zenml repository

1. Create a copy of the default stack and use it to setup your new stack and give it a new name such as `cheque_processing_stack`

```shell 
zenml stack copy default annot_stack2
```
2. Set this new stack as the current stack for your zenml repository

`zenml stack set annot_stack2` 

`zenml secrets-manager register azure_secret_mgr --key_vault_name=zenmlkeyvault -f azure`

`zenml stack update annot_stack2 -x azure_secret_mgr`

`zenml secrets-manager secret register azurestorage --schema=azure --account_name=$AZURE_STORAGE_ACCOUNT --account_key=$AZURE_STORAGE_SECRET`

`zenml artifact-store register azure_artifact_store2 -f azure --path="az://zenml3-zenmlartifactstore" --authentication_secret=azurestorage`

`zenml stack update annot_stack2 -a azure_artifact_store2`

`zenml secrets-manager secret register LABELSTUDIOAPIKEY --api_key=7a559ac37e4da3bb82daca330a8dc588ff7efe8c`

`zenml annotator register label-studio-annotator2 --flavor label_studio --authentication_secret=LABEL_STUDIOAPIKEY`

`zenml stack update annot_stack2 -an label-studio-annotator2`


1. Go to your account, Scroll to API section and Click **Expire API Token** to remove previous tokens

2. Click on **Create New API Token** - It will download kaggle.json file on your machine.

3. Go to your Google Colab project file and run the following commands:

**1)  ! pip install -q kaggle**

**2)   from google.colab import files**

**files.upload()**

- Choose the kaggle.json file that you downloaded

**3) ! mkdir ~/.kaggle**

**! cp kaggle.json ~/.kaggle/**

- Make directory named kaggle and copy kaggle.json file there.

**4) ! chmod 600 ~/.kaggle/kaggle.json**

- Change the permissions of the file.

**5)  ! kaggle datasets list**
    - That's all ! You can check if everything's okay by running this command.