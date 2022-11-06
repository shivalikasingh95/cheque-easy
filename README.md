# soch-ai

## Prerequisites:

1. Install all python related dependences to run this project using `requirements.txt` file available in the root directory of the project.

    **Note:** ZenML also supports directly installing dependencies of all of its integrations. So you can also install `zenml` and `zenml[server]` first using pip and then use `zenml integration` to install most of the other dependencies for this project.
    ```shell
    zenml integration install label_studio azure mlflow torch huggingface pytorch-lightning
    ```
    However, using this approach you will still need to install few additional dependencies like `sentencepiece` and `mysqlclient` separately.

2. Install all system level dependencies. 

- This is for being able to connect to the MySQL server which will be used by mlflow as a backend store to keep track of all experiments runs, metadata, etc.

```shell
sudo apt-get update

sudo apt-get install python3.9-dev default-libmysqlclient-dev build-essential`
```
- 

2. You need to setup the following environment variables

    - MYSQL_SERVER_HOST: This refers to the host name of the MySQL server which will be hosting the database that will be used by mlflow 
    - MYSQL_SERVER_USERNAME : This is the username that will be used to access the MySQL server.
    - MYSQL_SERVER_PASSWORD : This is the passowrd that will be used for authentication with the MySQL server. 
    - MLFLOW_TRACKING_URI: This refers to the URL of the Mlflow server. 
    - MLFLOW_TRACKING_USERNAME: This is the username for accessing the mlflow server.
    - MLFLOW_TRACKING_PASSWORD: This is the password for accessing the mlflow server.
    - AZURE_STORAGE_ACCOUNT: This is the name of the Azure storage account which contains the bucket that can be used by ZenML and Mlflow as an articact store.
    - AZURE_STORAGE_SECRET: This refers to the access token value for the azure storage account. 
    - LABEL_STUDIO_API
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
