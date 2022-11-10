zenml stack copy default $TRAIN_STACK_NAME
zenml stack set $TRAIN_STACK_NAME
zenml secrets-manager register local_mgr --flavor=local
zenml stack update $TRAIN_STACK_NAME -x local_mgr
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow --tracking_uri=$MLFLOW_TRACKING_URI --tracking_username=$MLFLOW_USERNAME --tracking_password=$MLFLOW_PASSWORD
zenml stack update $TRAIN_STACK_NAME -e mlflow_experiment_tracker
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack update $TRAIN_STACK_NAME -d mlflow_deployer
zenml stack describe

