# zenml stack register $ANNOT_STACK_NAME -o default -a default
zenml stack set $ANNOT_STACK_NAME
zenml secrets-manager register azure_secret_mgr2 --key_vault_name=$AZURE_KEY_VAULT -f azure

zenml stack update $ANNOT_STACK_NAME -x azure_secret_mgr2

zenml secrets-manager secret register azurestorage2 --schema=azure --account_name=$STORAGE_ACCOUNT --account_key=$STORAGE_ACCOUNT_KEY

zenml artifact-store register azure_artifact_store2 -f azure --path=$BUCKET_NAME --authentication_secret=azurestorage2

zenml stack update $ANNOT_STACK_NAME -a azure_artifact_store2

zenml secrets-manager secret register LABELSTUDIOAPIKEY --api_key=$LABEL_STUDIO_API_KEY

zenml annotator register label-studio-annotator2 --flavor label_studio --authentication_secret=LABELSTUDIOAPIKEY

zenml stack update $ANNOT_STACK_NAME -an label-studio-annotator2

zenml stack up

zenml stack describe