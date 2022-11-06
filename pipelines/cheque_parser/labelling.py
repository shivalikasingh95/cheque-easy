from zenml.pipelines import pipeline


@pipeline
def cheque_parser_labelling(
    get_or_create_ls_project,
    get_azure_image_uri,
    storage_create_sync,
):
    
    label_studio_project = get_or_create_ls_project()
    azure_images_uri,preds = get_azure_image_uri()
    storage_create_sync(uri=azure_images_uri, dataset_name=label_studio_project,predictions=preds)


@pipeline
def process_labelled_data(
    get_or_create_ls_project,
    fetch_labeled_data,
    convert_label_studio_annotation,
):
    project_name=get_or_create_ls_project()
    labelled_data=fetch_labeled_data(project_name)
    converted_data=convert_label_studio_annotation(labelled_data)