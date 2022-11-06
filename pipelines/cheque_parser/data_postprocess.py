from zenml.pipelines import pipeline

@pipeline
def data_postprocess(importer, data_splitter,
                     create_train_metadata_json, create_val_metadata_json,
                     create_test_metadata_json, push_data_to_hf_hub
                     ):
    
    labelled_data = importer()
    train_df, val_df, test_df = data_splitter(labelled_data)
    train_md = create_train_metadata_json(input_data=train_df)
    val_md = create_val_metadata_json(input_data=val_df)
    test_md = create_test_metadata_json(input_data=test_df)
    push_data_to_hf_hub(train_md=train_md, test_md=test_md, val_md = val_md)


