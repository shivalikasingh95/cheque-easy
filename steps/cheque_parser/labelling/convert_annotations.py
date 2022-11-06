from typing import Any, Dict, List
from zenml.steps import Output, step
from params import AnnotationParams
import pandas as pd
import os



@step(enable_cache=False)
def convert_label_studio_annot_to_train_format(
    params: AnnotationParams,
    label_studio_annotations: List[Dict[Any, Any]]
    ) -> pd.DataFrame:

    text_annots=[]
    label_annots=[]
    labels = params.cheque_parser_labels
    converted_annot_df = pd.DataFrame(columns=['cheque_no']+labels)
    
    for img_data in label_studio_annotations:

        for annotation in img_data['annotations']:
            if 'result' in annotation.keys():
                annot_values = annotation['result']
                for annot in annot_values:
                    text_annot={}
                    label_annot={}
                    if 'labels' in annot['value'].keys():
                        label_annot['label_value'] = annot['value']['labels']
                        label_annot['id'] = annot['id']
                        label_annots.append(label_annot)
                    
                    if 'text' in annot['value'].keys():
                        text_annot['text_value'] = annot['value']['text']
                        text_annot['id'] = annot['id']
                        text_annots.append(text_annot)

        final_annotations = []

        for entry in label_annots:
            final_annotation = {}
            final_annotation['id'] = entry['id']
            for text_ in text_annots:
                if entry['id'] == text_['id']:
                    final_annotation['label_val'] = entry['label_value'][0]
                    final_annotation['text_val'] = text_['text_value'][0]
            final_annotations.append(final_annotation)     
        
        final_annot = {}
        for annot in final_annotations:
            final_annot['cheque_no'] = os.path.basename(img_data['data']['ocr'])
            final_annot[annot['label_val']] = annot['text_val']
            
        converted_annot_df = converted_annot_df.append(final_annot,ignore_index=True)

    return converted_annot_df



    