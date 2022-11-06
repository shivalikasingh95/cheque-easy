from zenml.steps import step
import pandas as pd
from params import DataParams

@step(enable_cache=False)
def import_clean_data(params: DataParams) -> pd.DataFrame:
    
    labelled_data = pd.read_csv(params.annotation_file_path)

    labelled_data = labelled_data.loc[labelled_data['BANK_NAME'].isin(['axis','canara','ICICI','HSBC'])]
    labelled_data.rename(columns={'VALUE_LETTERS':'amt_in_words', 'VALUE_NUMBERS': 'amt_in_figures', \
                           'USER2NAME': 'payee_name','BANK_NAME':'bank_name',
                           'CHEQUE_NO':'cheque_no'},inplace=True)
    labelled_data = labelled_data.drop(['USER1','USER2','SIGNATURE_FILE','valid'],axis=1)

    labelled_data.loc[labelled_data['bank_name'] == 'axis', 'bank_name'] = 'AXIS BANK'
    labelled_data.loc[labelled_data['bank_name'] == 'ICICI', 'bank_name'] = 'ICICI Bank'
    labelled_data.loc[labelled_data['bank_name'] == 'canara', 'bank_name'] = 'Canara Bank'

    labelled_data['cheque_date'] = '06/05/22'

    return labelled_data