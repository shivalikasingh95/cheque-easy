from zenml.steps import step, Output
import pandas as pd

HSBC_BANK = 'HSBC'
CANARA_BANK = 'Canara Bank'
ICICI_BANK = 'ICICI Bank'
AXIS_BANK = 'AXIS BANK'


def create_train_test_val_split(bank_name,labeled_data):

    given_bank_data = labeled_data.loc[labeled_data['bank_name'] == bank_name]
    given_bank_train_data = given_bank_data.iloc[:800,:]
    given_bank_val_data = given_bank_data.iloc[800:,:]
    given_bank_test_data = given_bank_train_data.iloc[700:,:]
    given_bank_train_data = given_bank_train_data.iloc[:700,:]

    return given_bank_train_data, given_bank_val_data, given_bank_test_data



@step(enable_cache=False)
def split_data(labelled_data: pd.DataFrame) -> Output(
    train_data=pd.DataFrame,
    val_data=pd.DataFrame,
    test_data=pd.DataFrame):

    hsbc_train_data, hsbc_val_data , hsbc_test_data = create_train_test_val_split(HSBC_BANK,labelled_data)
    canara_train_data, canara_val_data, canara_test_data = create_train_test_val_split(CANARA_BANK,labelled_data)

    icici_train_data, icici_val_data, icici_test_data = create_train_test_val_split(ICICI_BANK,labelled_data)
    axis_train_data, axis_val_data, axis_test_data = create_train_test_val_split(AXIS_BANK,labelled_data)

    val_data = pd.concat([axis_val_data,canara_val_data,hsbc_val_data,icici_val_data])
    train_data = pd.concat([axis_train_data,canara_train_data,hsbc_train_data,icici_train_data])
    test_data = pd.concat([axis_test_data, canara_test_data, hsbc_test_data, icici_test_data])

    return train_data, val_data, test_data