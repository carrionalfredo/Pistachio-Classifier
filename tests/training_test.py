import training_aws
import pandas as pd

def read_data_test():
    data_file = 'pistachio_20230724'
    #expected dataframe
    expected_df=pd.read_csv(f'data/{data_file}.csv')
    #actual dataframe
    actual_df = training_aws.read_data.fn(data_file)
    assert expected_df.shape==actual_df.shape