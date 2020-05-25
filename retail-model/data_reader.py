import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(data):
    df = pd.read_csv(data)
    df["family_code"] = df["family"].astype('category').cat.codes
    encoded_df = df.drop(['family', 'date'], axis=1)
    print(encoded_df.dtypes)
    return train_test_split(encoded_df, train_size=None, test_size=0.4)
