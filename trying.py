import numpy as np
import pandas as pd


data = {
    'FloatColumn': [1.5, 2.7, 3.2, 4.8, 5.1],
    'IntAsFloatColumn': [1.0, 2.0, 3.0, 4.0, 5.0]
}

df = pd.DataFrame(data)

def convert_int_columns_to_int(df):
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.floating) and np.all(df[col] % 1 == 0):
            # If the column is a float and all values are integers, convert to int
            df[col] = df[col].astype(int)
    return df

df = convert_int_columns_to_int(df)
# Display the DataFrame
print(df)
print(df.dtypes)

df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
print(df.dtypes==np.float64)