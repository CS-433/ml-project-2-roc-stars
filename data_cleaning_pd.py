import pandas as pd
import numpy as np

# Load dataset
df_raw = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# Convert <no-response> and <not-shown> to NaN
df_raw[df_raw == '<not-shown>'] = np.nan
df_raw[df_raw == '<no-response>'] = np.nan

# Drop columns with only NaN values
df = df_raw.dropna(axis=1, how='all')

# Drop all the columns ending with _RT
columns_to_drop = [col for col in df.columns if col.endswith('_RT')]
df.drop(columns=columns_to_drop, inplace=True)

# Remove all 'CUD' or 'PTSD' from headers
df.columns = df.columns.str.replace('_CUD|_PTSD', '')

# Group by column names and merge columns
merged_df = df.groupby(df.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())), axis=1))
merged_df.to_csv('data/clean_pd.csv', sep = ',')
print(merged_df)

merged_df.to_csv('data/cleaned_data_pd.csv', sep = ',')