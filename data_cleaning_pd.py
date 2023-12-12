import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#Turn off Warning: Modyfing views of the dataframe
pd.set_option('mode.chained_assignment', None)


# Load dataset
df_raw = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# Convert <no-response> and <not-shown> to NaN
df_raw[df_raw == '<not-shown>'] = np.nan
df_raw[df_raw == '<no-response>'] = np.nan

# Drop columns with only NaN values
df = df_raw.dropna(axis=1, how='all')

# Covert predictions to 0 (PTSD) and 1 (CUD)
labels = df['SURVEY_NAME']
labels[labels == 'Intrusionsfragebogen (T)'] = 0
labels[labels == 'Intrusionsfragebogen (K)'] = 1

# Drop column with the labels
df.drop(columns=['SURVEY_NAME'], inplace=True)

# Drop all the columns ending with _RT
columns_to_drop = [col for col in df.columns if col.endswith('_RT')]
df.drop(columns=columns_to_drop, inplace=True)

# Remove "_CUD" from column names
df.columns = df.columns.str.replace('_CUD', '')
# Remove "_PTSD" from column names
df.columns = df.columns.str.replace('_PTSD', '')

# Group by column names and merge columns
merged_df = df.groupby(df.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())) if len(x.columns) > 1 else str(y.iloc[0]), axis=1))

# Remove duplicates
def remove_consecutive_duplicates(cell):
    words = cell.split()
    cleaned_words = [word for i, word in enumerate(words) if i == 0 or word != words[i-1]]
    return ' '.join(cleaned_words)

cleaned_df = merged_df.applymap(remove_consecutive_duplicates)

# Replace numbers to numeric objects
cleaned_df = cleaned_df.apply(pd.to_numeric, errors='coerce')

# Replace values larger than 10 to NaN 
cleaned_df = cleaned_df.mask(cleaned_df > 10, np.nan)

# Replace non-standard representations with actual NaN
cleaned_df.replace(['nan', ''], np.nan, inplace=True)

# Saves dataframe for NaN Visualisation
cleaned_df.to_csv('data/nan_data.csv', sep = ';')

nan_perc_limit = 15
# Calculate the percentage of NaN values in each column
nan_percentage = (cleaned_df.isna().mean() * 100)

# Select columns with less than 15% NaN values
selected_columns = nan_percentage[nan_percentage <= nan_perc_limit].index
cleaned_df = cleaned_df[selected_columns]

cleaned_df = pd.concat([cleaned_df, labels], axis = 1)

# Drop rows with 80% or more empty values
threshold = 0.8
cleaned_df_filtered = cleaned_df.dropna(thresh=int(cleaned_df.shape[1] * (1 - threshold)))

new_labels = cleaned_df_filtered['SURVEY_NAME']

# Separate continuous and binary columns to handle NaN values
continuous_df = cleaned_df_filtered.loc[:, ~cleaned_df_filtered.columns.str.endswith(tuple(map(str, range(10)))) & (cleaned_df_filtered.columns != 'SURVEY_NAME')]
continuous_df.fillna(continuous_df.median(), inplace=True)
# Standardize continuous columns
scaler = StandardScaler()
continuous_df[continuous_df.columns] = scaler.fit_transform(continuous_df[continuous_df.columns])

binary_df = cleaned_df_filtered.loc[:, cleaned_df_filtered.columns.str.endswith(tuple(map(str, range(10))))]
binary_df[binary_df.columns] = binary_df[binary_df.columns].apply(lambda x: x.fillna(x.mode().iloc[0]))

filled_df = pd.concat([continuous_df, binary_df, new_labels], axis = 1)

# Drop columns with low std
min_std = 0.1
high_std_columns = filled_df.columns[filled_df.std() > min_std]
final_df = filled_df[high_std_columns]
print(final_df)

final_df.to_csv('data/final_data.csv', sep = ';')