import pandas as pd
import numpy as np

# Load dataset
df_raw = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# convert <no-response> and <not-shown> to NaN
df_raw[df_raw == '<not-shown>'] = np.nan
df_raw[df_raw == '<no-response>'] = np.nan


# drop columns with only Nan values
df = df_raw.dropna(axis=1, how='all')

# drop all the col ending with _RT
columns_to_drop = [col for col in df.columns if col.endswith('_RT')]
df.drop(columns=columns_to_drop, inplace=True)

# Add row that corresponds to the index of the columns of the raw dataset
new_row = pd.Series(range(df.shape[1]), index=df.columns)
df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)

print(df)

# Generate one subset for CUD columns and one for PTSD columns
# Save headers containing "CUD"
data_cud = df.filter(like='CUD', axis=1)
print(data_cud.to_numpy())
columns_containing_cud = data_cud.columns.to_numpy()

# Save headers containing "PTSD"
data_ptsd = df.filter(like='PTSD', axis=1)
columns_containing_ptsd = data_ptsd.columns.to_numpy()

# Extract pairs of corresponding features from both datasets
pairs_same_experiments = []
for i in range(len(columns_containing_cud)):
    col_i = columns_containing_cud[i]
    for j in range(len(columns_containing_ptsd)):
        col_j = columns_containing_ptsd[j]
        if (col_i[:len(col_i) - 4] == col_j[:len(col_j) - 5]):
            pairs_same_experiments.append([i, j])
        if col_i[-1].isdigit() and col_j[-1].isdigit():
            if col_i[:len(col_i)-6] == col_j[:len(col_j)-7] and col_i[-1] == col_j[-1]:
                    pairs_same_experiments.append([i, j])

pairs_same_experiments = np.array(pairs_same_experiments)



data_cud_arr = data_cud.to_numpy()
data_ptsd_arr = data_ptsd.to_numpy()
df_arr = df.to_numpy()

print(df)

new_cols = []
for el in pairs_same_experiments:
    merge = np.concatenate(([df_arr[:, data_cud_arr[0, el[0]]]], [df_arr[:, data_ptsd_arr[0, el[1]]]]), axis = 0)
    new_cols.append(merge)

new_cols = np.array(new_cols)

# Merge the columns corresponding to the same experiment for the 2 groups
n_pairs = new_cols.shape[0]
cleaned_df = pd.DataFrame()
for n in range(n_pairs):
    index = str(n)
    pair = new_cols[n]
    DF = pd.DataFrame(pair)
    DF[pd.isnull(DF)] = 0
    DF = DF.apply(pd.to_numeric, errors = 'coerce')
    merged_DF = DF.sum(skipna=False).to_numpy()
    cleaned_df[index] = merged_DF

# Drop the first row
cleaned_df = cleaned_df.drop(cleaned_df.index[0])

print(cleaned_df)

group = df['SURVEY_NAME']
print(group)

# Convert predictions to 0 (PTSD) and 1 (CUD)
group[group == 'Intrusionsfragebogen (T)'] = 0
group[group == 'Intrusionsfragebogen (K)'] = 1

print(group)

cleaned_df.to_csv('data/cleaned_data.csv', sep = ',')
