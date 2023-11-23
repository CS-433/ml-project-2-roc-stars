import pandas as pd
import numpy as np

# Load dataset
df_raw = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# drop columns with only Nan values
df = df_raw.dropna(axis=1, how='all')

# drop all the col ending with _RT
columns_to_drop = [col for col in df.columns if col.endswith('_RT')]
df.drop(columns=columns_to_drop, inplace=True)

# convert <no-response> to <not shown>
df[df == '<not-shown>'] = '<no-response>'

# Remove columns that contain only <no response>
cols_no_response = df.columns[(df == '<no-response>').all(axis=0)]
df.drop(columns=cols_no_response, inplace=True)

print(df)

columns_containing_cud = df.filter(like='CUD', axis=1).columns.to_numpy()
print(columns_containing_cud[2])

col_i = columns_containing_cud[1]
print(col_i)
print(col_i[:len(col_i)-4])

columns_containing_ptsd = df.filter(like='PTSD', axis=1).columns.to_numpy()
print(columns_containing_ptsd[2])

for i in range(len(columns_containing_cud)):
    col_i = columns_containing_cud[i]
    for j in range(len(columns_containing_ptsd)):
        col_j = columns_containing_ptsd[j]
        if (col_i[:len(col_i) - 4] == col_j[:len(col_j) - 4]):
            print(col_j)
            print("Colums ", i, "and ", j, "are the same experiment")

df.to_csv("new_data.csv", sep = ',')

data = df.to_numpy()
# Separate PTSD from CUD patients
PTSD = (data[:, 6] == 'Intrusionsfragebogen (T)')
CUD = (data[:, 6] == 'Intrusionsfragebogen (K)')
data_T = data[PTSD]
data_K = data[CUD]


cols_to_delete_T = np.where((data_T == '<no-response>').all(axis=0))[0]
cols_to_delete_K = np.where((data_K == '<no-response>').all(axis=0))[0]

print(len(cols_to_delete_K))
print(len(cols_to_delete_T))


group = raw_data[:, 6]

# Remove group labels (prediction) from dataset
data_new = np.delete(data, 6, 1)

# Covert predictions to 0 (PTSD) and 1 (CUD)
group[group == 'Intrusionsfragebogen (T)'] = 0
group[group == 'Intrusionsfragebogen (K)'] = 1

print(group.shape)
