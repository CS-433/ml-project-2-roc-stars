import pandas as pd
import numpy as np

# Load dataset
df_raw = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# drop columns with only Nan values
df = df_raw.dropna(axis=1, how='all')

# Convert to numpy array and convert <no-response> to <not shown>
raw_data = df.to_numpy()
raw_data[raw_data == '<not-shown>'] = '<no-response>'
print(raw_data.shape)

# Remove columns that contain only <no response>
cols_to_delete = np.where((raw_data == '<no-response>').all(axis=0))[0]
data = np.delete(raw_data, cols_to_delete, axis=1)
print(data.shape)

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
