import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'PARTICIPANT_ID': ['s061255111', 's061255111', 's061255111', 's061255111', 's061255111', 's061255111'],
    'STIMMUNG_ZUFRIEDEN_2_PTSD': [1, 2, 3, np.nan, np.nan, 4],
    'STIMMUNG_ZUFRIEDEN_2_CUD': [np.nan, np.nan, np.nan, 5, 6, np.nan],
    'STIMMUNG_ENERGIE_2_PTSD': [7, 8, 9, np.nan, np.nan, 10],
    'STIMMUNG_ENERGIE_2_CUD': [np.nan, np.nan, np.nan, 11, 12, np.nan],
    'SOZKT_PTSD_4': ['home', 'hone', 'hole', np.nan, np.nan, 'hoje'],
    'SOZKT_CUD_4': [np.nan, np.nan, np.nan, 'home', 'hone', np.nan],
}

df = pd.DataFrame(data)

new_row = pd.Series(range(df.shape[1]), index=df.columns)
df = pd.concat([pd.DataFrame([new_row]), df], ignore_index=True)
df.iloc[0] = pd.to_numeric(df.iloc[0], errors='coerce').fillna(0).astype(int)
#df.set_index(df.iloc[0].astype(str), inplace=True)

# Drop the first row (which is now the index)
#df = df.drop(df.index[0])

# Sort columns based on the index values in the first row
df = df.sort_values(by=df.iloc[0], axis=1)



# Remove "CUD" and "PTSD" from column names
df.columns = df.columns.str.replace('_CUD|_PTSD', '')

# Group by column names and merge columns
merged_df = df.groupby(df.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())), axis=1))
merged_df = merged_df[df.columns]
print(merged_df)



"""
# Extract common prefixes from column names
prefixes = df.columns.str.extract(r'^(.+?)_(?:PTSD|CUD)_\d+$', expand=False)

# Create a dictionary to store merged columns
merged_columns = {}

# Iterate over unique prefixes and merge columns
for prefix in prefixes.dropna().unique():
    columns_to_merge = df.columns[df.columns.str.startswith(prefix)].tolist()
    merged_columns[prefix] = df[columns_to_merge].fillna('').apply(lambda x: ' '.join(x), axis=1).replace('', np.nan)

# Include non-matching columns in the merged DataFrame
non_matching_columns = df.columns[~df.columns.str.contains(r'^(.+?)_(?:PTSD|CUD)_\d+$')].tolist()
for column in non_matching_columns:
    merged_columns[column] = df[column]

# Create a new DataFrame with merged columns
merged_df = pd.DataFrame(merged_columns)

# Print the result
print(merged_df)
"""