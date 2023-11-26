import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'PARTICIPANT_ID': ['s061255111', 's061255111', 's061255111', 's061255111', 's061255111', 's061255111'],
    'STIMMUNG_ZUFRIEDEN_2_PTSD': [np.nan, np.nan, np.nan, np.nan, np.nan, 4],
    'STIMMUNG_ZUFRIEDEN_2_CUD': [np.nan, np.nan, np.nan, 5, 6, np.nan],
    'STIMMUNG_ENERGIE_2_PTSD': [7, 8, 9, np.nan, np.nan, 10],
    'STIMMUNG_ENERGIE_2_CUD': [np.nan, np.nan, np.nan, 11, 12, np.nan],
    'SOZKT_PTSD_4': ['home', 'hone', 'hole', np.nan, np.nan, 'hoje'],
    'SOZKT_CUD_4': [np.nan, np.nan, np.nan, 'home', 'hone', np.nan],
}

df = pd.DataFrame(data)

# Remove "CUD" and "PTSD" from column names
df.columns = df.columns.str.replace('_CUD|_PTSD', '')

# Group by column names and merge columns
merged_df = df.groupby(df.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())), axis=1))
merged_df.to_csv('data/brouillon_pd.csv', sep = ',')
print(merged_df)