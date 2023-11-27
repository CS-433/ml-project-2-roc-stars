import pandas as pd
import numpy as np

# Example DataFrame
data = {
    'PARTICIPANT_ID': ['s061255111', 's061255111', 's061255111', 's061255111', 's061255111', 's061255111'],
    'STIMMUNG_ZUFRIEDEN_2_PTSD': [np.nan, np.nan, np.nan, np.nan, np.nan, 4],
    'STIMMUNG_ZUFRIEDEN_2_CUD': [np.nan, np.nan, np.nan, 5, 6, np.nan],
    'STIMMUNG_ENERGIE_2_PTSD': [7, 8, 9, np.nan, np.nan, 10],
    'STIMMUNG_ENERGIE_2_CUD': [2, 'yolo', np.nan, 11, 12, np.nan],
    'SOZKT_PTSD_4': ['home', np.nan, 'hole', np.nan, np.nan, 'hoje'],
    'SOZKT_CUD_4': [np.nan, np.nan, np.nan, 'home', 'hone', np.nan],
}

dg = pd.DataFrame(data)


# Make column names unique
#dg.columns = pd.io.parsers.ParserBase({'names': dg.columns})._maybe_dedup_names(dg.columns)

# Remove "_CUD" from column names
dg.columns = dg.columns.str.replace('_CUD', '')

# Remove "_PTSD" from column names
dg.columns = dg.columns.str.replace('_PTSD', '')

print(dg)

# Remove "CUD" and "PTSD" from column names
#dg.columns = dg.columns.str.replace('_CUD|_PTSD', '')

print(dg)

# Group by column names and merge columns
merged_dg = dg.groupby(dg.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())) if len(x.columns) > 1 else str(y.iloc[0]), axis=1))

def remove_consecutive_duplicates(cell):
    words = cell.split()
    cleaned_words = [word for i, word in enumerate(words) if i == 0 or word != words[i-1]]
    return ' '.join(cleaned_words)

cleaned_dg = merged_dg.applymap(remove_consecutive_duplicates)

print(cleaned_dg)
#merged_dg.to_csv('data/brouillon_pd.csv', sep = ',')