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

# Make column names unique
df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

# Remove "_CUD" from column names
df.columns = df.columns.str.replace('_CUD', '')

# Remove "_PTSD" from column names
df.columns = df.columns.str.replace('_PTSD', '')

# Group by column names and merge columns
merged_df = df.groupby(df.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())) if len(x.columns) > 1 else str(y.iloc[0]), axis=1))

print(merged_df)
# Remove duplicates
def remove_consecutive_duplicates(cell):
    words = cell.split()
    cleaned_words = [word for i, word in enumerate(words) if i == 0 or word != words[i-1]]
    return ' '.join(cleaned_words)

cleaned_df = merged_df.applymap(remove_consecutive_duplicates)

# Print the resulting DataFrame
print(cleaned_df)
cleaned_df.to_csv('data/cleaned_data_pd.csv', sep = ';')
