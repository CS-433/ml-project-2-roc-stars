import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data/cleaned_data.csv', sep=",", header=0)
print(df)