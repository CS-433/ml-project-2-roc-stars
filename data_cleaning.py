# < -------------------------------Import libraries--------------------------------------->
import pandas as pd
from helper import *

# < ------------------------------Load dataset here--------------------------------------->
df_raw = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# < --------------------------------------Run--------------------------------------------->
# Create clean dataframe
final_df = clean(df_raw)
final_df.to_csv('data/final_data.csv', sep=';')
# < -------------------------------------------------------------------------------------->