import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/EMemory_data.csv', sep=";", header=0)

# Visualization of the columns distributions
col1_25 = df.columns[:25].to_numpy()
col26_50 = df.columns[26:50].to_numpy()
col51_75 = df.columns[51:75].to_numpy()
col76_86 = df.columns[76:].to_numpy()
cols = [col1_25, col26_50, col51_75, col76_86]

for col_set in cols:
    data = []
    for i in col_set[1:]:
        data.append(df[i])

    fig = plt.figure(figsize=(30, 14))
    name  = col_set[1:]

    for i in range(len(data)):   
        plt.subplot(5, 5,1+i)
        plt.hist(data[i], bins=150)
        plt.title(name[i])
    plt.tight_layout(pad=3.0)
    plt.show()