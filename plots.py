import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Directory where the images will be saved
path = "plots/"

# Set figures fontisze
font_size = 12

# Set figures dimensions
im_size = (8,6)

# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

# Seperate X from prediction y
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

# Seperate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
x_tr_PCA = pca.fit_transform(X_train)

# Define colors for each label
label_colors = {0: 'red', 1: 'green'} 

# Plot the 2D PCA result with colored points
ind_CUD = np.where(y_train == 0)
ind_PTSD = np.where(y_train == 1)
x_tr_PCA_CUD = x_tr_PCA[ind_CUD]
x_tr_PCA_PTSD = x_tr_PCA[ind_PTSD]

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Plot the 2D PCA result
plt.figure(figsize=im_size) 
plt.scatter(x_tr_PCA_CUD[:, 0], x_tr_PCA_CUD[:, 1], s=20, color='dodgerblue', label='CUD', alpha=0.3)
plt.scatter(x_tr_PCA_PTSD[:, 0], x_tr_PCA_PTSD[:, 1], s=20, color='red', label='PTSD', alpha=0.3)
plt.xlabel('Principal Component 1', fontsize=font_size)
plt.ylabel('Principal Component 2', fontsize=font_size)
plt.grid(True)
plt.savefig(path + "pca_colors.png")
plt.rc('font', size=18)
plt.legend()
plt.show()


# 3d PCA plot
 
# Apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components = 3)
x_tr_PCA = pca.fit_transform(X_train)

# Plot the 3D PCA result with colored points
ind_CUD = np.where(y_train == 0)
ind_PTSD = np.where(y_train == 1)
x_tr_PCA_CUD = x_tr_PCA[ind_CUD]
x_tr_PCA_PTSD = x_tr_PCA[ind_PTSD]

axis_fontsize = 8
fig = plt.figure()
plt.figure(figsize=im_size)
ax = fig.add_subplot(111, projection='3d')
x = x_tr_PCA_CUD[:, 0]
y = x_tr_PCA_CUD[:, 1]
z = x_tr_PCA_CUD[:, 2]
ax.scatter(x, y, z, c='b', marker='o')
x1 = x_tr_PCA_PTSD[:, 0]
y1 = x_tr_PCA_PTSD[:, 1]
z1 = x_tr_PCA_PTSD[:, 2]
ax.scatter(x1, y1, z1, c='r', marker='o')
ax.set_xlabel('Principal Component 1', fontsize=axis_fontsize)
ax.set_ylabel('Principal Component 2', fontsize=axis_fontsize)
ax.set_zlabel('Principal Component 3', fontsize=axis_fontsize)
ax.set_title('3D PCA Plot')
ax.view_init(elev=0, azim=80)
plt.show()

# CATEGORICAL AND CONTINOUS FEATURES

# Categorical headers keyword
categorical_headers = ['REAKTION', 'MODALITAET', 'STRATEGIE', 'TRIGGER']

# Get headers
headers_list = df.columns.tolist()

# Intialize categorical count
categorical = 0

# Check if categorical or continous
for keyword in categorical_headers:
    
    matching_strings = [string for string in headers_list if keyword in string]
    categorical += len(matching_strings)

# Deduce continous features
nbr_features = len(headers_list)
continuous = nbr_features - categorical

# Calculates ratio of continous and categorical features
categorical_ratio = categorical / nbr_features
continuous_ratio = continuous / nbr_features
ratios = [continuous_ratio, categorical_ratio]
types = ['Continuous', 'Categorical']

# Plots categorical vs continuous features
plt.figure(figsize=im_size) 
plt.bar(types, ratios, color='grey', width=0.5)
plt.ylabel("Ratio", fontsize=font_size)
plt.ylim(0,1)
plt.xticks(types, ["Continuous", "Categorical"], fontsize=font_size)
plt.savefig(path + "catcont.png")
plt.rc('font', size=18)
plt.show()