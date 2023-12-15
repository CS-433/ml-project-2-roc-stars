import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.decomposition import SparsePCA

from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV



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


# Trying sparse PCA
spca = SparsePCA(n_components=2)
spca_result = spca.fit_transform(X_train)

# Create a DataFrame for visualization (optional)
df_spca = pd.DataFrame(data=spca_result, columns=['PC1', 'PC2'])

# Scatter plot
plt.scatter(df_spca['PC1'], df_spca['PC2'])
plt.title('Sparse PCA - Scatter Plot of PC1 vs PC2')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.savefig(path + "pca_sparse.png")
plt.show()



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
print("Explained Variance Ratio obtained with PCA:", explained_variance_ratio)

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

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio obtained with PCA:", explained_variance_ratio)

# Plot the 3D PCA result with colored points
ind_CUD = np.where(y_train == 0)
ind_PTSD = np.where(y_train == 1)
x_tr_PCA_CUD = x_tr_PCA[ind_CUD]
x_tr_PCA_PTSD = x_tr_PCA[ind_PTSD]

axis_fontsize = 10
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
x = x_tr_PCA_CUD[:, 0]
y = x_tr_PCA_CUD[:, 1]
z = x_tr_PCA_CUD[:, 2]
scatter1 = ax.scatter(x, y, z, c='b', marker='o', label='CUD')
x1 = x_tr_PCA_PTSD[:, 0]
y1 = x_tr_PCA_PTSD[:, 1]
z1 = x_tr_PCA_PTSD[:, 2]
scatter2 = ax.scatter(x1, y1, z1, c='r', marker='o', label='PTSD')
ax.set_xlabel('PC 1', fontsize=axis_fontsize)
ax.set_ylabel('PC 2', fontsize=axis_fontsize)
ax.set_zlabel('PC 3', fontsize=axis_fontsize)
ax.set_title('3D PCA Plot of training data')
ax.view_init(elev=0, azim=80)
ax2d = fig.add_axes([0, 0, 1, 1], zorder=-1)
ax2d.set_axis_off()
ax2d.legend([scatter1, scatter2], ['CUD', 'PTSD'], loc='upper left',  bbox_to_anchor=(0.8, 0.7))
ax.set_yticks([-10, -5, 0, 5, 10])
ax.set_yticklabels(['-10', '-5', '0', '5', '10'])  # Replace with your desired labels

ax.w_xaxis.set_pane_color((1, 1, 1, 1.0))  # Adjust RGB and alpha as needed
ax.w_yaxis.set_pane_color((1, 1, 1, 1.0))
ax.w_zaxis.set_pane_color((1, 1, 1, 1.0))
plt.show()

# Higher dimension PCA
pca = PCA(n_components = 10)
x_tr_PCA = pca.fit_transform(X_train)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio obtained with PCA:", explained_variance_ratio)

# CATEGORICAL AND CONTINOUS FEATURES

# Categorical headers keyword
categorical_headers = ['REAKTION', 'MODALITAET', 'STRATEGIE', 'TRIGGER', 'STIMMUNG']

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
print(" Ratio continous features: ", continuous_ratio )
print(" Ratio categorical features: ", categorical_ratio )


# Plots categorical vs continuous features
plt.figure(figsize=im_size) 
plt.bar(types, ratios, color='grey', width=0.5)
plt.ylabel("Ratio", fontsize=font_size)
plt.ylim(0,1)
plt.xticks(types, ["Continuous", "Categorical"], fontsize=font_size)
plt.savefig(path + "catcont.png")
plt.rc('font', size=18)
plt.show()

# Plots missing values 

#Import df before nan removal
df_nan = pd.read_csv('data/nan_data.csv', sep=";", header=0, index_col=0)


# Calculate the percentage of NaN values in each column
df_nan = (df_nan.isna().mean() * 100)

# Converts df into numpy array
nans = df_nan.values

# Define feature vector
features = np.arange(1, len(nans)+1)

# Plot NaN ratio
plt.figure(figsize=(8, 6)) 
plt.bar(features, nans, color='orange')
plt.xlabel("Feature number", fontsize=font_size)
plt.ylabel("Ratio of NaN values", fontsize=font_size)
plt.savefig(path + "nan.png")
plt.rc('font', size=18)
plt.show()

# Number fo features larger than 15 %
features_nan = np.sum(nans > 15)
print( features_nan , "% of features have a percentage of NaN higher than 15%. ")

# t-SNE =================================================================================>
# Hyperparameters to try
params_tsne = {
    'perplexity': [5, 10, 20, 30, 40],
    'learning_rate': [50, 100, 200],
    'n_iter': [250, 500, 1000],
}

# Create t-SNE model
tsne = TSNE(n_components=2)

# Use GridSearchCV to perform the grid search
grid_search = GridSearchCV(tsne, params_tsne, cv=3, scoring='f1_weighted', verbose=1)
grid_search.fit(X, y)

# Best hyperparams
print("Best Hyperparameters:", grid_search.best_params_) # {'learning_rate': 50, 'n_iter': 250, 'perplexity': 5}

# Visualize
tsne = TSNE(learning_rate=50, n_iter=250, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a scatter plot to visualize the reduced-dimensional data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('t-SNE Visualization')
plt.colorbar(scatter, ticks=np.arange(3), label='Target Class')
plt.show()

# PAIRWISE PLOT =======================================================================>
import seaborn as sns

# Assuming X_train is your training set DataFrame
sns.pairplot(X_train, hue='target_variable', diag_kind='kde')
plt.show()