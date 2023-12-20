# < -------------------------------Import libraries-------------------------------------- >
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import SparsePCA, PCA
from sklearn.linear_model import LogisticRegression


# < -----------------------------Define path to save plots------------------------------- >
# Directory where the images will be saved
path = "plots/"

# < -----------------------------------Load dataset-------------------------------------- >
# Load dataset
df = pd.read_csv('Datasets/final_data.csv', sep=";", header=0, index_col=0)

# Separate X from prediction y
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

# Separate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# < --------------------------------------2D PCA----------------------------------------- >
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
plt.figure(figsize=(8,6)) 
plt.scatter(x_tr_PCA_CUD[:, 0], x_tr_PCA_CUD[:, 1], s=20, color='dodgerblue', label='CUD', alpha=0.3)
plt.scatter(x_tr_PCA_PTSD[:, 0], x_tr_PCA_PTSD[:, 1], s=20, color='red', label='PTSD', alpha=0.3)
plt.xlabel('Principal Component 1', fontsize=18)
plt.ylabel('Principal Component 2', fontsize=19)
plt.grid(True)
plt.savefig(path + "pca_colors.png")
plt.rc('font', size=18)
plt.legend()
plt.show()

# < ---------------------------------------3D PCA---------------------------------------- >
# Plot the 3D PCA result with colored points
pca = PCA(n_components=3)
x_tr_PCA = pca.fit_transform(X_train)

# Define colors for each label
label_colors = {0: 'red', 1: 'green'} 
ind_CUD = np.where(y_train == 0)
ind_PTSD = np.where(y_train == 1)
x_tr_PCA_CUD = x_tr_PCA[ind_CUD]
x_tr_PCA_PTSD = x_tr_PCA[ind_PTSD]

axis_fontsize = 10
fig = plt.figure(figsize=(10,10))
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
ax.set_yticklabels(['-10', '-5', '0', '5', '10']) 

ax.w_xaxis.set_pane_color((1, 1, 1, 1.0)) 
ax.w_yaxis.set_pane_color((1, 1, 1, 1.0))
ax.w_zaxis.set_pane_color((1, 1, 1, 1.0))
plt.show()

# < ----------------------------Ratio of variance explained by PCs----------------------- >
# Higher dimension PCA
pca = PCA(n_components = 10)
x_tr_PCA = pca.fit_transform(X_train)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio obtained with PCA:", explained_variance_ratio)

# < -----------------------Visualize categorical vs. continuous features----------------- >
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
plt.figure(figsize=(8,6))
plt.bar(types, ratios, color='dodgerblue', width=0.5)
plt.ylabel("Ratio", fontsize=30)
plt.ylim(0,1)
plt.xticks(types, ["Continuous", "Categorical"], fontsize=30)
plt.savefig(path + "catcont.png")
plt.rc('font', size=18)
plt.show()

# < ---------------------------------Visualize missing values---------------------------- >
# Plots missing values 
#Import df before nan removal
df_nan = pd.read_csv('data/nan_data.csv', sep=";", header=0, index_col=0)

# Calculate the percentage of NaN values in each column
df_nan = (df_nan.isna().mean() * 100)
nans = df_nan.values

# Define feature vector
features = np.arange(1, len(nans)+1)

# Plot NaN ratio
plt.figure(figsize=(10, 8)) 
plt.bar(features, nans, color='royalblue')
plt.xlabel("Feature number", fontsize=30)
plt.ylabel("Ratio of NaN values", fontsize=30)
plt.savefig(path + "nan.png")
plt.rc('font', size=18)
plt.show()

# Number fo features larger than 15 %
features_nan = np.sum(nans > 15)
print( features_nan , "% of features have a percentage of NaN higher than 15%. ")

# < -------------------------------------Risk Status------------------------------------- >
ptsd_count = sum(y == 0)
cud_count = sum(y == 1)
total = len(y)
ratios = [ptsd_count/total, cud_count/total]
types = ['PTSD', 'CUD']
plt.figure(figsize=(8, 6))
plt.bar(types, ratios, color='crimson', width=0.5)
plt.ylabel("Count", fontsize=30)  
plt.ylim(0, 1) 
plt.xticks(fontsize=30)  
plt.savefig(path + "bias.png")
plt.rc('font', size=30)
plt.yticks(fontsize=18)
plt.show()
# < ---------------------------Logistic Regression Weights------------------------------- >
# Plot weigths linear regression
# Create the model
logreg_model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg_model.predict(X_test)

# Display the weights
weights = logreg_model.coef_
intercept = logreg_model.intercept_ # maybe it could also bring smth to look at the intercept

# Display the weights
weights = logreg_model.coef_[0]  # Assuming binary classification, so there is only one set of weights
intercept = logreg_model.intercept_

# Selects significant weights
weights[np.abs(weights) < 0.5] = 0

# Get the indices of non-zero elements
nonzero_indices = np.nonzero(weights)[0]

# Plot the norm of the weights
plt.bar(range(len(weights)), weights)
plt.xlabel('Feature Index')
plt.ylabel('Norm of Weights')
plt.title('Norm of Weights in Logistic Regression')
plt.savefig(path + "weights.png")
plt.show()

# Find inidices with significant weights
nonzero_indiced = np.nonzero(weights)

# Initialize feature vector
features = []

# Replace indice by feature
for indice in nonzero_indices:
    features.append(df.columns[indice])
    
# Display features
print(" Features with significant weights: ", features)
    

# < -------------------------------------ROC CURVE--------------------------------------- >
# Best decision threshold for Logistic Regression 
logreg = LogisticRegression(C=1, penalty='l2')
logreg.fit(X_train, y_train)
thrs = np.linspace(0.2,0.8,13)
accs = []
f1s = []
for thr in thrs:
    y_prob = logreg.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_pred_custom, y_test)
    f1 = f1_score(y_pred_custom, y_test)
    accs.append(acc)
    f1s.append(f1)
    print(f"Threshold {thr}\n","Accuracy: ", acc, "\n", "F1 score :", f1) 

max_ind = np.argmax(f1s)
f1_max = np.max(f1s)
acc_max = accs[max_ind]
thr_max = thrs[max_ind]
print(f"The maximum f1 score is obtained with a threshold set at {thr_max} and renders: \n",
      "Accuracy: ", acc_max, "\n", "F1 score :", f1_max)

# ROC Curve plot
y_prob = logreg.predict_proba(X_test)[:, 1]

# Compute the ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
closest_index = np.argmin(np.abs(thresholds - 0.5))

# Compute the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.scatter(fpr[closest_index], tpr[closest_index], color='black', marker='o', label='Selected Threshold = 0.5', s=80, zorder=10)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.savefig("plots/ROC_curve.png")
plt.legend(loc='lower right')
plt.show()