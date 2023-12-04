import numpy as np
# Import models 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import scoring methods
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score

def average_f1_scores(x_train, y_train, k_folds=5):
    # Create models
    knn_model = KNeighborsClassifier()
    svm_model = SVC()
    logreg_model = LogisticRegression(max_iter=1000)

    # Create StratifiedKFold
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Define scoring metric 
    f1_scorer = make_scorer(f1_score, average='weighted')

    # Perform cross-validation and calculate average F1 scores
    knn_f1 = cross_val_score(knn_model, x_train, y_train, cv=kf, scoring=f1_scorer)
    svm_f1 = cross_val_score(svm_model, x_train, y_train, cv=kf, scoring=f1_scorer)
    logreg_f1 = cross_val_score(logreg_model, x_train, y_train, cv=kf, scoring=f1_scorer)

    # Calculate average F1 scores
    avg_f1_knn = np.mean(knn_f1)
    avg_f1_svm = np.mean(svm_f1)
    avg_f1_logreg = np.mean(logreg_f1)

    return avg_f1_knn, avg_f1_svm, avg_f1_logreg