import numpy as np
# Import models 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import scoring methods
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score

def performance(model, X_train, y_train, k_folds=5):   
    """Assess the performance of a machine learning model using cross-validation.

    Args:
        model (class): Machine learning model to assess.
        X_train (numpy array): Feature matrix of the training set.
        y_train (numpy array): Target variable of the training set.
        k_folds (int, optional): Number of folds for cross-validation. Defaults to 5.

    Returns:
        avg_accuracy (float): Average accuracy across folds.
        avg_f1 (float): Average F1 score across folds.
    """
    # Create StratifiedKFold
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Define scoring metric 
    f1_scorer = make_scorer(f1_score, average='weighted')
    accuracy_scorer = make_scorer(accuracy_score)

    # Perform cross-validation and calculate F1 score
    f1 = cross_val_score(model, X_train, y_train, cv=kf, scoring=f1_scorer)
    accuracy = cross_val_score(model, X_train, y_train, cv=kf, scoring=accuracy_scorer)
    

    # Calculate average F1 scores
    avg_f1 = np.mean(f1)
    avg_accuracy = np.mean(accuracy)
   
    return avg_f1, avg_accuracy