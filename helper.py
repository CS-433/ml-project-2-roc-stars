import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

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

def convert_int_columns_to_int(df):
    """
    Convert columns with integer values (even if represented as floats) to integer type in a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with columns containing integer values converted to integer type.

    Notes:
    - The function iterates over each column in the DataFrame.
    - If a column is of type float and all its values are integers, it converts the column to integer type using `astype(int)`.
    - Columns with non-integer values or other data types remain unchanged.
    """
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.floating) and np.all(df[col] % 1 == 0):
            df[col] = df[col].astype(int)
    return df

def remove_outliers(X, iqr_multiplier=1.5):
    """
    Remove outliers from continuous columns of a DataFrame using the IQR (Interquartile Range) method.

    Parameters:
    - X (pd.DataFrame): Input DataFrame containing a mix of categorical and numerical columns.
    - iqr_multiplier (float, optional): Multiplier for determining the outlier boundaries based on the IQR.
      Defaults to 1.5.

    Returns:
    - np.ndarray: NumPy array with outliers replaced by the median of each numerical column.
    
    Notes:
    - Categorical columns are not processed; outlier removal is applied only to numerical columns.
    - Outliers are identified based on the IQR method for each numerical column.
    - Integer columns are converted to integer types before outlier removal on float columns.
    - The function converts the DataFrame to a NumPy array for processing.
    """
    # Identify and convert integer columns to integer type
    X = convert_int_columns_to_int(X)

    # Create a mask for numerical columns (after integer conversion)
    numerical_mask = X.dtypes.apply(lambda x: np.issubdtype(x, np.number))

    # Convert DataFrame to NumPy array
    X_array = X.values

    for col_ind in range(X_array.shape[1]):
        col = X_array[:, col_ind]

        if numerical_mask.iloc[col_ind]:
            # Process only numerical columns
            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            iqr = q3 - q1

            # Define the lower and upper bounds for outliers
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            # Replace outliers with the median of the column
            col[col < lower_bound] = np.median(col)
            col[col > upper_bound] = np.median(col)

    return X_array

def logistic(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model using Grid Search for hyperparameter tuning and evaluate its performance.
    This function's sole purpose is to assess whether outliers removal improves performance or not, c.f. file outliers.py for implementation.

    Parameters:
    - X_train (array-like): Training features.
    - X_test (array-like): Testing features.
    - y_train (array-like): Training labels.
    - y_test (array-like): Testing labels.

    Returns:
    - dict: Dictionary containing the best hyperparameters and performance metrics.

    Notes:
    - The model is trained with hyperparameter tuning using Grid Search.
    - The hyperparameters considered are 'C' (inverse regularization strength) and 'penalty'.
    - The model is evaluated on the provided test set.
    - Performance metrics include accuracy and F1 score.
    """
    # Hyperparameters to try
    params_lr = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100], 
        'penalty': ['l1', 'l2'], 
    }

    # Model definition
    logreg_model = LogisticRegression(random_state=42, max_iter=1000)

    # WITH OUTLIERS
    # Tuning and fitting using Grid Search
    grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)

    # Best hyperparams
    best_params_lr = grid_search.best_params_
    print(best_params_lr) # C = 1, penalty='l2'

    # Assess accuracy and f1 score on X_test_nout
    best_lr_model = grid_search.best_estimator_
    y_pred_lr = best_lr_model.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    f1score_lr = f1_score(y_test, y_pred_lr)
    print("LR with outliers\n","Accuracy: ", accuracy_lr, "\n", "F1 score :", f1score_lr) 
