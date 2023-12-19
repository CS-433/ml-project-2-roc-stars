# < -------------------------------Import libraries-------------------------------------- >
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# < -----------------------------------Functions----------------------------------------- >

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
    grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='accuracy')
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

def model_performance(model, X_test, y_test, CV=True):
    """
    Given a model and a test set, this method makes predictions and print the accuracy, f1 score and, if a cross-
    validation was performed, the best parameters of the model.

    Parameters:
    - model : the model to use to make the predictions
    - X_test (array-like): Testing features.
    - y_test (array-like): Testing labels.
    - CV (Bool) : indicates whether the model has been tuned with cross-validation. By default: True

    """
    if CV:
        best_model = model.best_estimator_
        y_pred = best_model.predict(X_test)
        best_parameters = model.best_params_
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Accuracy: ", accuracy, "\n", "F1 score :", f1, "\n", "Best parameters :", best_parameters) 
    else:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Accuracy: ", accuracy, "\n", "F1 score :", f1, "\n") 

def remove_consecutive_duplicates(cell):
    """
    Remove consecutive duplicates from a space-separated string of words.

    Parameters:
    - cell (str): Input string containing space-separated words.

    Returns:
    str: String with consecutive duplicates removed.

    This function takes a string of words, splits it into a list, and removes consecutive
    duplicates, keeping only the first occurrence of each word. It then joins the cleaned
    words back into a string and returns the result.
    """
    words = cell.split()
    cleaned_words = [word for i, word in enumerate(words) if i == 0 or word != words[i-1]]
    return ' '.join(cleaned_words)


def clean(df_raw):
    """
    Clean and preprocess a raw dataframe from survey data.

    Parameters:
    - df_raw (pd.DataFrame): The raw dataframe containing survey data.

    Returns:
    - final_df (pd.DataFrame): Cleaned and preprocessed dataframe.

    The function performs the following steps:
    1. Convert '<no-response>' and '<not-shown>' to NaN.
    2. Drop columns with only NaN values.
    3. Convert predictions to 0 (PTSD) and 1 (CUD) based on 'SURVEY_NAME'.
    4. Drop unnecessary columns.
    5. Group by column names and merge columns.
    6. Remove consecutive duplicates in cells.
    7. Replace values larger than 10 with NaN.
    8. Replace non-standard representations with actual NaN.
    9. Save the cleaned dataframe to 'data/nan_data.csv'.
    10. Calculate the percentage of NaN values in each column.
    11. Select columns with less than 15% NaN values.
    12. Concatenate selected columns with labels.
    13. Drop rows with 80% or more empty values.
    14. Separate continuous and binary columns to handle NaN values.
    15. Fill NaN values in continuous columns with the median and standardize them.
    16. Fill NaN values in binary columns with the mode.
    17. Concatenate continuous, binary columns, and labels.
    18. Assess and print the three smallest standard deviation values.
    19. Drop columns with low standard deviation (<0.1).
    20. Return the final cleaned dataframe.
    """
    # Convert <no-response> and <not-shown> to NaN
    df_raw[df_raw == '<not-shown>'] = np.nan
    df_raw[df_raw == '<no-response>'] = np.nan

    # Drop columns with only NaN values
    df = df_raw.dropna(axis=1, how='all')

    # Convert target variables to 0 (PTSD) and 1 (CUD)
    labels = df['SURVEY_NAME'].copy()  # Make a copy to avoid chained indexing
    labels.loc[labels == 'Intrusionsfragebogen (T)'] = 0
    labels.loc[labels == 'Intrusionsfragebogen (K)'] = 1

    # Drop target variables column
    df.drop(columns=['SURVEY_NAME'], inplace=True)

    # Drop all the columns ending with _RT
    columns_to_drop = [col for col in df.columns if col.endswith('_RT')]
    df.drop(columns=columns_to_drop, inplace=True)

    # Remove "_CUD" from column names
    df.columns = df.columns.str.replace('_CUD', '')
    # Remove "_PTSD" from column names
    df.columns = df.columns.str.replace('_PTSD', '')

    # Group by column names and merge columns
    merged_df = df.groupby(df.columns, axis=1).apply(lambda x: x.apply(lambda y: ' '.join(map(str, y.dropna())) if len(x.columns) > 1 else str(y.iloc[0]), axis=1))

    # Remove duplicates within the same cell (e.g. 1 1 -> 1)
    cleaned_df = merged_df.applymap(remove_consecutive_duplicates)

    # Replace numbers to numeric objects and remove strings
    cleaned_df = cleaned_df.apply(pd.to_numeric, errors='coerce')

    # Replace values larger than 10 to NaN 
    cleaned_df = cleaned_df.mask(cleaned_df > 10, np.nan)

    # Replace non-standard representations with actual NaN
    cleaned_df.replace(['nan', ''], np.nan, inplace=True)

    # Saves dataframe for NaN Visualisation
    cleaned_df.to_csv('data/nan_data.csv', sep = ';')

    nan_perc_limit = 15
    # Calculate the percentage of NaN values in each column
    nan_percentage = (cleaned_df.isna().mean() * 100)

    # Select columns with less than 15% NaN values
    selected_columns = nan_percentage[nan_percentage <= nan_perc_limit].index
    cleaned_df = cleaned_df[selected_columns]

    # Add target column at the end of dataset
    cleaned_df = pd.concat([cleaned_df, labels], axis = 1)

    # Drop rows with 80% or more empty values
    threshold = 0.8
    cleaned_df_filtered = cleaned_df.dropna(thresh=int(cleaned_df.shape[1] * (1 - threshold)))
    new_labels = cleaned_df_filtered['SURVEY_NAME']

    # Separate continuous and binary columns to handle NaN values
    continuous_df = cleaned_df_filtered.loc[:, ~cleaned_df_filtered.columns.str.endswith(tuple(map(str, range(10)))) & (cleaned_df_filtered.columns != 'SURVEY_NAME')]
    continuous_df.fillna(continuous_df.median(), inplace=True)

    binary_df = cleaned_df_filtered.loc[:, cleaned_df_filtered.columns.str.endswith(tuple(map(str, range(10))))]
    binary_df[binary_df.columns] = binary_df[binary_df.columns].apply(lambda x: x.fillna(x.mode().iloc[0]))
    
    # Standardize continuous columns
    scaler = StandardScaler()
    continuous_df[continuous_df.columns] = scaler.fit_transform(continuous_df[continuous_df.columns])
    
    # Regroup continous, binary columns and labels
    filled_df = pd.concat([continuous_df, binary_df, new_labels], axis = 1)

    # Assess smallest std features for report
    numeric_columns = filled_df.select_dtypes(include='number')
    std_values = numeric_columns.std()
    second_min_std = std_values.nsmallest(3)
    #print(second_min_std)

    # Drop columns with low std (<0.1)
    min_std = 0.1
    high_std_columns = filled_df.columns[filled_df.std() > min_std]
    final_df = filled_df[high_std_columns]
    
    return final_df
