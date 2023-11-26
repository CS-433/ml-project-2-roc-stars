# ML Project 2
In this machine learning project, we want to forecast the diagnostic group of patients from their intrusion memories characteristics. The two diagnostic groups are Post-Traumatic Stress Disorder (PTSD) and Cocaine Use Disorder (CUD). We have data of 1001 individuals and over 600 features. This is a classification problem. 


## Code structure
The project consists of the following files:
- `run.py` : generates predictions using the best model.
- `implementations.py` : contains all 6 functions asked, as well as additional functions.
- `data_cleansing`: generates the cleaned data sets `x_train_clean.csv` and `x_test_clean.csv`.
- `helpers.py` : contains helper functions to load .csv files and generate .csv submissions.
- `predictions.py` : generates predictions with other models.

  
The folder `plots`contains the following files:
- `visualization.py` : generates data visualization plots used in report.
- `best_hyperparameters.py` : generates ROC curve of best model as well as plot of tuning of learning rate hyperparameter.
- Additional files: plots as well as `.csv` files called in previous `.py`files.
## Datasets
The predictions are made using three files:
- `x_train.csv` : dataset used to train and tune the methods.
- `y_train.csv` : labels used to train and tune the methods, contains binary values (-1,1).
- `x_test.csv` : dataset used to generate submitted predictions once tuning of methods is optimized.

Link to download the dataset: https://github.com/epfml/ML_course/blob/master/projects/project1/data/dataset.zip

## How to run the code
1. Download `data_to_release` and put it in a folder called `data`.
2. Run the `data_cleansing.py` file to generate `x_train_clean.csv` and `x_test_clean.csv`.   
3. Run the `run.py` file to generate our predictions with the best model.  

If an error occurs (e.g. "function ... not defined"), run the `implementations.py` file.

## License
© 2023 GitHub, Inc.


EPFL © [Clara Chappuis](https://github.com/clarachappuis), [Renuka Singh Virk](https://github.com/renukasinghvirk), [Camille Pittet](https://github.com/camicc)

