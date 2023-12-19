# ML Project 2
In this machine learning project, we want to forecast the diagnostic group of patients from their intrusion memories characteristics. The two diagnostic groups are Post-Traumatic Stress Disorder (PTSD) and Cocaine Use Disorder (CUD). We have data of 1001 individuals and over 600 features. This is a classification problem. 


## Code structure
The project consists of the following files:
- `data_cleansing`: generates the cleaned data set `final_data.csv`.
- `helper.py` : contains helper functions.
- `methods.py` : contains the tuning as well as performance assessment of each methods except for multilayer perceptron.
- `neural_networks.py`: contains the tuning as well as performance assessment of multilayer perceptron. 
- `outliers.py` : generates predictions using logistic regression on both the dataset with and without outliers.
- `plots.py` : generates data visualization plots.
- `ethics.py` : generates plots of age, gender, origins and years of education.
- `run.py` : generates predictions using the best model.

  
The folder `plots`contains data visualization `.png`files for the report as well as the `ethics` folder, which also contains `.png` files.

## Datasets
The tuning and predictions are made using `final_data.csv`, which is split into `X_train`, `y_train`, `X_test` and `y_test` in each `.py` file where it is necessary.

Link to download the dataset: https://github.com/epfml/ML_course/blob/master/projects/project1/data/dataset.zip

## How to run the code
1. Download `EMemory_data.csv` and put it in a folder called `data`.
2. Run the `data_cleaning_pd.py` file to generate `final_data.csv`.   
3. Run the `run.py` file to generate our predictions with the best model.  

If an error occurs (e.g. "function ... not defined"), run the `helper.py` file.

## License
© 2023 GitHub, Inc.


EPFL © [Clara Chappuis](https://github.com/clarachappuis), [Renuka Singh Virk](https://github.com/renukasinghvirk), [Camille Pittet](https://github.com/camicc)

