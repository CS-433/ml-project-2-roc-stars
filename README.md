# ML Project 2
In this machine learning project, we want to forecast the diagnostic group of patients from their intrusive memories characteristics. 
The two diagnostic groups are Post-Traumatic Stress Disorder (PTSD) and Cocaine Use Disorder (CUD). 
In the raw dataset we have data of 1001 surveys and over 600 features. 
This is a classification problem. 


The project is done under the supervision of Dr. Lina Dietker at the Experimentelle Psychopathologie und Psychotherapie laboratory at the University of Zurich.


## Code structure
The project consists of the following files:
- `data_cleaning.py`: generates the cleaned dataset `final_data.csv`.
- `helper.py` : contains helper functions.
- `methods.py` : contains the tuning as well as performance assessment of each methods except for multilayer perceptron.
- `neural_networks.py` : contains the tuning as well as performance assessment of multilayer perceptron. 
- `ablation.py` : generates predictions using logistic regression on both the dataset without outliers and with feature augmentation to assess performance.
- `plots.py` : generates data visualization plots.
- `ethics.py` : generates plots of age, gender, origins and years of education.
- `run.py` : generates predictions using the best model.

  
The folders `plots` and `ethics` contain data visualization `.png` files for the report.


## How to run the code 
1. Clone github repository.
2. Download `EMemory_data.csv` from this [site](https://filesender.switch.ch/filesender2/?s=download&token=59c86fac-3ab3-46c7-9c44-7a6b2a3d6f3c) and store it in a folder called `data`.
3. Create a folder `Datasets`.
4. Run the `data_cleaning.py` file to generate `final_data.csv`.   
5. Run the `run.py` file to generate our predictions with the best model.  

If an error occurs (e.g. "function ... not defined"), run the `helper.py` file.


Note that the `Data_Patients.csv` file is confidential and could not be shared. The file `ethics.py` cannot be run without it, however the plots it generates are in the folder `plots/ethics`. 


Please note that the link to download the 'EMemory_data.csv' file will expire on February 3, 2024.

## Libraries
The required libraries that must be installed are listed in the `requirements.txt` file. 

## License
© 2023 GitHub, Inc.


EPFL © [Clara Chappuis](https://github.com/clarachappuis), [Renuka Singh Virk](https://github.com/renukasinghvirk), [Camille Pittet](https://github.com/camicc)

