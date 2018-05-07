# Predicting Antimicrobial Resistance

There are two main components to the pipeline for predicting antimicrobial resitance (AMR) using machine learning. The first step is to convert raw genetic data into predictive features using GWAS analysis and mutation annotation software. This code can be found in `src/featureEngineering` along with a README.md file which explains how to use the code. The second step in the pipeline is to perform feature selection, model training, and model evaluation in order to select the best model.The code for this step can be found in `src/mlPipeline/modelTraining.py` and to run this file please use the jupyternotebook in the main directory titled `ML_notebook.ipynb`

# Files Needed to Run the Machine Learning Notebook

1. `ML_notebook.ipynb`: This notebook runs all of the code to read in the ml training data. It doesn ot perform any feature engineering. It only performs feature selection, model training, and model evaluation.
2. `src/mlPipeline/modelTraining.py`: all functions needed to perform girdsearch CV, feature selection, and evaluation of the models.
3. `src/mlPipeline/plotting.py`: All functions needed to plot performance after models have been trained.
4. `data/MLData/pbr_ml_project_datasets_20180423.zip` This zip file contains all data needed for the pipeline. There are two kinds of datasets within. dataset_i_full.csv contains all data (both training and testing) and maps observations (rows) to features (columns). This file is read in and used in 10-fold cv to determine the best models. data_set_i_train.csv, data_set_i_test.csv and data_set_i_gwas.csv contain the training validation, and GWAS correlation scores respectively. Training and validation are self explanatory and are used instead of 10-fold CV because of the fact that GWAS scores need to be trained on the data. GWAS correlation scores are used for feature selection.


# Libraries and Modules needed to run the project
All cope is implemented in python and R, and requies the use of jupternotebooks. The machine learning pipeline requires numpy, pandas, and sk-learn in order to run correctly. The feature engineering pipeline requires the libraries tidyverse, reshape2, devtools, treeWAS, and caret. Installation for treeWAS is a bit special but code ahs been provided to download and install the git repository. Please see the README.md file in  `src/featureEngineering` for a detailed explanation of the files for that part of the pipeline.
