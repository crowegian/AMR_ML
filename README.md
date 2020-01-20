# Predicting Antimicrobial Resistance

There are two main components to the pipeline for predicting antimicrobial resistance (AMR) using machine learning. The first step is to convert raw genetic data into predictive features using GWAS analysis and mutation annotation software. This code can be found in `src/featureEngineering` along with a README.md file which explains how to use the code. The second step in the pipeline is to perform feature selection, model training, and model evaluation in order to select the best model.The code for this step can be found in `src/mlPipeline/modelTraining.py`. To run the different scripts please see the last section.

# A Quick Explanation of Files Used

1. There are four main scripts to run. `run_model_grid_search.py` runs the grid search over multiple datasets and model configurations and saves a pickle object of grid search CV object. `extract_performance_metrics.py` Saves a csv file with the best model's performance on each dataset. `extract_feature_importance.py` extracts feature importance of the best model configuration for multiple models on a given dataset. Finally, `run_gwas_bootstrap.py` bootstraps a gwas dataset in order to estimate confidence intervals for performance.
2. `src/mlPipeline/modelTraining.py`: all functions needed to perform girdsearch CV, feature selection, and evaluation of the models.
3. `src/mlPipeline/plotting.py`: All functions needed to plot performance after models have been trained.
4. `data/MLData/pbr_ml_project_datasets_20180423.zip` This zip file contains all data needed for the pipeline. There are two kinds of datasets within. dataset_i_full.csv contains all data (both training and testing) and maps observations (rows) to features (columns). This file is read in and used in 10-fold cv to determine the best models. data_set_i_train.csv, data_set_i_test.csv and data_set_i_gwas.csv contain the training validation, and GWAS correlation scores respectively. Training and validation are self explanatory and are used instead of 10-fold CV because of the fact that GWAS scores need to be trained on the data. GWAS correlation scores are used for feature selection.


# Libraries and Modules needed to run the project
All code is implemented in python and R. The machine learning pipeline requires numpy, pandas, and sk-learn in order to run correctly. The feature engineering pipeline requires the libraries tidyverse, reshape2, devtools, treeWAS, and caret. Installation for treeWAS is a bit special but code ahs been provided to download and install the git repository. Please see the README.md file in  `src/featureEngineering` for a detailed explanation of the files for that part of the pipeline.


# Running the pipeline
There are four main scripts which can be run via the command line.

## Running grid search
Explanation for arguments can be found within the script, but -dp is the path to the folder containing all datasets to be run, -mp is the path to save the model to, -nj is the number of jobs, -nf is the number of folds, -idx is the dataset numbers to be run, and --testing runs a smaller grid search for debugging purposes.
`python run_model_grid_search.py -dp data/MLData/pbr_ml_project_datasets_20180423/ \
-mp data/modelDictMultiDataRun.pkl -nj 2 -nf 10 -idx 1 2 10 --testing`

## Extracting performance metrics
Explanations for arguments can be found within the script, but -cm is the metric to be used to compare models, -mp is the pickle file where the model dict is saved, -op is the out path where performance metrics will be written. -cm is necessary as models have information for multiple metrics and different model configurations can perform differently depending on the metric.
`python extract_performance_metrics.py -cm f1 -mp data/modelDictMultiDataRun.pkl \
-op data/model_performance/`

## Extracting feature importance
Explanations for arguments can be found within the script, but -pr is the prefix of the dataset to use, -mp is the pickle file for the model dictionary, -dp is where the feature importance csv file should be written.
`python extract_feature_importance.py -pr dataset_1_ \
-mp data/featureImportanceTestData/modelDictMultiDataRun_datasets_1_2_3_7_20190722.pkl \
-dp data/featureImportanceTestData/dataset_1_full.csv`

## Bootstrapping GWAs datasets
This script is kind of special in that all the files to use are hardcoded to run over all the gwas datasets using specific pickle files.
`python run_gwas_data_bootstrap.py`
