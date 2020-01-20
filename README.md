# Predicting phenotypic polymyxin resistance in *Klebsiella pneumoniae* through machine learning analysis of genomic data

There are two main components to the pipeline for predicting polymyxin resistance using machine learning. The first step is to convert raw genetic data into predictive features. We have done this using a reference-based approach that relies on variant calling and IS, as well as a reference-free approach that relies on k-mer detection. For details on the reference-based approach, please see Macesic et al. *Clin Infect Dis* 2019 Sep 12. (PMID 31513705). For the k-mer detection in the reference-free approach, we used the [DSK k-mer counter](https://github.com/GATB/dsk) set at n=31. Sample datasets that we used as inputs in the manuscript can be found in `data`, as well as a file describing the datasets `data/pbr_ml_final_datasets.xlsx`.



# Input datasets
All input datasets are CSV files with particular layouts. Please see the included sample datasets in `data` to understand the layout.

## Datasets without GWAS filtering
To run the pipeline without using GWAS filtering, two files are needed:
- Metadata file with naming format of `dataset_xx_metadata.csv`. This includes the isolate name, polymyxin susceptibility as a binary variable and the dataset that the isolate came from.
- Feature file with naming format of `dataset_xx_full.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix.
1. There are four main scripts to run. `run_model_grid_search.py` runs the grid search over multiple datasets and model configurations and saves a pickle object of grid search CV object. `extract_performance_metrics.py` Saves a csv file with the best model's performance on each dataset. `extract_feature_importance.py` extracts feature importance of the best model configuration for multiple models on a given dataset. Finally, `run_gwas_bootstrap.py` bootstraps a gwas dataset in order to estimate confidence intervals for performance.
2. `src/mlPipeline/modelTraining.py`: all functions needed to perform girdsearch CV, feature selection, and evaluation of the models.
3. `src/mlPipeline/plotting.py`: All functions needed to plot performance after models have been trained.
4. `data/MLData/pbr_ml_project_datasets_20180423.zip` This zip file contains all data needed for the pipeline. There are two kinds of datasets within. dataset_i_full.csv contains all data (both training and testing) and maps observations (rows) to features (columns). This file is read in and used in 10-fold cv to determine the best models. data_set_i_train.csv, data_set_i_test.csv and data_set_i_gwas.csv contain the training validation, and GWAS correlation scores respectively. Training and validation are self explanatory and are used instead of 10-fold CV because of the fact that GWAS scores need to be trained on the data. GWAS correlation scores are used for feature selection.

## Reference-free datasets
To run the pipeline using GWAS filtering, four files are needed:
- Metadata file with naming format of `dataset_xx_metadata.csv`. This includes the isolate name, polymyxin susceptibility as a binary variable and the dataset that the isolate came from.
- Training feature file with naming format of `dataset_xx_train.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix. You need to split your data into a training and testing set as this approach does not use cross-validation due to the risk of data leakage when the GWAS is performed.
- Testing feature file with naming format of `dataset_xx_test.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix. This is the data that you will test on
- GWAS file with naming format of `dataset_xx_gwas.csv`. This file needs to contain the feature names and *P* values from your GWAS. We used [treeWAS](https://github.com/caitiecollins/treeWAS) in the simultaneous mode in the manuscript.

# Files needed to run the prediction pipeline
1. `run_model_grid_search.py`: This script will use the input files described above and run without GWAS filtering if two input files are present. It will run with GWAS filtering if the correct four files are present. It performs feature selection, model training, and model evaluation. The output is a .pkl file that is a saved version of the model. There are several flags for the script:
-nj: number of cores
-nf: number of folds
-dp: path to folder where input files are
-mp: path where the model should be saved to
-idx: names/numbers of datasets to be tested. Needs to be entered in the following format: '1 2 3' (would test for dataset_1, dataset_2, dataset_3)

2. `extract_performance_metrics.py`: This script will use a .pkl file with the models as an input and output CSVs of the performance metrics.

3. `extract_feature_importance.py`: This script will use a .pkl file with the models as an input and output CSVs with metrics for feature importance for models using Logistic Regression, Random Forests and Gradient Boosted Trees Classifier.

4. `run_gwas_data_bootstrap.py`: This script will use a .pkl file with datasets that used GWAS filtering and output results of bootstrapping. This is done as cross-validation is not performed for GWAS-filtered datasets.

# Libraries and Modules needed to run the project
All code is implemented in Python. The pipeline requires numpy, pandas, matplotlib and sk-learn in order to run correctly.
This can be installed as Conda environment using the `data\environment.yml` file.

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
