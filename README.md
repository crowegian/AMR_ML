# Predicting phenotypic polymyxin resistance in *Klebsiella pneumoniae* through machine learning analysis of genomic data

There are two main components to the pipeline for predicting polymyxin resistance using machine learning. The first step is to convert raw genetic data into predictive features. We have done this using a reference-based approach that relies on variant calling and IS, as well as a reference-free approach that relies on k-mer detection. For details on the reference-based approach, please see Macesic et al. *Clin Infect Dis* 2019 Sep 12. (PMID 31513705). For the k-mer detection in the reference-free approach, we used the [DSK k-mer counter](https://github.com/GATB/dsk) set at n=31. Sample datasets that we used as inputs in the manuscript can be found in `data/kmer_datasets.tar.gz` and `data/non_kmer_datasets.tar.gz`, as well as a file describing the datasets `data/pbr_ml_final_datasets.xlsx`.

There are two main components to the pipeline for predicting antimicrobial resistance (AMR) using machine learning. The first step is to convert raw genetic data into predictive features using GWAS analysis and mutation annotation software. This code can be found in `src/featureEngineering` along with a README.md file which explains how to use the code. The second step in the pipeline is to perform feature selection, model training, and model evaluation in order to select the best model.The code for this step can be found in `src/mlPipeline/modelTraining.py`. To run the different scripts please see the last section.

# A Quick Explanation of Files Used


# Input datasets
All input datasets are CSV files with particular layouts. Please see the included sample datasets in `data` to understand the layout.

## Datasets without GWAS filtering
To run the pipeline without using GWAS filtering, two files are needed:
- Metadata file with naming format of `dataset_xx_metadata.csv`. This includes the isolate name, polymyxin susceptibility as a binary variable and the dataset that the isolate came from.
- Feature file with naming format of `dataset_xx_full.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix.

## Datasets with GWAS filtering
To run the pipeline using GWAS filtering, four files are needed:
- Metadata file with naming format of `dataset_xx_metadata.csv`. This includes the isolate name, polymyxin susceptibility as a binary variable and the dataset that the isolate came from.
- Training feature file with naming format of `dataset_xx_train.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix. You need to split your data into a training and testing set as this approach does not use cross-validation due to the risk of data leakage when the GWAS is performed.
- Testing feature file with naming format of `dataset_xx_test.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix. This is the data that you will test on
- GWAS file with naming format of `dataset_xx_gwas.csv`. This file needs to contain the feature names and *P* values from your GWAS. We used [treeWAS](https://github.com/caitiecollins/treeWAS) in the simultaneous mode in the manuscript.

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
Explanations for arguments can be found within the script, but -pr is the prefix of the dataset to use, -mp is the pickle file for the model dictionary, -dp is where the feature importance CSV file should be written.
`python extract_feature_importance.py -pr dataset_1_ \
-mp data/featureImportanceTestData/modelDictMultiDataRun_datasets_1_2_3_7_20190722.pkl \
-dp data/featureImportanceTestData/dataset_1_full.csv`

## Bootstrapping GWAS datasets
This script is kind of special in that all the files to use are hardcoded to run over all the GWAS datasets using specific pickle files.
`python run_gwas_data_bootstrap.py`
