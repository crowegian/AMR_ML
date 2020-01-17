# Predicting phenotypic polymyxin resistance in *Klebsiella pneumoniae* through machine learning analysis of genomic data

There are two main components to the pipeline for predicting polymyxin resistance using machine learning. The first step is to convert raw genetic data into predictive features. We have done this using a reference-based approach that relies on variant calling and IS, as well as a reference-free approach that relies on k-mer detection. For details on the reference-based approach, please see Macesic et al. *Clin Infect Dis* 2019 Sep 12. (PMID 31513705). For the k-mer detection in the reference-free approach, we used the [DSK k-mer counter](https://github.com/GATB/dsk) set at n=31. Sample datasets that we used as inputs in the manuscript can be found in `data`, as well as a file describing the datasets `data/pbr_ml_final_datasets.xlsx`.

# Input datasets
All input datasets are CSV files with particular layouts. Please see the included sample datasets in `data` to understand the layout.

## Datasets without GWAS filtering
To run the pipeline without using GWAS filtering, two files are needed:
- Metadata file with naming format of `dataset_xx_metadata.csv`. This includes the isolate name, polymyxin susceptibility as a binary variable and the dataset that the isolate came from.
- Feature file with naming format of `dataset_xx_full.csv`. This is a a file where isolates are rows and features (either variants or k-mers) are columns. It should be a one-hot matrix.

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
