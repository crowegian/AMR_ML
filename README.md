# Predicting Antimicrobial Resistance

There are two main components to the pipeline for predicting antimicrobial resitance (AMR) using machine learning. The first step is to convert raw genetic data into predictive features using GWAS analysis and mutation annotation software. This code can be found in `src/featureEngineering` along with a README.md file which explains how to use the code. The second step in the pipeline is to perform feature selection, model training, and model evaluation in order to select the best model.The code for this step can be found in `src/mlPipeline/modelTraining.py` and to run this file please use the jupyternotebook in the main directory titled `ML_notebook.ipynb`


