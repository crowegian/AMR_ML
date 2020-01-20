import os
import numpy as np
import matplotlib.pyplot as plt
from src.mlPipeline.modelTraining import performGridSearch, printBestModelStatistics
from src.mlPipeline.plotting import plotBestModelComparison, plotDatasetModelComparison
import pandas as pd
import pickle
import copy
import pprint
import glob
pp = pprint.PrettyPrinter(indent=4)
import argparse


def search_for_best_model(dataPath, n_folds, n_jobs, modelDictPicklePath, datasets,  testing):
	"""
	Description: Calls the performGridSearch function and fills the allDataModelDict object
		with model runs and performance metrics for multiple datasets. A new grid search is 
		performed for each new dataset. This code can take hours to run through all 14 datasets
		and should be run over night.
	Input: 
		dataPath (str): The path to the folder where data is held using standard prefixes
		n_folds (int): The number of folds to do for CV when able to do so.
		n_jobs (int): The number of cores to use during grid search and model training
		modelDictPicklePath (str): Where to save the model dictionary with information on 
			gridsearch metrics for all datasets
		datasets (list): A list of datasets indices to look at.
		testing (bool): Whether or not to run a smaller gridsearch for debugging purposes.
	Output:
		allDataModelDict (dict): A dictionary of grid search runs for each dataset.
	TODO:
		1)
	"""


	allDataModelDict = {}
	for i in datasets:
	    print(dataPath)
	    dataPrefix = "dataset_{}_".format(i)
	    print("Running model search on data {}".format(dataPrefix))
	    allDataModelDict[dataPrefix] = performGridSearch(dataPath = dataPath, dataPrefix = dataPrefix,
	                                                     n_folds = n_folds, n_jobs = n_jobs,
	                                                     testing = testing)
	    print("\n\n\n\n\n")

	# Save performance results.
	# modelDictPicklePath = "data/modelDictMultiDataRun.pkl"
	with open(modelDictPicklePath, "wb") as pklFile:
	    pickle.dump(allDataModelDict, pklFile)
	return(allDataModelDict)
def main(dataPath, n_jobs, n_folds, modelDictPicklePath, testing, datasets):
	# dataPath = "data/MLData/pbr_ml_project_datasets_20180423/"
	# dataPath = "data/MLData/largeData/"
	# modelDictPicklePath = "data/modelDictMultiDataRun_tempDelete.pkl"
	if os.path.isfile(modelDictPicklePath):
		print("Found previous gridsearch run. Not running so we don't overwrite the file {}."\
			.format(modelDictPicklePath))
	else:
		allDataModelDict = search_for_best_model(dataPath = dataPath, n_folds = n_folds,
			n_jobs = n_jobs, modelDictPicklePath = modelDictPicklePath, datasets = datasets,
			testing = testing)
		with open(modelDictPicklePath, "wb") as pklFile:
			pickle.dump(allDataModelDict, pklFile)


if __name__ == '__main__':
	# script can be run with
	# python run_model_grid_search.py -dp data/MLData/pbr_ml_project_datasets_20180423/ \
	# -mp data/modelDictMultiDataRun.pkl -nj 2 -nf 10 -idx 1 2 10 --testing
	parser = argparse.ArgumentParser(description="Runs model grid search over multiple"\
		" models and datasets")
	parser.add_argument('--data_path', "-dp", type = str,
	                    help = "Folder containing all the data to perform gridsearch over")
	parser.add_argument('--model_pickle', "-mp", type = str,
	                    help = "Where to pickle the grid search dict to")
	parser.add_argument("--n_jobs", "-nj", type = int,
						help = "The number of cores to use")
	parser.add_argument("--n_folds", "-nf",  type = int,
						help = "The number of folds to perform for cv")
	parser.add_argument("--dataset_indices", "-idx", type = int,
						nargs = "+",
						help = "The dataset numbers to run eg `1 2 5 20`")
	parser.add_argument("--testing", action='store_true',
		help = "If added then the code will be run in test mode")
	args = parser.parse_args()

	main(dataPath = args.data_path, n_jobs = args.n_jobs, n_folds = args.n_folds, testing = args.testing,
		datasets = args.dataset_indices, modelDictPicklePath = args.model_pickle)