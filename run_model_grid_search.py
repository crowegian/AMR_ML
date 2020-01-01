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


def search_for_best_model(dataPath, n_folds, n_jobs, modelDictPicklePath, testing):
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
		testing (bool): Whether or not to run a smaller gridsearch for debugging purposes.
	Output:
		allDataModelDict (dict): A dictionary of grid search runs for each dataset.
	TODO:
		1)
	"""


	allDataModelDict = {}
	for i in range(1,15):
	    print(dataPath)
	    dataPrefix = "dataset_{}_".format(i)
	    print("Running model search on data {}".format(dataPrefix))
	    allDataModelDict[dataPrefix] = performGridSearch(dataPath = dataPath, dataPrefix = dataPrefix,
	                                                     n_folds = n_folds, n_jobs = n_jobs)
	    print("\n\n\n\n\n")

	# Save performance results.
	modelDictPicklePath = "data/modelDictMultiDataRun.pkl"
	with open(modelDictPicklePath, "wb") as pklFile:
	    pickle.dump(allDataModelDict, pklFile)
	return(allDataModelDict)
def main():
	dataPath = "data/MLData/pbr_ml_project_datasets_20180423/"
	# dataPath = "data/MLData/largeData/"
	modelDictPicklePath = "data/modelDictMultiDataRun_tempDelete.pkl"
	n_folds = 10
	n_jobs = 2
	testing = True
	if os.path.isfile(modelDictPicklePath):
		print("Found previous gridsearch run. Loading model dictionary from {}.".format(
				modelDictPicklePath))
		with open(modelDictPicklePath, "rb") as pklFile:
			allDataModelDict = pickle.load(pklFile)
	else:
		allDataModelDict = search_for_best_model(dataPath = dataPath, n_folds = n_folds,
			n_jobs = n_jobs, modelDictPicklePath = modelDictPicklePath, testing = testing)
	print('Entered main')



if __name__ == '__main__':
	main()