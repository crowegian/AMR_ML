import numpy as np
import matplotlib.pyplot as plt
from src.mlPipeline.modelTraining import performGridSearch, printBestModelStatistics
from src.mlPipeline.plotting import plotBestModelComparison, plotDatasetModelComparison
import pandas as pd
import pickle
import copy
import pprint
import glob
import csv
import os
import argparse
pp = pprint.PrettyPrinter(indent=4)




def main(compMetric, out_path, model_path):
	"""
	Description: Extracts metrics for the best model on each dataset from a pickled dictionary
		saved during a run_model_grid_search run. 
	Input:
		compMetric (str): The metric to choose with which to choose the best model. One of
			f1, accuracy, precision, recall and roc_auc
		out_path (str): Directory to write the csv file to.
		model_path (str): Where to load the pickled output from run_model_grid_search.py
	Output:
		None. All information is written to a csv file.
	TODO
		1)
	"""



	allowable_metrics = ["f1", "accuracy", "precision", "recall", "roc_auc"]
	assert compMetric in allowable_metrics, "{} not in the allowable comparison metrics" \
		.format(allowable_metrics)

	if not os.path.isdir(out_path):
		os.mkdir(out_path)
	out_path = os.path.join(out_path, "{}_performance.csv".format(compMetric))

	with open(model_path, "rb") as pklFile:
		allDataModelDict = pickle.load(pklFile)



	dataSetComparisonList = ["dataset_1_", "dataset_2_", "dataset_3_", "dataset_4_", 
	"dataset_5_", "dataset_6_", "dataset_7_", "dataset_8_", 
	"dataset_9_", "dataset_10_", "dataset_11_", "dataset_12_", 
	"dataset_13_", "dataset_14_", ]
	datasetBestModelDict = plotDatasetModelComparison(allDataModelDict = allDataModelDict,
							   dataSetComparisonList = dataSetComparisonList,
								   compMetric = compMetric, removeZeroF1Score = True)
	pp.pprint(datasetBestModelDict)
	with open(out_path, "w") as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(["dataset","best_model",
			"best_{}_score".format(compMetric), "{}_standard_deviation".format(compMetric)])
		for dataset, best_model_perf in datasetBestModelDict.items():
			model_name = best_model_perf[0]
			best_perf_metric = best_model_perf[1]
			std = best_model_perf[2]
			csv_writer.writerow([dataset, model_name, best_perf_metric, std])

if __name__ == '__main__':
	# Script can be run like this.
	# python extract_performance_metrics.py -cm f1 -mp data/modelDictMultiDataRun.pkl \
	# -op data/model_performance/
	# compMetric = "f1"
	# out_path = "data/model_performance/"
	# model_path = "data/modelDictMultiDataRun.pkl"


	parser = argparse.ArgumentParser(description = "Creates a CSV file of model metrics")
	parser.add_argument('--compMetric', "-cm", type = str,
	                    help = "metric to compare models on. Must be one of f1, accuracy"\
	                    	", precision, recall, or roc_auc")
	parser.add_argument('--model_path', "-mp", type = str,
	                    help = "Where to load the model dictionary from.")
	parser.add_argument('--out_path', "-op", type = str,
	                    help = "What folder to write the csv results to.")
	args = parser.parse_args()


	main(compMetric = args.compMetric, out_path = args.out_path, model_path = args.model_path)