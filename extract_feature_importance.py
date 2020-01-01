import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import csv
import os
import sys


def extract_feature_importance(modelName, allDataModelDict, dataSetOfInterest, dataPath,
		out_path):
	"""
	Description: Extracts feature importance for a given model name. The dataSetOfInterest
		and dataPath must match (this is not checked by the code), so be careful when entering 
		this information. In fact, it's important that allDataModelDict was actually trained on the
		dataSetOfInterest in the folder dataPath
	Input:
		modelName (str): String name for the model to access in allDataModelDict
		allDataModelDict (dict): A dict created by run_model_grid_search
		dataSetOfInterest (str): prefic for the dataset used. 
		dataPath (str): Path to the folder containing all datasets
		out_path (str): Where to write csv file of feature importance.
	Output:
		None. Information is all saved to a file.
	TODO:
		1)
	"""
	assert modelName in ["randomForest", "GBTC", "logistic", "SVC"], "invalid model name."
	##################################################
	#######################NOTE#######################
	##################################################
	# Be sure that the dataPath and dataSetOfInterest
	# correspond to the same dataset used to train the
	# models
	modelDict = allDataModelDict[dataSetOfInterest][modelName]
	bestModel = modelDict["gridcv"].best_estimator_
	featureTransformer = bestModel.steps[0][1]# 0th step is the feature extraction, and the first element
	# of that is the actual function for it. 



	isolateList = []
	with open(dataPath) as fp:
		csvReader = csv.reader(fp)
		header = next(csvReader)
	if len(header) > 100000:
		print("Reading large data with more efficient code")
		with open(dataPath) as fp:
			csvReader = csv.reader(fp)
			header = next(csvReader)
			for line in csvReader:
				isolateList.append(line[0])
		df = np.loadtxt(dataPath, delimiter = ",", skiprows = 1, usecols = range(1, len(header))) 
		df = pd.DataFrame(df, columns = header[1:])
		df.insert(loc = 0, column = "isolate", value = isolateList) 
	else:
		df = pd.read_csv(dataPath)



	df = df.set_index("isolate")
	X_df = df.drop(labels = ["pbr_res"], axis = 1)
	X = X_df.values
	allFeatureNames = np.array(list(X_df))
	chosenFeatures = allFeatureNames[[featureTransformer.transform(np.arange(X.shape[1]).reshape([1, X.shape[1]]))]]
	chosenFeatures = chosenFeatures[0]


	if modelName == "randomForest":
		# Random forest had a bit of a naming mishap where I switched from
		# RFC to randomForest
		bestModelParams = bestModel.get_params()["RFC"]
	else:
		bestModelParams = bestModel.get_params()[modelName]
		
		
	if modelName == "GBTC":
		print("feature importance for GBTC")
		featureImportance = bestModelParams.feature_importances_
	elif modelName == "randomForest":
		print("feature importance for random forest")

		featureImportance = bestModelParams.feature_importances_
	elif modelName == "logistic":
		print("Feature importance logistic")
		featureImportance = bestModelParams.coef_[0]
	elif modelName == "SVC":
		if bestModel["SVC"].kernel != "linear":
			print("no coefficients are available for SVC without a linear kernel\n",
				 "This SVC's kernel is {}".format(bestModel["SVC"].kernel))
			print("Breaking the code because we can't get feature importance here")
			return
	else:
		assert False, "Model name: {} not implemented for feature importance".format(modelName)

	featureImportanceDict = defaultdict(int) # features and coefficients line up
	# but because there can be repeated features for logistic regression this needs to be 
	# added together.
	for feat, imp in zip(chosenFeatures, featureImportance):
		featureImportanceDict[feat] += imp

	nNonZeroFeats = 0
	for _, imp in featureImportanceDict.items():
		if imp != 0.0:
			nNonZeroFeats += 1
	featureImportanceTupleList = sorted(featureImportanceDict.items(),
									   key=lambda p:np.abs(p[1]), reverse = True)# sorting list    


	print("{} features before feature selection".format(len(allFeatureNames)))
	print("{} features passed to this model".format(len(chosenFeatures))) # for logistic
	print("{} unique features passed to this model".format(len(set(chosenFeatures)))) # for logistic
	print("{} features given feature importance above 0\n".format(nNonZeroFeats))

	print("SANITY CHECK. This is the best model being used. Please be sure that",
		 "The parameters match up with the model you specified. Sometimes an old model",
		 "Can be used by accident if the kernel is not restarted\n\n",
		  bestModel[modelName] if modelName != 'randomForest' else bestModel["RFC"],
		 "\n\n\n\n")


	print("human readable feature importance")
	for feature, importanceMeasure in featureImportanceTupleList:
		if importanceMeasure != 0.0:
			print("Feature {} coefficient/importance {}\n".format(feature, importanceMeasure))


			
	print("Writing feature importance to a csv file")
	print("featureName,importanceMetric")
	with open(out_path, 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(["featureName","importanceMetric"])	
		for feature, importanceMeasure in featureImportanceTupleList:
			if importanceMeasure != 0.0:
				csv_writer.writerow([feature, importanceMeasure])



def main():
	modelDictPath = "data/featureImportanceTestData/modelDictMultiDataRun_datasets_1_2_3_7_20190722.pkl"
	with open(modelDictPath, "rb") as pklFile:
		allDataModelDict = pickle.load(pklFile)
	dataSetOfInterest = "dataset_1_"
	dataPath = "data/featureImportanceTestData/dataset_1_full.csv"
	feat_importance_save_prefix = os.path.dirname(modelDictPath)
	for model_name in ["randomForest", "GBTC", "logistic", "SVC"]:
		feat_importance_csv_path = os.path.join(feat_importance_save_prefix,
			"feature_importance_{}.csv".format(model_name))
		extract_feature_importance(modelName = model_name, allDataModelDict = allDataModelDict, 
			dataSetOfInterest = dataSetOfInterest, dataPath = dataPath,
			out_path = feat_importance_csv_path)	
if __name__ == '__main__':
	main()