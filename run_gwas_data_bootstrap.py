import pandas as pd
from sklearn.metrics import precision_score, recall_score,\
	f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
import pickle
from src.mlPipeline.plotting import findBestModelPerDataset
import glob
import copy
from collections import defaultdict
import pprint
pp = pprint.PrettyPrinter(indent=4)




def read_in_data(data_path):
	"""
	Description: Reads in the train, validation, and if the gwas data for a given
		data_path prefix.
	Input:
		data_path (str): Data path prefix.
	Output:
		trainDF, valDF (pandas df): Datframes of the training and validation sets read from csv
		GWASDF (pandas df): Dataframe of gwas data read in from a csv file.

	"""
	trainPath = data_path  + "_train.csv"
	valPath = data_path  + "_test.csv"
	gwasPath = data_path  + "_gwas.csv"
	trainDF = pd.read_csv(trainPath)
	valDF = pd.read_csv(valPath)
	allData = pd.concat([trainDF, valDF])
	GWASDF = pd.read_csv(gwasPath)
	nTrain = trainDF.shape[0]
	nVal = valDF.shape[0]
	trainDF = trainDF.set_index("isolate")
	valDF = valDF.set_index("isolate")
	return(trainDF, valDF, GWASDF)


def main():
	model_pkl_path_17_18 = "data/oli_gwas_cross_validation/"\
		"modelDictMultiDataRun_dataset_17_18_20190826.pkl"
	with open(model_pkl_path_17_18, "rb") as pklFile:
		model_dict_17_18 = pickle.load(pklFile)
	model_pkl_path_19_20_21 = "data/oli_gwas_cross_validation/"\
		"modelDictMultiDataRun_dataset_19_20_21_20190826.pkl"
	with open(model_pkl_path_19_20_21, "rb") as pklFile:
		model_dict_19_20_21 = pickle.load(pklFile)



	model_pkl_path_25 = "data/bootstrap_datasets_25_26/modelDictMultiDataRun_dataset_25.pkl"
	with open(model_pkl_path_25, "rb") as pklFile:
		model_dict_25 = pickle.load(pklFile)

	model_pkl_path_26 = "data/bootstrap_datasets_25_26/modelDictMultiDataRun_dataset_26.pkl"
	with open(model_pkl_path_26, "rb") as pklFile:
		model_dict_26 = pickle.load(pklFile)



	dataset_prefix_list_1 = ['dataset_17_', 'dataset_18_']
	dataset_prefix_list_2 = ['dataset_19_', 'dataset_20_', 'dataset_21_']

	dataset_prefix_list_3 = ['dataset_25_', 'dataset_26_']




	dataset_model_bootstrap_performance_dict = defaultdict(dict)

	for dataset_prefix_list in [dataset_prefix_list_1, dataset_prefix_list_2]:
		for dataset_prefix in dataset_prefix_list:
			if dataset_prefix in dataset_prefix_list_1:
				model_dict_all = model_dict_17_18
				model_dict_id = "17-18"
			elif dataset_prefix in dataset_prefix_list_2:
				model_dict_all = model_dict_19_20_21
				model_dict_id = "19-20-21"
			elif dataset_prefix == "dataset_25_":
				model_dict_all = model_dict_25
				model_dict_id = "25"
			elif dataset_prefix == "dataset_26_":
				model_dict_all = model_dict_26
				model_dict_id = "26"
			else:
				assert False, "invalid dataset prefix"
			print("running dataset: {}, with model_dict for datasets: {}".format(dataset_prefix, model_dict_id))
			for model_name, model_dict in model_dict_all[dataset_prefix].items():
				best_model = copy.deepcopy(model_dict["gridcv"].best_estimator_)
				for data_idx in range(0,10):
					if model_dict_id in ["25", "26"]:
						data_path = "data/bootstrap_datasets_25_26/"
					else:
						data_path = "data/bootstrap_datasets_25_26/"	
					data_path = data_path + \
						dataset_prefix[:-1] + ".{}".format(data_idx)
					# Read in all data
					trainDF, valDF, gwasDF = read_in_data(data_path)
					# Split training a testing matrices
					X_train = trainDF.drop(labels = ["pbr_res"], axis = 1).values
					Y_train = trainDF["pbr_res"].values
					X_val = valDF.drop(labels = ["pbr_res"], axis = 1).values
					Y_val = valDF["pbr_res"].values
					# Refit model
					best_model.fit(X_train, Y_train)
					# Get metrics
					preds = best_model.predict(X_val)
					if model_name == "SVC" and not best_model.steps[1][1].probability:
						scores = None
						rocauc = None
					else:
						scores = best_model.predict_proba(X_val)
						rocauc = roc_auc_score(y_true = Y_val, y_score = scores[:,1])
					f1 = f1_score(y_true = Y_val, y_pred = preds)
					prec = precision_score(y_true = Y_val, y_pred = preds)
					rec = recall_score(y_true = Y_val, y_pred = preds)        
					bal_acc = balanced_accuracy_score(y_true = Y_val, y_pred = preds)
					acc = accuracy_score(y_true = Y_val, y_pred = preds)
					dataset_model_bootstrap_performance_dict\
						[dataset_prefix + model_name]["f1_{}".format(data_idx)] = f1
					dataset_model_bootstrap_performance_dict\
						[dataset_prefix + model_name]["prec_{}".format(data_idx)] = prec
					dataset_model_bootstrap_performance_dict\
						[dataset_prefix + model_name]["rec_{}".format(data_idx)] = rec
					dataset_model_bootstrap_performance_dict\
						[dataset_prefix + model_name]["balanced_acc_{}".format(data_idx)] = bal_acc
					dataset_model_bootstrap_performance_dict\
						[dataset_prefix + model_name]["acc_{}".format(data_idx)] = acc
					dataset_model_bootstrap_performance_dict\
						[dataset_prefix + model_name]["rocauc_{}".format(data_idx)] = rocauc

	out_path = "data/oli_gwas_cross_validation/bootstrap_results.csv"
	results_df = pd.DataFrame.from_dict(dataset_model_bootstrap_performance_dict, orient = "index")
	results_df.to_csv(out_path)


	pp.pprint(dataset_model_bootstrap_performance_dict)










if __name__ == '__main__':
	main()