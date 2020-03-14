import pandas as pd
from sklearn.metrics import precision_score, recall_score,\
	f1_score, balanced_accuracy_score, roc_auc_score, accuracy_score
import pickle
# from src.mlPipeline.plotting import findBestModelPerDataset
import glob
import copy
from collections import defaultdict
import pprint
pp = pprint.PrettyPrinter(indent=4)
import argparse



def main(model_dict_path, model_dict_data_prefix, model_name, out_path, data_path):
	with open(model_dict_path, "rb") as pklFile:
		model_dict = pickle.load(pklFile)# assumes this is an sklearn estimator, or it's
		# a dictionary that needs to be indexed
		if model_dict_data_prefix is not None:
			assert model_name is not None, "Need model name"
			model = model_dict[model_dict_data_prefix][model_name]["gridcv"].best_estimator_	
		else:
			model = model_dict
		# should be an entire data path to a csv file. Will it need to have the labels?
	data_df = pd.read_csv(data_path)
	data_df = data_df.set_index("isolate")
	if "pbr_res" in data_df.columns:
		X = data_df.drop(labels = ["pbr_res"], axis = 1).values
		Y = data_df["pbr_res"].values
	else:
		X = data_df.values
		Y = None

	preds = model.predict(X)
	if Y is not None:
		performance_dict = {}
		if model_name == "SVC" and not model.steps[1][1].probability:
			scores = None
			rocauc = None
		else:
			scores = model.predict_proba(X)
			rocauc = roc_auc_score(y_true = Y, y_score = scores[:,1])		
			f1 = f1_score(y_true = Y, y_pred = preds)
			prec = precision_score(y_true = Y, y_pred = preds)
			rec = recall_score(y_true = Y, y_pred = preds)        
			bal_acc = balanced_accuracy_score(y_true = Y, y_pred = preds)
			acc = accuracy_score(y_true = Y, y_pred = preds)
			performance_dict["f1"] = f1
			performance_dict["prec"] = prec
			performance_dict["rec"] = rec
			performance_dict["balanced_acc"] = bal_acc
			performance_dict["acc"] = acc
			performance_dict["rocauc"] = rocauc
	pp.pprint(performance_dict)
	pred_df = pd.DataFrame(preds, columns = ["pbr_res_pred"], index = data_df.index)
	pred_df.to_csv(out_path)
	print(pred_df)

if __name__ == '__main__':
	# to run from command line, use the following command without '#'
	# python run_prediction.py -dp data/oli_gwas_cross_validation/dataset_17_test.csv \
	# -mp data/oli_gwas_cross_validation/modelDictMultiDataRun_dataset_17_18_20190826.pkl \
	# -pr dataset_17_ \
	# -mn SVC \
	# -op ./predictions.csv
	parser = argparse.ArgumentParser(description="Runs model grid search over multiple"\
		" models and datasets")
	parser.add_argument('--data_path', "-dp", type = str,
	                    help = "Folder containing all the data to perform gridsearch over")
	parser.add_argument('--model_pickle', "-mp", type = str,
	                    help = "Where to pickle the grid search dict to")
	parser.add_argument("--model_dict_data_prefix", "-pr", type = str,
						default = None,
						help = "Prefix for the model dictionary.")
	parser.add_argument("--model_name", "-mn",  type = str,
						default = None,
						help = "Model name to be used to index into dictionry")
	parser.add_argument("--out_path", "-op", type = str,
						help = "Path to save predictions to")
	args = parser.parse_args()
	main(
		model_dict_path = args.model_pickle,
		model_dict_data_prefix = args.model_dict_data_prefix,
		model_name = args.model_name,
		out_path = args.out_path,
		data_path = args.data_path,
		)