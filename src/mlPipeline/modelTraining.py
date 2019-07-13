import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression, LassoCV, Lasso
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import balanced_accuracy_score
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import LassoCV
import numpy as np
import glob
import warnings
import sys
import csv
warnings.filterwarnings("ignore")

"""
Description:
Input:
Output:
TODO:
"""











class gwasFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, xLocusTags, gwasDF, gwasCutoff = 0.1):
        """
        Description: Performs GWAS feature selection which can be used in a sklearn pipeline.
        Input:
            xLocusTags (str): The names of locus tags in the feature columns. Thus each column in
                the X data should correspond to a correlation score and name in gwasDF
            gwasDF (pandas DF): A dataframe of xLocusTags and correlation score.
            gwasCutoff (float): The minimum absolute correlation score needed to be considered as
                a feature for models.
        Output:
            None: Results are printed.
        TODO:
        """
        self.gwasDF = gwasDF
        self.gwasCutoff = gwasCutoff
        self.xLocusTags = xLocusTags
        self.colsToUse = None
        # print("hey")
        # print(self.gwasPath)

    def transform(self, x):
        # print(len(self.colsToUse))
        cols = x[:,self.colsToUse]
        return(cols)
    def fit(self, x, y = None):
        if self.gwasDF is not None:
            if "corr_dat" in self.gwasDF.columns:
                # print("Using correlation as part of feature selection")
                # This is kept in when we're using correlation and want to use 
                # those features highly correlated with the outcome.
                corrVector = self.gwasDF.corr_dat.values
                gwasLocusTags = self.gwasDF.LOCUS_TAG.values
                candidateLoci = gwasLocusTags[np.where(corrVector >= self.gwasCutoff)]
            elif "p.vals" in self.gwasDF.columns:
                # print("Using pvalues as part of feature selection")
                # WHen using pvalues we look for pvalues less than the cutoff
                pvalVector = self.gwasDF['p.values'].values
                gwasLocusTags = self.gwasDF.LOCUS_TAG.values
                candidateLoci = gwasLocusTags[np.where(pvalVector <= self.gwasCutoff)]
            else:
                print("Unexpected gwas csv structure.", sys.exc_info()[0])
                raise
            self.colsToUse = [idx for idx, locusTag in enumerate(self.xLocusTags) if locusTag in candidateLoci]
        else:
            self.colsToUse = []
        return(self)



def printBestModelStatistics(gridCV, scoring, modelName):
    """
    Description: Prints information about the model specified in modelName.
    Input:
        gridCV (dict): A dictionary returned by sklearn's GridSearchCV function.
        scoring (list): A list of metrics in gridCV which are to be printed
        modelName (str): Name of model to be printed.
    Output:
        None: Results are printed.
    TODO:
    """
    scoringDict = {}
    bestModelIndex = gridCV.best_index_
    for score in scoring:
        scoringDict[score] = gridCV.cv_results_["mean_test_" + score][bestModelIndex]
        outStr = "For Model {}:".format(modelName)
    for scoreName, scoreVal in scoringDict.items():
        outStr += "\n\t{}: {}".format(scoreName, np.round(scoreVal, decimals = 3))
    print(outStr)  


def performGridSearch(dataPath, dataPrefix):
    """
    Description: Perfroms gridsearch over 4 sklearn models. Hyperparameters tuned include 
        model parameters as well as feature selection parameters. Feature selection is performed
        using GWAS correlation scores (if available) and linear SVC to rank feature importance
        and taking the top n features where n is a tunable parameter.
    Input:
        dataPath (str): Path to dataset folder.
        dataPrefix (str) prefix for the dataset in question. All relevant files for this datset
            share the same prefix.
    Output:
        modelDict (dict(dict(str))): A dictionary of grid search information for all models. The 
        inner keys are string names for the models, and the values correspond to dictionaries of
        grid search runs.
    TODO:
        1) 
    """

    # Read in the data
    nModels = 4
    relevantFiles = glob.glob(dataPath + dataPrefix + "*.csv")
    # print(relevantFiles)
    assertStr = "Unexpected item in bagging area. Have you unzipped the data file? Please refer to the README for information on running the ML notebook"
    assert len(relevantFiles) in [2,4], assertStr

    validationData = False
    GWASDF = None
    gwasCutOffList = [0.0]
    selectFromThreholds = ["mean*0.25", "mean", "mean*1.25"]
    # selectFromThreholds = ["mean", 0.25, 1e-5]
    if len(relevantFiles) == 2:# no testing data provided so use CV
        cv = 2
        trainPath = dataPath + dataPrefix + "full.csv"
        print("reading in data")
        isolateList = []
        with open(trainPath) as fp:
            csvReader = csv.reader(fp)
            header = next(csvReader)
        if len(header) > 200000:
            with open(trainPath) as fp:
                csvReader = csv.reader(fp)
                header = next(csvReader)
                for line in csvReader:
                    isolateList.append(line[0])
            print("isolates:", isolateList[0:10])
            allData = np.loadtxt(trainPath, delimiter = ",", skiprows = 1, usecols = range(1, len(header))) 
            allData = pd.DataFrame(allData, columns = header[1:])
            allData.insert(loc = 0, column = "isolate", value = isolateList) 
                # print(time.time() - start) 
                # allData = pd.read_csv(trainPath, dtype = myDtypes)
                # allData = np.loadtxt(trainPath, delimiter = ",", skiprows = 1,
                #     usecols = range(1, len(header))) 
        else:
            allData = pd.read_csv(trainPath)
        print("data read")

    else: # validation data provided. Should always have GWAS
        validationData = True
        trainPath = dataPath + dataPrefix + "train.csv"
        valPath = dataPath + dataPrefix + "test.csv"
        gwasPath = dataPath + dataPrefix + "gwas.csv"
        print("Using trainFile: {}\nvalFile: {}\ngwasFile: {}".format(trainPath,
                                                                      valPath,
                                                                      gwasPath))
        trainDF = pd.read_csv(trainPath)
        print(trainDF.shape)
        valDF = pd.read_csv(valPath)
        print(valDF.shape)
        allData = pd.concat([trainDF, valDF])
        GWASDF = pd.read_csv(gwasPath)
        GWASDF.corr_dat = np.abs(GWASDF.corr_dat)
        nTrain = trainDF.shape[0]
        nVal = valDF.shape[0]
        cv = lambda: zip([np.arange(nTrain)], [np.arange(nTrain, nTrain + nVal)])
        gwasCutOffList = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        gwasCutOffList = gwasCutOffList[gwasCutOffList <=
                                        np.max(GWASDF.corr_dat.values)]
        selectFromThreholds = ["mean*0.25", "mean", "mean*1.25", np.inf]
        secondMax = np.sort(GWASDF.corr_dat.values)[::-1][1]
        min = np.min(GWASDF.corr_dat.values)
        gwasCutOffList = np.arange(start = 0.0, step = 0.05, stop = secondMax)
        # print(gwasCutOffList)
        # print("maximum gwas corr: {}".format(np.sort(GWASDF.corr_dat.values)[::-1][1]))
        print("GWAS summary stats")
        print(GWASDF.corr_dat.agg(["mean", "max", "min"]))
        # return(GWASDF)
    # return(0)
    print("data read")
    print("splitting data")
    allData = allData.set_index("isolate")
    X_df = allData.drop(labels = ["pbr_res"], axis = 1)
    X = X_df.values
    Y_df = allData["pbr_res"]
    Y = Y_df.values 
    print("data split")
    # print(gwasCutOffList)
    # return(X_df)




    # df = pd.read_csv(dataPath)
    # df = df.set_index("isolate")
    # X_df = df.drop(labels = ["pbr_res"], axis = 1)
    # X = X_df.values
    # Y_df = df["pbr_res"]
    # Y = Y_df.values
    print("performing grid search")
    print("X shape: {}".format(X.shape))
    print("Y positive examples: {}".format(sum(Y)))
    if type(cv) is int:
        print("Using {} fold cv".format(cv))
    else:
        print("Training n = {}\nValidation n = {}".format(nTrain, nVal))
    # 1/0
    # feature selection
    features = []
    # features.append(('pca', PCA(n_components=3)))
    # features.append(('select_best', SelectKBest(k=6)))
    # Set a minimum threshold of 0.25
    # features.append(("linSVC_dimReduction", SelectFromModel(LinearSVC(C=1, penalty="l1", loss = 'l2', dual=True), 0.25)))
    # features.append(("linSVC_dimReduction", SelectFromModel(LinearSVC(C=0.5, penalty="l1",
    # 				loss = 'squared_hinge', dual=False), 0.25)))
    features.append(("linSVC_dimReduction", SelectFromModel(LinearSVC(C=1, penalty="l2",
    				loss = 'squared_hinge', dual=True))))# default settings
    features.append(("gwasFeatures", gwasFeatureExtractor(xLocusTags = list(X_df),
                                     gwasDF = GWASDF)))
    linearSVC_Cs = [100, 10, 1, 0.75, 0.25,]
    # loss='l2', penalty='l1', dual=False
    # features.append(("lasso_dimReduction", SelectFromModel(LassoCV(), 0.25)))
    feature_union = FeatureUnion(features)


    # Specify the models
    modelDict = {}
    # cv = 5
    n_jobs = 10
    # TODO when you perform CV with this stuff consider doing memory option stuff
    scoring = ["accuracy", "f1", "precision", "recall", "roc_auc", "balanced_accuracy"]
    importantMetric = "roc_auc"
    print("Choosing the best model based on {}".format(importantMetric))
    # print("Performing {} fold cv".format(cv))



    # create pipeline for logisttic Regression
    estimators_LR = []
    estimators_LR.append(('feature_union', feature_union))
    estimators_LR.append(('logistic', LogisticRegression()))
    # estimators.append(models)
    paramGrid_LR = [
        {
            "feature_union__gwasFeatures__gwasCutoff":gwasCutOffList,
        	"feature_union__linSVC_dimReduction__estimator__C":linearSVC_Cs,
        	"feature_union__linSVC_dimReduction__threshold": selectFromThreholds,
            "logistic__penalty": ['l1', 'l2'],
            "logistic__C": [1, 10, 100, 1000]
        }
    ]
    # print(paramGrid_LR)
    modelDict["logistic"] = {"pipe": Pipeline(estimators_LR),
                             "params": paramGrid_LR}
    modelDict["logistic"]["gridcv"] = GridSearchCV(estimator = modelDict["logistic"]["pipe"],
                             param_grid = modelDict["logistic"]["params"],
                             cv = cv if type(cv) is int else cv(),
                             n_jobs = n_jobs, return_train_score = False,
                             scoring = scoring, refit = importantMetric,
                             error_score = np.NaN)
    # error_score = -1 means a score of np.NaN is returned when fit doesn't work
    # TODO understand how linearSVC works with these parameters




    # create pipeline for RF
    estimators_RF = []
    estimators_RF.append(('feature_union', feature_union))
    estimators_RF.append(('RFC', RandomForestClassifier()))
    # estimators.append(models)
    paramGrid_RF = [
        {
            "feature_union__gwasFeatures__gwasCutoff":gwasCutOffList,
        	"feature_union__linSVC_dimReduction__estimator__C":linearSVC_Cs,
        	"feature_union__linSVC_dimReduction__threshold": selectFromThreholds,
            "RFC__n_estimators": [5, 10, 15, 20],# second most important feature to tune. First
            # is max number of feats.
            "RFC__max_features": ["sqrt", "log2", 0.5],# we have lots of possibly dumb
            # features so it might be good to use lower numbers here
            "RFC__max_depth": [None],# still need to understand if deeper trees are better.
            "RFC__criterion":["gini"],# no idea if this will make a difference. can check
        }
    ]

    modelDict["randomForest"] = {"pipe": Pipeline(estimators_RF),
                                 "params": paramGrid_RF}
    modelDict["randomForest"]["gridcv"] = GridSearchCV(estimator = modelDict["randomForest"]["pipe"],
                           param_grid = modelDict["randomForest"]["params"],
                           cv = cv if type(cv) is int else cv(), 
                           n_jobs = n_jobs, return_train_score = False,
                           scoring = scoring, refit = importantMetric,
                             error_score = np.NaN)
    #TODO is it better to build RF trees to purity and prune?





    # Create pipeline for SVC
    estimators_SVC = []
    estimators_SVC.append(('feature_union', feature_union))
    estimators_SVC.append(('SVC', SVC()))
    # estimators.append(models)
    paramGrid_SVC = [
        {
            "feature_union__gwasFeatures__gwasCutoff":gwasCutOffList,
        	"feature_union__linSVC_dimReduction__estimator__C":linearSVC_Cs,
        	"feature_union__linSVC_dimReduction__threshold": selectFromThreholds,
            "SVC__kernel": ['rbf', 'poly', "sigmoid"],
            "SVC__C": [0.5, 1, 10, 100, 1000]
        }
    ]
    modelDict["SVC"] = {"pipe":Pipeline(estimators_SVC),
                        "params": paramGrid_SVC}
    modelDict["SVC"]["gridcv"] = GridSearchCV(estimator = modelDict["SVC"]["pipe"],
                           param_grid = modelDict["SVC"]["params"],
                           cv = cv if type(cv) is int else cv(), 
                           n_jobs = n_jobs, return_train_score = False,
                           scoring = scoring, refit = importantMetric,
                             error_score = np.NaN)



    # create pipeline for GBTC
    estimators_GBTC = []
    estimators_GBTC.append(('feature_union', feature_union))
    estimators_GBTC.append(('GBTC', GradientBoostingClassifier()))
    # estimators.append(models)
    paramGrid_GBTC = [
        {
            "feature_union__gwasFeatures__gwasCutoff":gwasCutOffList,
        	"feature_union__linSVC_dimReduction__estimator__C":linearSVC_Cs,
        	"feature_union__linSVC_dimReduction__threshold": [0.25],
            "GBTC__learning_rate": [0.001, 0.01, 0.1],
            "GBTC__n_estimators": [50, 100, 200, 300, 400, 500],
            "GBTC__max_depth": [1, 3, 5, 10, 12]
        }
    ]
    modelDict["GBTC"] = {"pipe": Pipeline(estimators_GBTC),
                         "params": paramGrid_GBTC}
    modelDict["GBTC"]["gridcv"] = GridSearchCV(estimator = modelDict["GBTC"]["pipe"],
                           param_grid = modelDict["GBTC"]["params"],
                           cv = cv if type(cv) is int else cv(), 
                           n_jobs = n_jobs, return_train_score = False,
                           scoring = scoring, refit = importantMetric,
                             error_score = np.NaN,
                             verbose = 2)
    # modelDict["gradientBosting"] = Pipeline(estimators_GBTC)
    # return(modelDict, X, Y)
    print("Gwas Cutoff {}".format(gwasCutOffList))
    for modelName, currModelDict in modelDict.items():
        # if modelName != "GBTC":
        #     continue
        print("Training {}".format(modelName))
        currModelDict["gridcv"].fit(X,Y)
        printBestModelStatistics(gridCV = currModelDict["gridcv"],
                             scoring = scoring, modelName = modelName)
        currModelDict["refitMetric"] = importantMetric
        print("Best Model Parameters {}".format(currModelDict["gridcv"].best_params_))
        print("*"*100)
        # break
    return(modelDict)