from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt





def plotBestModelComparison(modelDictAll, scoring = ["f1", "recall", "precision"],
                            comparisonSubset = {"randomForest": "Random\nForest",
                                                "logistic": "Logistic\nRegression",
                                                "GBTC": "Gradient Boosting\nTrees",
                                                "SVC":"Support Vector\nClassifier"}):
    """
    Description: Collects metrics about a set of models for a given dataset and returns
        a dictionary for scoring metrics collected during training and validation.
    Input:
        modelDictAll (dict): A dictionary mapping model names to GridSearchCV results.
        scoring (list): A list of metrics to be looked at. Need to be a subset of the metrics
            using during model training.
        comparisonSubset (dict): A dictionary where keys represent the models to be
            investigated and valeues represent the cleaner names to be pritned.
    Output:
        scoringDict (dict(dict(str))) A dictionary mapping model names to various metrics.
    TODO:
    """
    assert set(comparisonSubset.keys()).issubset(set(modelDictAll.keys())),\
    "Comparison set is not a subset of the model set"
    scoringDict = defaultdict(lambda: defaultdict(float))
#     scoreMeans = defaultdict(list)
#     scoreStds = defaultdict(list)
    refitMetric = ""
    for modelName, modelDict in modelDictAll.items():
        refitMetric = modelDict["refitMetric"]
        currModelBestIdx = modelDict["gridcv"].best_index_
        for score in scoring:
            scoringDict[modelName][score + "_mean"] = modelDict["gridcv"].cv_results_["mean_test_" + score][currModelBestIdx]
            scoringDict[modelName][score + "_std"] = modelDict["gridcv"].cv_results_["std_test_" + score][currModelBestIdx]
    for score in scoring:
        scoreMeans = []
        scoreStds = []
        xLabels = []
        for modelName in list(comparisonSubset.keys()):
            scoreMeans.append(scoringDict[modelName][score + "_mean"])
            scoreStds.append(scoringDict[modelName][score + "_std"])
            xLabels.append(comparisonSubset[modelName])
        print(score)
        print(np.round(scoreMeans, decimals = 3))
        print(np.round(scoreStds, decimals = 3))
        ind = np.arange(len(scoreMeans))
        fig=plt.figure(figsize=(9, 6), dpi= 300, facecolor='w', edgecolor='k')
        barBorderWidths = [0]*len(ind)
        barBorderWidths[np.argmax(scoreMeans)] = 1
        plt.bar(left = ind, height = scoreMeans, yerr = scoreStds,
               edgecolor='r', linewidth = barBorderWidths)
        plt.xticks(ind, xLabels, rotation = 0)
        plt.ylim((0,1))
        plt.title("Best Model Based on {} for Metric {}".format(refitMetric.upper(),
                                                                score.upper()))
    return(scoringDict)















def findBestModel(modelList, resultKey, removeZeroF1Score):
    """
    Description: Iterates through a dictionary of models to find the best performing
        model on the resultKey.
    Input:
        modelList (dict): A poorly names dictionary which maps model names to gridSearchCV
            results.
        resultKey (str): The metric for which the best model should be chosen on. Needs to be
            a metric the gridSearchCV calculated beforehand.
        removeZeroF1Scores (bool): A boolean of whether or not to remove models which have
            zero F1 scores but decent performance in other scores. This can happen for metrics like
            AUROC where the FPR is high enough to give good AUROC but F1 shows that performance
            is actually terrible.
    Output:
        bestModelName (str): Name of the best model for a given dataset's gridSearch CV 
        bestPerformance (float): The highest value acheived for resultKey
        modelStd (float): Standard deviation of resultKey for the best model
        bestF1Performance (float): Best F1 performance
        f1Std (float): Std of F1 scores.
    TODO:
    """
    bestModelName = ""
    bestPerformance = 0# this assumes we want to maximize performance
    bestF1Performance = 0
    f1Key = "mean_test_f1"
    modelStd = -1
    f1Std = -1
    stdKey = resultKey.replace("mean_test_", "std_test_")
    f1StdKey = f1Key.replace("mean_test_", "std_test_")
    for modelName, cvDict in modelList.items():
    #     print(modelName)
        currModelIndex = cvDict["gridcv"].best_index_
        currModelPerformance = cvDict["gridcv"].cv_results_[resultKey][currModelIndex]
        if currModelPerformance >= bestPerformance:
#             if removeZeroF1Score and 
            f1Score = cvDict["gridcv"].cv_results_[f1Key][currModelIndex]
#             print(f1Score)
            if removeZeroF1Score and f1Score == 0.0:
                print("0.0 f1 score found. Model {} is not counted".format(modelName))
                continue
            bestModelName = modelName
            bestPerformance = currModelPerformance
            modelStd = cvDict["gridcv"].cv_results_[stdKey][currModelIndex]
            bestF1Performance = cvDict["gridcv"].cv_results_[f1Key][currModelIndex]
            f1Std = cvDict["gridcv"].cv_results_[f1StdKey][currModelIndex]
    return(bestModelName, bestPerformance, modelStd, bestF1Performance, f1Std)




def findBestModelPerDataset(allDataModelDict, dataSetComparisonList, compMetric, removeZeroF1Score):
    """
    Description: Iterates through all dataset performances and finds the best model for each
        performance
    Input:
        allDataModelDict (dict(dict(str))): Maps dataset names to their dictionary of grid search
            CV performances.
        dataSetComparisonList (list): List of names of datasets to be compared.
        compMetric (str): Metric for which to compare datasets on.
        removeZeroF1Score (boolnea): A boolean of whether or not to remove models which have
            zero F1 scores but decent performance in other scores. This can happen for metrics like
            AUROC where the FPR is high enough to give good AUROC but F1 shows that performance
            is actually terrible.
    Output:
        datasetBestModelDict (dit(list)) A dictionary mapping dataset names to the best model
            information. See findBestModel for specifics of the values.
    TODO:
    """
#     allDataModelDict = modelDictHolder
#     dataSetComparisonList = list(allDataModelDict.keys())
#     compMetric = "roc_auc"
    resultKey = "mean_test_" + compMetric
#     dataSetComparisonList
    datasetBestModelDict = {}
    for datasetName in dataSetComparisonList:
    #     print(datasetName)
        modelList = allDataModelDict[datasetName]
        dataSetBestPerformance  = findBestModel(modelList, resultKey, removeZeroF1Score)
        datasetBestModelDict[datasetName] = dataSetBestPerformance
    return(datasetBestModelDict)






def plotDatasetModelComparison(allDataModelDict, dataSetComparisonList,
                               compMetric, removeZeroF1Score = True, removeZeroScores = True):
    """
    Description: Plots comparisons of the best model for each dataset on the metric compMetric
    Input:
        allDataModelDict (dict(dict(str))): Maps dataset names to their dictionary of grid search
                CV performances.
        dataSetComparisonList (list): List of names of datasets to be compared.
        compMetric (str): Metric for which to compare datasets on.
        removeZeroF1Score (boolnea): A boolean of whether or not to remove models which have
            zero F1 scores but decent performance in other scores. This can happen for metrics like
            AUROC where the FPR is high enough to give good AUROC but F1 shows that performance
            is actually terrible.
        removeZeroScores (boolean): Models with zero scores for the given compMetric are removed.
    Output:
        datasetBestModelDict (dicr): See findBestModelPerDataset for dictionary information.
    TODO:
    """
    datasetBestModelDict = findBestModelPerDataset(allDataModelDict, dataSetComparisonList,
                                                   compMetric, removeZeroF1Score)
    modelIdx = 0
    perfIdx = 1
    stdidx = 2
    if removeZeroScores:
        print("Dataset with all zero f1 scores are being removed from plots")
        dataSetList = [dset for dset in dataSetComparisonList
                       if datasetBestModelDict[dset][modelIdx] != ""]

    # dataSetList.sort()
    # print(dataSetComparisonList)
    # print(dataSetList)
    perfList = [datasetBestModelDict[dset][perfIdx] for dset in dataSetList]
    # print(perfList)
    stdList = [datasetBestModelDict[dset][stdidx] for dset in dataSetList]
    modelList = [datasetBestModelDict[dset][modelIdx] for dset in dataSetList]
    modelNameMapper = {"GBTC": "GBTC",
                       "randomForest": "RF",
                       "logistic": "LG",
                       "SVC": "SVC"}
    modelList = [modelNameMapper[mdl] for mdl in modelList]
    ind = np.arange(len(modelList))
    if len(dataSetComparisonList) > 8:
        figSize = (12,6)
    elif len(dataSetComparisonList) >= 4:
        figSize = np.array([9,6])*1.0
    else:
        figSize = np.array([9,6])*0.75
    fig=plt.figure(figsize=figSize, dpi= 300, facecolor='w', edgecolor='k')
    barBorderWidths = np.array([0]*len(ind))
    barBorderWidths[np.where(np.array(stdList) == 0)] = 1.5
    x = np.arange(0, len(modelList))
    plt.bar(x = x, height = perfList, yerr = stdList,
           edgecolor='#763626', linewidth = barBorderWidths, color = "#336B87")
    xLbls = [dset.replace("_", " ").strip() for dset in dataSetList]
    xLbls = [lbl + "\n" + mdlName for mdlName, lbl in zip(modelList, xLbls)]
    plt.xticks(ind, xLbls, rotation = 45, ha = "center")
    plt.ylim((0,1))
    title = plt.title("Dataset Performance Comparison {}".format(compMetric.replace("_", " ")))
    return(datasetBestModelDict)