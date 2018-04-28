from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt





def plotBestModelComparison(modelDictAll, scoring = ["f1", "recall", "precision"],
                            comparisonSubset = {"randomForest": "Random\nForest",
                                                "logistic": "Logistic\nRegression",
                                                "GBTC": "Gradient Boosting\nTrees",
                                                "SVC":"Support Vector\nClassifier"}):
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
    bestModelName = ""
    bestPerformance = 0# this assumes we want to maximize performance
    f1Key = "mean_test_f1"
    modelStd = -1
    stdKey = resultKey.replace("mean_test_", "std_test_")
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
    return(bestModelName, bestPerformance, modelStd)



from collections import defaultdict

def findBestModelPerDataset(allDataModelDict, dataSetComparisonList, compMetric, removeZeroF1Score):
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
    datasetBestModelDict = findBestModelPerDataset(allDataModelDict, dataSetComparisonList,
                                                   compMetric, removeZeroF1Score)
    modelIdx = 0
    perfIdx = 1
    stdidx = 2
    if removeZeroScores:
        print("Dataset with all zero f1 scores are being removed from plots")
        dataSetList = [dset for dset in list(datasetBestModelDict.keys())
                       if datasetBestModelDict[dset][modelIdx] != ""]

    # dataSetList.sort()
    perfList = [datasetBestModelDict[dset][perfIdx] for dset in dataSetList]
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
    else:
        figSize = np.array([9,6])*0.75
    fig=plt.figure(figsize=figSize, dpi= 300, facecolor='w', edgecolor='k')
    barBorderWidths = np.array([0]*len(ind))
    barBorderWidths[np.where(np.array(stdList) == 0)] = 1.5
    plt.bar(left = ind, height = perfList, yerr = stdList,
           edgecolor='#763626', linewidth = barBorderWidths, color = "#336B87")
    xLbls = [dset.replace("_", " ").strip() for dset in dataSetList]
    xLbls = [lbl + "\n" + mdlName for mdlName, lbl in zip(modelList, xLbls)]
    plt.xticks(ind, xLbls, rotation = 45, ha = "center")
    plt.ylim((0,1))
    title = plt.title("Dataset Performance Comparison")
    return(datasetBestModelDict)