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
        plt.bar(x = ind, height = scoreMeans, yerr = scoreStds,
               edgecolor='r', linewidth = barBorderWidths)
        plt.xticks(ind, xLabels, rotation = 0)
        plt.ylim((0,1))
        plt.title("Best Model Based on {} for Metric {}".format(refitMetric.upper(),
                                                                score.upper()))
    return(scoringDict)