
# new todo:
# create a few heatmaps:
# correlation between each streamflow metric and each predictor
# for each size
    # correlation between each streamflow metric and each predictor
# heatmap of the changes in correlation
# p values associated with the change in correlation
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from scipy import stats
from sklearn import linear_model
from scipy.stats.mstats import theilslopes as tslope
from ColorCatchments import *
from tqdm import tqdm
import copy
import seaborn as sns
from matplotlib.patches import Rectangle

'''
The goal of this script is to quantify and plot the relationships 
between predictor variables (precipitation, temperature, and stream order)
and streamflow qualities.

We will also seek to quantify how those relationships change as stream
order increases.
'''

numSplits = 3 #int(sys.argv[1])
variables = ["MeanTempAnn","MeanPrecAnn","gord"]
metrics = ["domf_mean", "masd_mean","spectral_mean"]

df = pd.read_csv("mergedData.csv")
# normalize the data
df = (df - df.mean()) / df.std()

# prepare info about splitting the data
splitVals = list(df[df.columns[0]])
interval = len(splitVals) // numSplits

startIndices = []
stopIndices = []

start = 0
stop = interval
for split in range(numSplits):
    startIndices.append(start)
    stopIndices.append(stop)
    start = start + interval
    stop = stop + interval



varToTitle = {
        "masd_mean":"mean MASD",
        "masd_slope":"slope of MASD",
        "masd_slope_normalized":"% change in MASD",
        "domf_mean":"mean DOMF",
        "domf_slope":"slope of DOMF",
        "domf_slope_normalized":"% change in DOMF",
        "spectral_mean":"mean spectral",
        "spectral_slope":"slope of spectral",
        "spectral_slope_normalized":"% change in spectral",
        "spectral_full":"full spectral",
        "MeanTempAnn":"Mean Annual Temp.",
        "MeanPrecAnn":"Mean Annual Precip.",
        "gord":"stream order"
        }


varToTitleShort = {
        "MeanTempAnn":"temp.",
        "MeanPrecAnn":"precip.",
        "gord":"stream order"
        }


root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/"
resultsDir = os.path.join(root, "CorrelationResults")

if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)


def getChangeInCorrelation(xs, ys):

    xs1 = xs[startIndices[0]:stopIndices[0],:]
    ys1 = ys[startIndices[0]:stopIndices[0]]

    xs2 = xs[startIndices[-1]:stopIndices[-1],:]
    ys2 = ys[startIndices[-1]:stopIndices[-1]]

    clf1 = linear_model.Lasso(alpha=0.1)
    clf1.fit(xs1, ys1)
    c1s = clf1.coef_

    clf2 = linear_model.Lasso(alpha=0.1)
    clf2.fit(xs2, ys2)
    c2s = clf2.coef_

    diffs = c1s - c2s
    return diffs, c1s, c2s


def getPermutedDistribution(xVars, yVar, numRepeats=100000):
    xs = copy.copy(df[xVars].to_numpy())
    ys = copy.copy(df[yVar].to_numpy())
    
    changesInCorrelation = np.zeros((numRepeats, xs.shape[-1]))
    firsts = np.zeros((numRepeats,xs.shape[-1]))
    seconds = np.zeros((numRepeats,xs.shape[-1]))

    loop = tqdm(total=numRepeats, position=0)
    for i in range(numRepeats):
        np.random.shuffle(ys) 
        np.random.shuffle(xs) 
        changesInCorrelation[i], first, second = getChangeInCorrelation(xs, ys)        
        firsts[i] = first
        seconds[i] = second
        loop.update()
    loop.close()

    return changesInCorrelation
   

def roundArrayTo(array, numDigits):
    coefficient = 10 ** numDigits
    roundedArray = np.round(array * coefficient) / coefficient
    return roundedArray

def getPValue(changesInCorrelation, changeInCorrelationReal):
    

    changesInCorrelation = np.array(copy.copy(changesInCorrelation)) + 2 # -2 = the maximum possible negative change
    changeInCorrelationReal = np.array(copy.copy(changeInCorrelationReal)) + 2 # this makes them all positive

    numGreater = np.sum(changesInCorrelation <  changeInCorrelationReal)
    numLess    = np.sum(changesInCorrelation >= changeInCorrelationReal)

    numMoreExtreme = np.min([numGreater, numLess])
    numMoreExtreme = numMoreExtreme * 2 # two-tailed

    pValue = numMoreExtreme / np.max(changesInCorrelation.shape)

    return pValue


def matrixToDfChange(array1, array2, axis1Labels, axis2Labels, splitVar):
    doubledAxis1Labels = []
    doubledAxis2Labels = []

    for axis in axis2Labels: # intentionally switch the order
        doubledAxis2Labels.append(axis)
        doubledAxis2Labels.append(axis)

    dataDict = {}
    for index, axis2Label in enumerate(doubledAxis2Labels):
        if index % 2 == 0:
            dataDict[varToTitle[axis2Label] + "\n (low " + varToTitleShort[splitVar] + ")"] = array1[:,int(np.floor(index / 2))]
        if index % 2 == 1:
            dataDict[varToTitle[axis2Label] + "\n (high " + varToTitleShort[splitVar] + ")"] = array2[:,int(np.floor(index / 2))]

    newDf = pd.DataFrame.from_dict(dataDict)
    newAxis2Labels = [varToTitle[var2] for var2 in axis1Labels]
    newDf.index = newAxis2Labels
    return newDf


def getMeanSplitVar(ldf, numDigits=5):
    return np.round(np.mean(ldf[splitVar]) * numDigits) / numDigits
    

def getChangesDf(dfs):

    newMets = [varToTitle[met] for met in metrics]
   
    correlations = np.zeros((len(metrics), len(variables) * 2))
    pValues = np.zeros((len(metrics), len(variables)))
    scores = np.zeros((len(metrics),2))

    ldf1 = dfs[0]
    ldf2 = dfs[-1]

    newIndex = []
    for index1 in range(len(metrics)):
        metric = metrics[index1]
        newIndex.append(varToTitle[metric])

        xs1 = ldf1[variables].to_numpy()
        ys1 = ldf1[metric].to_numpy()
        clf1 = linear_model.Lasso(alpha=0.1)
        clf1.fit(xs1, ys1)
        c1s = clf1.coef_
        s1 = clf1.score(xs1, ys1)

        xs2 = ldf2[variables].to_numpy()
        ys2 = ldf2[metric].to_numpy()
        clf2 = linear_model.Lasso(alpha=0.1)
        clf2.fit(xs2, ys2)
        c2s = clf2.coef_
        s2 = clf2.score(xs2, ys2)

        newRow = np.zeros(len(variables) * 2) 
        newCols = []
        newColsPvals = []
        for index2 in range(len(variables)):
            i1 = index2 * 2
            i2 = i1 + 1
            newRow[i1] = c1s[index2]
            newRow[i2] = c2s[index2]

            newCols.append(varToTitle[variables[index2]] + "\n (low " + varToTitleShort[splitVar] + ")")
            newCols.append(varToTitle[variables[index2]] + "\n (high " + varToTitleShort[splitVar] + ")")
            
            newColsPvals.append(varToTitle[variables[index2]])

        newColsScores = []
        newColsScores.append("low " + varToTitleShort[splitVar])
        newColsScores.append("high " + varToTitleShort[splitVar])
       
        correlations[index1] = newRow
        scores[index1][0] = s1
        scores[index1][1] = s2

        #changesDistribution = getPermutedDistribution(variables, metric, 1000000)

        #for index2 in range(changesDistribution.shape[-1]):
        #    changes = changesDistribution[:,index2]
        #    pVal = getPValue(changes, c1s[index2] - c2s[index2])
        #    pValues[index1, index2] = pVal

    corrDf = pd.DataFrame(correlations, columns = newCols)
    corrDf.index = newIndex
    
    pvalDf = pd.DataFrame(pValues, columns=newColsPvals)
    pvalDf.index = newIndex

    scoresDf = pd.DataFrame(scores, columns=newColsScores)
    scoresDf.index = newMets
    
    return corrDf, scoresDf, pvalDf


def drawPValHeatmap(pvalDf, prefix, plotYAxis):
    if plotYAxis:
        g = sns.heatmap(pvalDf, center=0.5, annot=True, cmap="YlGnBu", vmin=0, vmax=1, yticklabels=True, annot_kws={"size":25})
    else:
        g = sns.heatmap(pvalDf, center=0.5, annot=True, cmap="YlGnBu", vmin=0, vmax=1, yticklabels=False, annot_kws={"size":25})

    ax = g.axes
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 19)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    #plt.title("Changes in Correlation Coefficient from " + transitionWords , size=textSize)
    plt.title(transitionWords , size=textSize + 10)
    plt.yticks(verticalalignment="center")
    plt.tight_layout()

    if plotYAxis:
        plt.savefig(os.path.join(resultsDir, prefix + "heatmap_PVALS_short_yaxis.png"))
    else:
        plt.savefig(os.path.join(resultsDir, prefix + "heatmap_PVALS_short.png"))

    plt.show()
    plt.clf()



def drawChangesHeatmap(heatmapDf, prefix, plotYAxis):

    if plotYAxis:
        g = sns.heatmap(heatmapDf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, yticklabels=True, annot_kws={"size":25})
    else:
        g = sns.heatmap(heatmapDf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, yticklabels=False, annot_kws={"size":25})

    ax = g.axes
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 19)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    #plt.title("Changes in Correlation Coefficient from " + transitionWords , size=textSize)
    plt.title(transitionWords , size=textSize + 10)
    plt.yticks(verticalalignment="center")

    # add vertical axes to make it more visible
    for index in range(len(variables)):
        ax.add_patch(Rectangle((index * 2, 0), 2, len(metrics), fill=False, edgecolor='black', lw=1.5))

    # and horizontal axes
    for index in range(len(metrics)):
        ax.add_patch(Rectangle((0, index), len(variables) * 2, 2, fill=False, edgecolor='black', lw=0.5))

    plt.tight_layout()
    if plotYAxis:
        plt.savefig(os.path.join(resultsDir, prefix + "heatmap_CHANGES_short_yaxis.png"))
    else:
        plt.savefig(os.path.join(resultsDir, prefix + "heatmap_CHANGES_short.png"))
    plt.show()
    plt.clf()

def drawScoresChangesHeatmap(scoresDf, prefix, index):
    plt.figure(figsize=(5.5, 9))
    if index == 0:
        g = sns.heatmap(scoresDf, center=0, annot=True, cmap="icefire_r", vmin=-1, vmax=1, yticklabels=True, annot_kws={"size":25})
    else:
        g = sns.heatmap(scoresDf, center=0, annot=True, cmap="icefire_r", vmin=-1, vmax=1, yticklabels=False, annot_kws={"size":25})

    ax = g.axes
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 19)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)

    plt.yticks(verticalalignment="center")

    # add vertical axes to make it more visible
    ax.add_patch(Rectangle((0, 0), 2, len(metrics), fill=False, edgecolor='white', lw=1.5))
    ax.add_patch(Rectangle((1, 0), 2, len(metrics), fill=False, edgecolor='white', lw=1.5))

    plt.tight_layout()
    plt.savefig(os.path.join(resultsDir, prefix + "scores.png"))
    plt.show()
    plt.clf()



textSize = 15

df = pd.read_csv("mergedData.csv")
# normalize the data
df = (df - df.mean()) / df.std()


correlations = np.zeros((len(metrics), len(variables)))
scores = np.zeros((len(metrics),1))
XS = df[variables].to_numpy()
YS = df[metrics].to_numpy()
print(XS.shape)
print(YS.shape)
for i in range(YS.shape[-1]):
    y1 = YS[:,i]
    clf1 = linear_model.Lasso(alpha=0.1)
    clf1.fit(XS, y1)
    c1s = clf1.coef_
    correlations[i] = c1s
    scores[i] = clf1.score(XS, y1)
    print(metrics[i])
    print(c1s)

scores = scores.T
newVars = [varToTitle[var] for var in variables]
newMets = [varToTitle[met] for met in metrics]
ogDf = pd.DataFrame(correlations, columns=newVars)
ogDf.index = newMets

# plot the original heatmap

sns.set(rc={'figure.figsize':(10.5,9)})
g = sns.heatmap(ogDf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, yticklabels=True, annot_kws={"size":25})
ax = g.axes
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 19)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13.5)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=25)
#plt.title("Changes in Correlation Coefficient from " + transitionWords , size=textSize)
plt.title("Coefficients of Correlation across Entire Dataset", size=textSize + 10)
plt.yticks(verticalalignment="center")
plt.tight_layout()
plt.savefig(os.path.join(resultsDir, "og_heatmap.png"))
plt.show()
plt.clf()

scores = scores.T
scoresDf = pd.DataFrame(scores)
scoresDf.index = newMets
# plot the scores of the model for different flow features

def plotScores(scoresDf, saveName):
    plt.figure(figsize=(4, 9))
    g = sns.heatmap(scoresDf, center=0, annot=True, cmap="icefire_r", vmin=-1, vmax=1, xticklabels=False, yticklabels=False, annot_kws={"size":25})
    ax = g.axes
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 19)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=25)
    #plt.title("Changes in Correlation Coefficient from " + transitionWords , size=textSize)
    plt.title("Model Skill", size=textSize + 10, wrap=True)
    plt.yticks(verticalalignment="center")
    plt.tight_layout()
    plt.savefig(os.path.join(resultsDir, saveName))
    plt.show()
    plt.clf()

plotScores(scoresDf, "og_heatmap_skill.png")

plt.rcParams["figure.figsize"] = (10.5,9)

for idx, splitVar in enumerate(variables): 
    if splitVar == variables[0]:
        plotYAxis = True
    else:
        plotYAxis = False

    if splitVar == "gord":
        transitionWords = "Small to Large Streams"
        textSize = 15
        unit = ""
    elif splitVar == "MeanTempAnn":
        transitionWords = "Cold to Hot Streams"
        textSize = 15
        unit = " " + degreesC
    elif splitVar == "MeanPrecAnn":
        transitionWords = "Low Precip. to High Precip. Streams"
        textSize = 15
        unit = " (mm)"

    print("saving results to " + str(resultsDir))

    prefix = splitVar

    # evenly split the data after sorting according to stream order
    # so that we can test how relationships change across stream order
    
    # re-open the data so that pandas is okay with resetting the indices
    df = pd.read_csv("mergedData.csv")
    # normalize the data
    df = (df - df.mean()) / df.std()

    df = df.sort_values(by=splitVar)
    df = df[~df[splitVar].isna()]
    df = df.reset_index()
    indices = list(df.index)

    dfs = []
    start = 0
    stop = interval
    for split in range(numSplits):
        ldf = df.iloc[indices[start:stop]]
        dfs.append(ldf)
        start = start + interval
        stop = stop + interval

    # Create the Changes Heatmap
    # ************************************************************************************************

    heatmapDf, scoresDf, pvalDf = getChangesDf(dfs)
    drawChangesHeatmap(heatmapDf, prefix, plotYAxis)
    #drawPValHeatmap(pvalDf, prefix, plotYAxis)
    drawScoresChangesHeatmap(scoresDf, prefix, idx)
   
