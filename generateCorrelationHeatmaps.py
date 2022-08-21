
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
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

numSplits = int(sys.argv[1])
splitVar = sys.argv[2]
plotPredictors = sys.argv[3]
if len(sys.argv) > 3:
    plotYAxis = sys.argv[4]
    if plotYAxis == "True":
        plotYAxis = True
    else:
        plotYAxis = False
else:
    plotYAxis = False

if plotPredictors == "True":
    plotPredictors = True
else:
    plotPredictors = False


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
elif splitVar == "MeanPrecAnnDetrended":
    transitionWords = "Low Precip. to High Precip. Streams"
    textSize = 15
    unit = " (mm)"

plt.rcParams["figure.figsize"] = (10.5,7)

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
        "MeanPrecAnnDetrended":"Mean Annual Precip.",
        "gord":"stream order"
        }


varToTitleShort = {
        "MeanTempAnn":"temp.",
        "MeanPrecAnn":"precip.",
        "MeanPrecAnnDetrended":"precip.",
        "gord":"stream order"
        }


root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/"
if plotPredictors:
    resultsDir = os.path.join(root, "CorrelationResults")
else:
    resultsDir = os.path.join(root, "CorrelationResultsMetrics")

if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

if plotPredictors:
    prefix = "predictors_"
else:
    prefix = "metrics_"

print("saving results to " + str(resultsDir))

# evenly split the data after sorting according to stream order
# so that we can test how relationships change across stream order

df = pd.read_csv("mergedData.csv")

prefix = splitVar + "_" + prefix 

df = (df - df.mean()) / df.std()

df = df.sort_values(by=splitVar)
df = df[~df[splitVar].isna()]

# split into (numSplits) datasets
splitVals = list(df[splitVar])
interval = len(splitVals) // numSplits
df = df.reset_index()
indices = list(df.index)

startIndices = []
stopIndices = []

start = 0
stop = interval
for split in range(numSplits):
    startIndices.append(start)
    stopIndices.append(stop)
    start = start + interval
    stop = stop + interval


def predict(slope, intercept, ts):
    return slope * ts + intercept



# now split into different sub-sets of the data
dfs = []

start = 0
stop = interval
for split in range(numSplits):
    ldf = df.iloc[indices[start:stop]]
    dfs.append(ldf)
    start = start + interval
    stop = stop + interval

# Now the real work begins:
# ************************************************************************************************

def getChangeInCorrelation(xs, ys):
    xs1 = xs[startIndices[0]:stopIndices[0]]
    xs2 = xs[startIndices[-1]:stopIndices[-1]]
    
    # adjust the dimensions to work with scipy's code
    xs1 = np.expand_dims(xs1, axis=1)
    xs2 = np.expand_dims(xs2, axis=1)

    ys1 = ys[startIndices[0]:stopIndices[0]]
    ys2 = ys[startIndices[-1]:stopIndices[-1]]

    corr1, pVal1 = stats.spearmanr(xs1, ys1)
    corr2, pVal2 = stats.spearmanr(xs2, ys2)
    
    return corr1 - corr2
   

def getPermutedDistribution(xVar, yVar, numRepeats=100000):
    xs = np.asarray(copy.copy(df[xVar]))
    ys = np.asarray(copy.copy(df[yVar]))

    changeInCorrelationReal = getChangeInCorrelation(xs, ys)

    changesInCorrelation = np.zeros((numRepeats,))
    loop = tqdm(total=numRepeats, position=0)
    for i in range(numRepeats):
        np.random.shuffle(ys) 
        np.random.shuffle(xs) 
        changeInCorrelationFake = getChangeInCorrelation(xs, ys)
        changesInCorrelation[i] = changeInCorrelationFake
        loop.update()
    loop.close()

    return  changesInCorrelation, changeInCorrelationReal
   

def roundArrayTo(array, numDigits):
    coefficient = 10 ** numDigits
    roundedArray = np.round(array * coefficient) / coefficient
    return roundedArray

def getPValue(changesInCorrelation, changeInCorrelationReal):

    changesInCorrelation = np.array(copy.copy(changesInCorrelation)) + 2 # -2 = the maximum possible negative change
    changeInCorrelationReal = np.array(copy.copy(changeInCorrelationReal)) + 2 # this makes them all positive

    numGreater = np.sum(changesInCorrelation <= changeInCorrelationReal)
    numLess = np.sum(changesInCorrelation >= changeInCorrelationReal)
    numMoreExtreme = np.min([numGreater, numLess])
    numMoreExtreme = numMoreExtreme * 2 # two-tailed
    pValue = numMoreExtreme / np.max(changesInCorrelation.shape)

    return pValue

def matrixToDf(array, axis2Labels, axis1Labels):
    dataDict = {}
    for index, axis1Label in enumerate(axis1Labels):
        dataDict[varToTitle[axis1Label]] = array[:,index]
    newDf = pd.DataFrame.from_dict(dataDict)
    newAxis2Labels = [varToTitle[var2] for var2 in axis2Labels]
    newDf.index = newAxis2Labels
    return newDf


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
    

if not plotPredictors:
    metrics = ["domf_mean", "masd_mean","spectral_mean","spectral_full","domf_slope","masd_slope_normalized","spectral_slope"]
    variables =  metrics
    variablesWithoutSplitVar = metrics

else:
    metrics = ["domf_mean", "masd_mean","spectral_mean"]#,"spectral_full"]#,"domf_slope","masd_slope_normalized","spectral_slope"]
    variables = ["MeanTempAnn","MeanPrecAnnDetrended","gord"]
    variablesWithoutSplitVar = copy.copy(variables)
    variablesWithoutSplitVar.remove(splitVar)

heatmapOG = np.zeros((len(metrics), len(variables)))
heatmapOGPVals = np.zeros((len(metrics), len(variables)))

heatmaps = [] # create a 2-d array for each sub-section of the data
heatmapsPVals = [] # create a 2-d array for each sub-section of the data

#print(heatmapOG.shape)
#if not plotPredictors:
for ldf in dfs:
    heatmap = np.zeros_like(heatmapOG)
    heatmaps.append(heatmap)

    heatmapPVal = np.zeros_like(heatmapOG)
    heatmapsPVals.append(heatmapPVal)

heatmapPVals = np.zeros_like(heatmapOG)
heatmapChanges = np.zeros_like(heatmapOG)

#print(heatmapPVals.shape)
#print(heatmapPVals.shape)

#else:
#    for ldf in dfs:
#        heatmap = np.zeros_like((len(metrics), len(variables) - 1))
#        heatmaps.append(heatmap)

#        heatmapPVal = np.zeros_like((len(metrics), len(variables) - 1))
#        heatmapsPVals.append(heatmapPVal)

#    heatmapPVals = np.zeros_like((len(metrics), len(variables) - 1))
#    heatmapChanges = np.zeros_like((len(metrics), len(variables) - 1))

removalIndex = None

for i1, metric in enumerate(metrics):
    for i2, var in enumerate(variables):

        # get m and the colorvar 
        xVar = var
        yVar = metric

        # compute the correlation between the metric and the variable
        corr, pVal = stats.spearmanr(df[metric], df[var])

        # update the heatmaps
        heatmapOG[i1, i2] =  corr
        heatmapOGPVals[i1, i2] = pVal

        
        # FIXME: remove this
        #xs = np.asarray(copy.copy(df[xVar]))
        #ys = np.asarray(copy.copy(df[yVar]))

        #changeInCorrelationReal = getChangeInCorrelation(xs, ys)
        
        # FIXME: put this back
        changesInCorrelation, changeInCorrelationReal = getPermutedDistribution(xVar, yVar, numRepeats=10)
        changePVal = getPValue(changesInCorrelation, changeInCorrelationReal)

        # update the heatmaps
        heatmapPVals[i1, i2] = changePVal
        heatmapChanges[i1, i2] = changeInCorrelationReal
    
        for i in range(len(dfs)):
            heatmap = heatmaps[i]
            heatmapPVal = heatmapsPVals[i]
            ldf = dfs[i]
            corr, pVal = stats.spearmanr(ldf[metric], ldf[var])
            heatmap[i1, i2] = corr
            heatmapPVal[i1, i2] = pVal

        if var == splitVar:
            removalIndex = i2


# remove the extraneous index from the heatmaps
#print("before: ", heatmapPVals.shape)
if removalIndex != None:
    mask = np.arange(heatmapPVals.shape[1])
    mask = mask != removalIndex
    #print(mask)
    heatmapPVals = heatmapPVals[:,mask]
    heatmapChanges = heatmapChanges[:,mask]

    for i in range(len(dfs)):
        # access the array
        heatmap = heatmaps[i]
        heatmapPVal = heatmapsPVals[i]

        # apply the mask
        heatmap = heatmap[:,mask]
        heatmapPVal = heatmapPVal[:,mask]
       
        # save it
        heatmaps[i] = heatmap
        heatmapsPVals[i] = heatmapPVal 

print("after: ", heatmapPVals.shape)
#metrics = [metric.replace("_", " ") for metric in metrics]
#variables = [variable.replace("_", " ") for variable in variables]

heatmapOG = roundArrayTo(heatmapOG, 4)
#heatmapOGPVals = roundArrayTo(heatmapOGPVals, 4)
heatmapChanges = roundArrayTo(heatmapChanges, 4)
#heatmapPVals = roundArrayTo(heatmapPVals, 4)

heatmapOGDf = matrixToDf(heatmapOG, metrics, variables)
sns.heatmap(heatmapOGDf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1)
plt.title("Correlation Coefficients across All Stream Orders", size=textSize)
plt.yticks(verticalalignment="center")
plt.tight_layout()
plt.savefig(os.path.join(resultsDir, prefix + "heatmapsOG_short.png"))
plt.clf()
#plt.show()

heatmapOGPValsDf = matrixToDf(heatmapOGPVals, metrics, variables)
sns.heatmap(heatmapOGPValsDf, center=0, annot=True, cmap="magma_r", vmin=0, vmax=1)
plt.title("P-Values of Correlations across All Stream Orders", size=textSize)
plt.yticks(verticalalignment="center")
plt.tight_layout()
plt.savefig(os.path.join(resultsDir, prefix + "heatmapsOGPVals_short.png"))
plt.clf()
#plt.show()


heatmapChangesDf = matrixToDf(heatmapChanges, metrics, variablesWithoutSplitVar)
sns.heatmap(heatmapChangesDf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1)
#plt.show()

#plt.title("Changes in Correlation Coefficient from " + transitionWords , size=textSize)
plt.title(transitionWords , size=textSize + 10)
plt.yticks(verticalalignment="center")
plt.tight_layout()
plt.savefig(os.path.join(resultsDir, prefix + "heatmapsChanges_short.png"))
plt.clf()
#plt.show()



heatmapPValsDf = matrixToDf(heatmapPVals, metrics, variablesWithoutSplitVar)
sns.heatmap(heatmapPValsDf, center=0, annot=True, cmap="magma_r", vmin=-1, vmax=1)
plt.title("Empirical P-values for Changes in Correlation", size=15)
plt.yticks(verticalalignment="center")
plt.tight_layout()
plt.savefig(os.path.join(resultsDir, prefix + "heatmapsPVals_short.png"))
plt.clf()
#plt.show()


# get concise change figure

df1 = dfs[0]
df2 = dfs[-1]

meanSplitVar1 = getMeanSplitVar(df1, 3)
heatmap1 = heatmaps[0]

meanSplitVar2 = getMeanSplitVar(df2, 3)
heatmap2 = heatmaps[-1]

heatmap1 = roundArrayTo(heatmap1, 3)
heatmap2 = roundArrayTo(heatmap2, 3)


heatmapDf = matrixToDfChange(heatmap1, heatmap2, metrics, variablesWithoutSplitVar, splitVar)
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
for index in range(len(variablesWithoutSplitVar)):
    ax.add_patch(Rectangle((index * 2, 0), 2, len(metrics), fill=False, edgecolor='black', lw=1.5))
# and horizontal axes
for index in range(len(metrics)):
    ax.add_patch(Rectangle((0, index), len(variablesWithoutSplitVar) * 2, 2, fill=False, edgecolor='black', lw=0.5))

plt.tight_layout()
if plotYAxis:
    plt.savefig(os.path.join(resultsDir, prefix + "heatmap_CHANGES_short_yaxis.png"))
else:
    plt.savefig(os.path.join(resultsDir, prefix + "heatmap_CHANGES_short.png"))
plt.clf()


for i in range(len(dfs)):
    ldf = dfs[i]
    meanSplitVar = getMeanSplitVar(ldf, 3)
    heatmap = heatmaps[i]
    heatmapPVal = heatmapsPVals[i]

    heatmap = roundArrayTo(heatmap, 4)
    heatmapDf = matrixToDf(heatmap, metrics, variablesWithoutSplitVar)
    sns.heatmap(heatmapDf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1)
    plt.title("Correlation Coefficients for Catchments with Mean " + varToTitle[splitVar] + " of " + " {:.1f}".format(meanSplitVar) + unit, size=textSize)
    plt.yticks(verticalalignment="center")
    plt.tight_layout()
    plt.savefig(os.path.join(resultsDir, prefix + "heatmap_" + str(i) + "_short.png"))
    plt.clf()
    #plt.show()

    heatmapPVal = roundArrayTo(heatmapPVal, 4)
    heatmapPValDf = matrixToDf(heatmapPVal, metrics, variablesWithoutSplitVar)
    sns.heatmap(heatmapPValDf, center=0, annot=True, cmap="magma_r", vmin=-1, vmax=1)
    plt.title("P-values of Correlations for Catchments with Mean " + varToTitle[splitVar] + " of " + " {:.1f}".format(meanSplitVar) + unit, size=textSize)
    plt.yticks(verticalalignment="center")
    plt.tight_layout()
    plt.savefig(os.path.join(resultsDir, prefix + "heatmapPVal_" + str(i) + "_short.png"))
    plt.clf()
    #plt.show()



