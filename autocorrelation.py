
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
if len(sys.argv) > 4:
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
    variables = ["MeanTempAnn","MeanPrecAnn","gord"]
    variablesWithoutSplitVar = copy.copy(variables)
    variablesWithoutSplitVar.remove(splitVar)

indexToPosition = {0:"lowest third", 1:"middle third", 2:"highest third"}


cdf = copy.copy(df)
cdf = cdf[variables]
for var in variables:
    cdf[varToTitle[var]] = cdf[var]
    cdf = cdf.drop(var, axis=1)
cdf = cdf.rename(varToTitle)
print(cdf)
hm = cdf.corr()

g = sns.heatmap(hm, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, cbar=True)
ax = g.axes
ax.set_yticklabels(ax.get_ymajorticklabels())

plt.yticks(verticalalignment="center")
plt.title("Correlations between Independent Variables", fontsize=15, wrap=True)
plt.savefig(os.path.join(resultsDir, "correlationInData_all.png"))
plt.show()


for i in range(len(dfs)):
    ldf = dfs[i][variables]
    for var in variables:
        ldf[varToTitle[var]] = ldf[var]
        ldf = ldf.drop(var, axis=1)
    ldf = ldf.rename(varToTitle)
    print(ldf)
    hm = ldf.corr()
    if i == 0:
        g = sns.heatmap(hm, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, cbar=True)
        ax = g.axes
        ax.set_yticklabels(ax.get_ymajorticklabels())
    elif i == 1:
        g = sns.heatmap(hm, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, cbar=True)
        ax = g.axes
        ax.set_yticklabels([])
    elif i == 2:
        g = sns.heatmap(hm, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, cbar=True)
        ax = g.axes
        ax.set_yticklabels([])

    plt.yticks(verticalalignment="center")
    plt.title(indexToPosition[i] + " of data", fontsize=15, wrap=True)
    plt.savefig(os.path.join(resultsDir, "correlationInData_" + str(splitVar) + "_" + str(i) + ".png"))
    plt.show()

