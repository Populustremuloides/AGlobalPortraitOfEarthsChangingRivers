
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
from ColorCatchments import *
from tqdm import tqdm
import copy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from ColorCatchments import *

'''
utility script for plotting data
'''

numSplits = int(sys.argv[1])
splitVar = sys.argv[2]
xVar = sys.argv[3]
yVar = sys.argv[4]
colorVar = sys.argv[5]
cmap = sys.argv[6]

if len(sys.argv) > 7:
    postFix = sys.argv[7]
else:
    postFix = ""

print("postFix: ", postFix)

prefix = splitVar + "_" + xVar + "_" + yVar + "_" + colorVar + "_" + cmap + "_"

if splitVar == "gord":
    transitionWords = "Small to Large Streams"
elif splitVar == "MeanTempAnn":
    transitionWords = "Cold to Hot Streams"
elif splitVar == "MeanPrecAnn":
    transitionWords = "Low Precip. to High Precip. Streams"
elif splitVar == "MeanPrecAnnDetrended":
    transitionWords = "Low Precip. to High Precip. Streams (detrended across stream order)"

plt.rcParams["figure.figsize"] = (10.5,7)

varToTitle = {
        "masd_mean":"Mean Annual\nSpecific Dicharge",
        "masd_slope":"Absolute Change in\nMean Annual Specific Discharge",
        "masd_slope_normalized":"Percent Change in\nMean Annual Specific Discharge",
        "domf_mean":"Day of Mean Flow",
        "domf_slope":"Absolute Change in\nDay of Mean Flow",
        "domf_slope_normalized":"Percent Change in Day of Mean Flow",
        "spectral_mean":"Mean Annual Spectral Number",
        "spectral_slope":"Absolute Change in\nMean Annual Spectral Number",
        "spectral_slope_normalized":"Percent Change in\nMean Annual Spectral Number",
        "spectral_full":"Full Spectral Number",
        "MeanTempAnn":"Mean Annual Temperature",
        "MeanPrecAnn":"Mean Annual Precipitation",
        "MeanPrecAnnDetrended":"Detrended Mean\nAnnual Precipitation",
        "gord":"Stream Order"
        }


varToUnit = {
        "masd_mean": " (L/s/km" + "2".translate(trans) + ")",
        "masd_slope":"( L/s/km" + "2".translate(trans) + ")",
        "masd_slope_normalized":"",
        "domf_mean":" (day in water year)",
        "domf_slope":" (days)",
        "domf_slope_normalized":"",
        "spectral_mean":"",
        "spectral_slope":"",
        "spectral_slope_normalized":"",
        "spectral_full":"",
        "MeanTempAnn": " (" + degreesC + ")", 
        "MeanPrecAnn":" (mm / year)",
        "MeanPrecAnnDetrended":"",
        "gord":""
        }

'''
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
        "MeanPrecAnnDetrended":"Detrended Mean Annual Precip.",
        "gord":"stream order"
        }
'''

root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/"
resultsDir = os.path.join(root, "ScatterplotFigures")

if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

print("saving results to " + str(resultsDir))

# evenly split the data after sorting according to stream order
# so that we can test how relationships change across stream order

df = pd.read_csv("mergedData.csv")


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


# now split into different sub-sets of the data
if numSplits != 1:
    dfs = []
    start = 0
    stop = interval
    for split in range(numSplits):
        ldf = df.iloc[indices[start:stop]]
        dfs.append(ldf)
        start = start + interval
        stop = stop + interval

def getMeanSplitVar(ldf, numDigits=5):
    return np.round(np.mean(ldf[splitVar]) * numDigits) / numDigits


xVarMin = np.min(df[xVar])
xVarMax = np.max(df[xVar])

yVarMin = np.min(df[yVar])
yVarMax = np.max(df[yVar])

def plot(var1, var2, colorVar, m, ldf, index, transform=None):
    v1s = np.asarray(ldf[var1])
    v2s = np.asarray(ldf[var2])
    if var1 == "gord":
        v1s = v1s + np.random.uniform(-0.49, 0.49, size=len(v1s))

    
    colors = getColors(colorVar, m, ldf, transform=transform)
    print("transform: ", transform)

    fig = plt.figure()
    plt.scatter(v1s, v2s, c=colors, alpha=0.5)
    plt.xlabel(varToTitle[var1] + varToUnit[var1], size=20, wrap=True)
    if index == 0:
        plt.ylabel(varToTitle[var2] + varToUnit[var2], size=20, wrap=True)
    plt.xlim(xVarMin, xVarMax)
    plt.ylim(yVarMin, yVarMax)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)

    if index != 0:
        plt.tick_params(left=False, right=False, labelleft=False)
    meanSplitVal = getMeanSplitVar(ldf)
    if numSplits != 1:
        plt.title("Catchments with Mean " + varToTitle[splitVar] + " of " + " {:.1f}".format(meanSplitVal) + " " + varToUnit[splitVar], size=25, wrap=True)
    else:
        plt.title("Global Distribution of Catchment Properties", size=25, wrap=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(resultsDir, prefix + str(index) + ".png"))
    plt.show()


if numSplits == 1:
    m = getM(colorVar + postFix, cmap)
    transform = getTransform(colorVar + postFix)
    plot(xVar, yVar, colorVar, m, df, 0, transform=transform)

    cmap = getCmapFromString(cmap)
    plotColorbar(colorVar + postFix, cmap, save=True, saveDir=resultsDir)

else:

    for index, ldf in enumerate(dfs):
        m = getM(colorVar + postFix, cmap)
        transform = getTransform(colorVar + postFix)
        plot(xVar, yVar, colorVar, m, ldf, index, transform=transform)

    cmap = getCmapFromString(cmap)
    plotColorbar(colorVar + postFix, cmap, save=True, saveDir=resultsDir)

