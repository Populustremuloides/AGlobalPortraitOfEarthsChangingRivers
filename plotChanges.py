
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
from sklearn import linear_model
from sklearn import preprocessing
from scipy.stats import spearmanr

'''
utility script for plotting data
'''

absolute = True

def plot(ax, var1, var2, colorVar, m, df, index1, index2, transform=None):
    v1s = np.asarray(df[var1])
    v2s = np.asarray(df[var2])
    if var1 == "gord":
        v1s = v1s + np.random.uniform(-0.49, 0.49, size=len(v1s))
    
    colors = getColors(colorVar, m, df, transform=transform)
    print("transform: ", transform)

    #fig = plt.figure()
    if absolute:
        ax.scatter(v1s, abs(v2s), c=colors, alpha=0.9)
    else:
        ax.scatter(v1s, v2s, c=colors, alpha=0.9)

    if index1 == 2:
        ax.set_xlabel(varToTitle[var1] + varToUnit[var1], size=15, wrap=True)
    if index1 == 0:
        pass
        #ax.set_title("Magnitudes of Changes in Flow Regime", size=20)
    
    if index2 == 0:
        ax.set_ylabel(varToTitle[var2] + "\n" + varToUnit[var2], size=10, wrap=True)
    else:
        ax.set_yticklabels([])
        # get rid of y ticks and y labels

    #plt.xlim(xVarMin, xVarMax)
    #plt.ylim(0, yVarMax)
    #ax.yticks(fontsize = 15)
    #ax.xticks(fontsize = 15)

    #if index != 0:
    #    ax.tick_params(left=False, right=False, labelleft=False)
    #ax.title("Global Distribution of Catchment Properties", size=25, wrap=True)

    #ax.tight_layout(rect=[0, 0.03, 1, 0.95])


plt.rcParams["figure.figsize"] = (10.5,10)


if absolute:
    varToTitle = {
        "masd_mean":"Mean Annual\nSpecific Dicharge",
        "masd_slope":"Absolute Change in\nMean Annual Specific Discharge",
        "masd_slope_normalized":"Absolute Percent Change in\nMean Annual Specific Discharge",
        "domf_mean":"Day of Mean Flow",
        "domf_slope":"Absolute Change in\nDay of Mean Flow",
        "domf_slope_normalized":"Percent Change in Day of Mean Flow",
        "spectral_mean":"Mean Annual Spectral Number",
        "spectral_slope":"Absolute Change in\nMean Annual Spectral Number",
        "spectral_slope_normalized":"Percent Change in\nMean Annual Spectral Number",
        "spectral_full":"Full Spectral Number",
        "MeanTempAnn":"Mean Annual Temperature",
        "MeanPrecAnn":"Mean Annual Precipitation",
        "gord":"Stream Order (with jitter)"
        }
else:
    varToTitle = {
        "masd_mean":"Mean Annual\nSpecific Dicharge",
        "masd_slope":"Change in\nMean Annual Specific Discharge",
        "masd_slope_normalized":"Percent Change in\nMean Annual Specific Discharge",
        "domf_mean":"Day of Mean Flow",
        "domf_slope":"Change in\nDay of Mean Flow",
        "domf_slope_normalized":"Percent Change in Day of Mean Flow",
        "spectral_mean":"Mean Annual Spectral Number",
        "spectral_slope":"Change in\nMean Annual Spectral Number",
        "spectral_slope_normalized":"Percent Change in\nMean Annual Spectral Number",
        "spectral_full":"Full Spectral Number",
        "MeanTempAnn":"Mean Annual Temperature",
        "MeanPrecAnn":"Mean Annual Precipitation",
        "gord":"Stream Order (with jitter)"
        }


if absolute:
    varToTitleS = {
        "masd_mean":"Mean Annual Specific Dicharge",
        "masd_slope":"Absolute Change in Mean Annual Specific Discharge",
        "masd_slope_normalized":"Absolute Percent Change in Mean Annual Specific Discharge",
        "domf_mean":"Day of Mean Flow",
        "domf_slope":"Absolute Change in Day of Mean Flow",
        "domf_slope_normalized":"Percent Change in Day of Mean Flow",
        "spectral_mean":"Mean Annual Spectral Number",
        "spectral_slope":"Absolute Change in Mean Annual Spectral Number",
        "spectral_slope_normalized":"Percent Change in Mean Annual Spectral Number",
        "spectral_full":"Full Spectral Number",
        "MeanTempAnn":"Mean Annual Temperature",
        "MeanPrecAnn":"Mean Annual Precipitation",
        "gord":"Stream Order"
        }
else:
    varToTitleS = {
        "masd_mean":"Mean Annual Specific Dicharge",
        "masd_slope":"Change in Mean Annual Specific Discharge",
        "masd_slope_normalized":"Percent Change in Mean Annual Specific Discharge",
        "domf_mean":"Day of Mean Flow",
        "domf_slope":"Change in Day of Mean Flow",
        "domf_slope_normalized":"Percent Change in Day of Mean Flow",
        "spectral_mean":"Mean Annual Spectral Number",
        "spectral_slope":"Change in Mean Annual Spectral Number",
        "spectral_slope_normalized":"Percent Change in Mean Annual Spectral Number",
        "spectral_full":"Full Spectral Number",
        "MeanTempAnn":"Mean Annual Temperature",
        "MeanPrecAnn":"Mean Annual Precipitation",
        "gord":"Stream Order"
        }


varToUnit = {
        "masd_mean": " (L/s/km" + "2".translate(trans) + ")",
        "masd_slope":"( L/s/km" + "2".translate(trans) + ")",
        "masd_slope_normalized":"(% change / year)",
        "domf_mean":" (day in water year)",
        "domf_slope":" (change in days / year)",
        "domf_slope_normalized":"(% change / year)",
        "spectral_mean":"",
        "spectral_slope":"(change in spectral number / year)",
        "spectral_slope_normalized":"% change / year",
        "spectral_full":"",
        "MeanTempAnn": " (" + degreesC + ")", 
        "MeanPrecAnn":" (mm / year)",
        "gord":""
        }

root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/"
resultsDir = os.path.join(root, "ScatterplotFigures")

if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

print("saving results to " + str(resultsDir))

# evenly split the data after sorting according to stream order
# so that we can test how relationships change across stream order

df = pd.read_csv("mergedData.csv")


xVars = ["MeanTempAnn","MeanPrecAnn","gord"]
colorVars = ["MeanPrecAnn","MeanTempAnn","MeanTempAnn"]
cmaps = ["precip","temp","temp"]

predictors = ["MeanTempAnn", "MeanPrecAnn","gord"]
metrics = ["domf_slope","masd_slope_normalized","spectral_slope"]


for j, xVar in enumerate(xVars):
    cmap = cmaps[j]    
    colorVar = colorVars[j]

    if colorVar == "MeanPrecAnn":
        postFix = "Log"
    else:
        postFix = ""


    fig, axs = plt.subplots(3, 1, sharex=True)

    for i, yVar in  enumerate(metrics):
        ax = axs[i] 
        prefix = xVar + "_" + yVar + "_" + colorVar + "_" + cmap + "_" 
        xVarMin = np.min(df[xVar])
        xVarMax = np.max(df[xVar])

        yVarMin = np.min(df[yVar])
        yVarMax = np.max(df[yVar])

        m = getM(colorVar + postFix, cmap)
        transform = getTransform(colorVar + postFix)
        plot(ax, xVar, yVar, colorVar, m, df, i, j, transform=transform)

    if absolute:
        plt.savefig(os.path.join(resultsDir, prefix + "_absolute.png"))
    else:
        plt.savefig(os.path.join(resultsDir, prefix + ".png"))

    plt.show()

    cmapR = getCmapFromString(cmap)
    if j == 0:
        pLeft = True
    else:
        pLeft = False
    plotColorbar(colorVar + postFix, cmapR, save=True, saveDir=resultsDir, pLeft=pLeft)
  
# run the correlation analysis

# normalize the data
xs = df[predictors].to_numpy()
ys = df[metrics].to_numpy()
if absolute:
    ys = np.abs(ys)

scalerXs = preprocessing.StandardScaler().fit(xs)
xs = scalerXs.transform(xs)

scalerYs = preprocessing.StandardScaler().fit(ys)
ys = scalerYs.transform(ys)

coefficients = np.zeros((ys.shape[-1], xs.shape[-1]))

for i in range(ys.shape[-1]):
    y = ys[:,i]
    cc, pval = spearmanr(xs, y)

    print()
    print(metrics[i])
    print(cc)
    print(pval)
    print()

    clf1 = linear_model.Lasso(alpha=0.05)
    clf1.fit(xs, y)
    c1s = clf1.coef_
    coefficients[i] = c1s
    print(c1s)
    print(clf1.score(xs, y))
    print()

# get good names for figure
predictorsTitles = [varToTitleS[var] for var in predictors]
metricsTitles = [varToTitleS[var] for var in metrics]
# convert to dataframe
odf = pd.DataFrame(coefficients, columns=predictorsTitles)
odf.index = metricsTitles
print(odf)

textSize = 20

# plot it
g = sns.heatmap(odf, center=0, annot=True, cmap="vlag", vmin=-1, vmax=1, yticklabels=True, annot_kws={"size":25})
ax = g.axes
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 13, wrap=True)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 10, wrap=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.yticks(verticalalignment="center")
plt.title("Lasso Regression Coefficients Predicting Changes in Streamflow Regime", size=textSize + 10, wrap=True)
plt.savefig(os.path.join(resultsDir, "correlation_heatmap.png"))
plt.show()
