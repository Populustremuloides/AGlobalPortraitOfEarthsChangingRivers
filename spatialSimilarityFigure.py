import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import copy 

root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/spatialResults"
if not os.path.exists(root):
    os.mkdir(root)

name = "slopes_empiricalDistributions"
for i in range(1, 4):
    if i == 1:
        df = pd.read_csv(name + ".csv")
        print(df)
    else:
        df1 = pd.read_csv(name + str(i) + ".csv") 
        df = pd.concat([df, df1], axis=0)

print(df)

metrics = list(set(df["metric"]))
metrics.sort()

varToTitle = {
        "domf_slope":"Day of Mean Flow Slope",
        "masd_slope_normalized":"Mean Annual Specific Discharge Percent Change",
        "spectral_slope":"Spectral Number Slope"}

def getPValue(meanRanges, realMeanRange):

    meanRanges = np.array(copy.copy(meanRanges))  # -2 = the maximum possible negative change
    realMeanRange = np.array(copy.copy(realMeanRange[0]))  # this makes them all positive

    numGreater = np.sum(meanRanges <  realMeanRange)
    numLess    = np.sum(meanRanges >= realMeanRange)

    numMoreExtreme = np.min([numGreater, numLess])
    numMoreExtreme = numMoreExtreme * 2 # two-tailed

    pValue = numMoreExtreme / np.max(meanRanges.shape)

    return pValue


for metric in metrics:
    ldf = df[df["metric"] == metric]
    
    realVal = ldf[ldf["category"] == "closest"]["mean_range"].to_numpy()
    fakeVals = ldf[ldf["category"] == "random"]["mean_range"].to_numpy()
    
    pval = getPValue(fakeVals, realVal)

    print()
    print(metric)
    print(pval)
    print()

    heights, boundaries, patches = plt.hist(fakeVals, bins=50, density=True)
    plt.vlines(realVal, color="r", ymin=0, ymax=np.max(heights), label="real global mean range")
    plt.ylabel("density")
    plt.xlabel("mean range")
    plt.title(varToTitle[metric], fontname="Helvetica")
    plt.legend()
    plt.savefig(os.path.join(root, metric + "_pval.png"))
    plt.show() 


numCatchments = 5

distDf = pd.read_csv("distance_df.csv")
print(distDf)

# deal with some messiness in the distances data *************

# drop catchments without lat/lon info
cols = distDf.columns
for col in cols:
    if np.sum(distDf[col].isna()) == len(distDf[col]):
        distDf = distDf.drop(col, axis=1)
distDf = distDf[distDf[distDf.columns[0]].isin(distDf.columns)]

rows = list(distDf[distDf.columns[0]])

# Get rid of repeats (not sure how they got in there, so let's just get rid of them in case there is anything funny about them)
catsToNumSeen = {}
for catchment in rows:
    catsToNumSeen[catchment] = 0
for catchment in rows:
    catsToNumSeen[catchment] += 1

keeperCats = []
dropperCats = []
for catchment in catsToNumSeen.keys():
    if catsToNumSeen[catchment] == 1:
        keeperCats.append(catchment)
    else:
        dropperCats.append(catchment)

distDf = distDf[distDf[distDf.columns[0]].isin(keeperCats)]
distDf = distDf[keeperCats]
keeperCats = list(distDf.columns)

df = pd.read_csv("mergedData.csv")
df = df[df["catchment"].isin(keeperCats)]

A = set(df["catchment"])
B = set(distDf.columns)
keeperCats = list(A.intersection(B))
mask = []
for cat in distDf.columns:
    if cat in keeperCats:
        mask.append(True)
    else:
        mask.append(False)
mask = np.array(mask)

distDf = distDf.iloc[:,mask]
distDf = distDf.iloc[mask,:]
#distDf = distDf[keeperCats]
df = df[df["catchment"].isin(keeperCats)]

distDf.index = distDf.columns

# before we do anything, we need to make sure the distDf and df have the same set of catchments

catchments = np.asarray(distDf.columns)
catsToClosest = {}
distances = []
for index, col in enumerate(df["catchment"]):
    distancesCol = distDf[col]
    mask = np.asarray(~distancesCol.isna())

    distancesCol =  np.asarray(distancesCol)
    distancesCol = distancesCol[mask]
    indices = np.argsort(distancesCol)
    closestCats = []
    j = 0
    while len(closestCats) <= numCatchments:
        catOfInterest = catchments[indices[j]]
        if catOfInterest in distDf.columns and catOfInterest in list(df["catchment"]):
            if not np.isnan(distDf[col][catOfInterest]):
                closestCats.append(catOfInterest) 
                distances.append(distDf[col][catOfInterest])
        j = j + 1
    closestCats = catchments[indices[:numCatchments]] # keep only the n closest
    catsToClosest[col] = closestCats

for cat in catsToClosest:
    if len(catsToClosest[cat]) != numCatchments:
        print("error")


# read in flow data
catchments = df["catchment"].to_numpy()

print(np.mean(distances))
heights, boundaries, patches = plt.hist(distances, bins=75)
plt.vlines(np.mean(distances), color="r", ymin=0, ymax=np.max(heights), label="distribution mean")
plt.title("Distances to 5 Nearest Catchments")
plt.xlabel("distance")
plt.legend()
plt.ylabel("count")
plt.savefig(os.path.join(root, "distances.png"))
plt.show()
