import pandas as pd
from scipy import stats
import numpy as np

numCatchments = 5
numRepeats = 10000

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib as mpl
import math
from tqdm import tqdm


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

print(distDf)
print(df)

# before we do anything, we need to make sure the distDf and df have the same set of catchments

catchments = np.asarray(distDf.columns)
catsToClosest = {}
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
        j = j + 1
    closestCats = catchments[indices[:numCatchments]] # keep only the n closest
    catsToClosest[col] = closestCats

for cat in catsToClosest:
    if len(catsToClosest[cat]) != numCatchments:
        print("error")

def scoreSimilarity(catsToClosest, df, distanceVar):
    #print(" IN SCORE SIMILARITY ******************************************** ")
    ranges = []
    for catOG in catsToClosest.keys():
        closestCats = catsToClosest[catOG]
        ldf = df[df["catchment"].isin(closestCats)]
        rangeOfSlopes = np.max(ldf[distanceVar]) - np.min(ldf[distanceVar])
        if np.isnan(rangeOfSlopes):
            pass
            print("error calculating range for ", catOG)
            print(closestCats)
            print(ldf[distanceVar])
            print(ldf[distanceVar].isna())
            print(np.sum(ldf[distanceVar].isna()))
            print()
    #        input("any key")
        if len(ldf["catchment"]) != numCatchments:
            print("error calculating ranges - wrong number of columns")
            print(len(ldf["catchment"]))
        else:
    #        print("doing fine")
            ranges.append(rangeOfSlopes)
    return np.mean(ranges)

def getRandomCatsToClosest(catchments):
    newDict = {}
    for i, cat in enumerate(catchments):
        # don't self-sample
        mask = np.arange(catchments.shape[0]) != i
        # randomly pick 5 catchments
        newDict[cat] = np.random.choice(catchments[mask], numCatchments, replace=False)
    return newDict


# read in flow data
dataDict = {"mean_range":[],"category":[], "metric":[], "repeat":[]}
metrics = ["domf_slope","masd_slope_normalized","spectral_slope"]

loop = tqdm(total=len(metrics) * numRepeats, position=0)

catchments = df["catchment"].to_numpy()
for metric in metrics:
    meanRange = scoreSimilarity(catsToClosest, df, metric)
    dataDict["mean_range"].append(meanRange)
    dataDict["category"].append("closest")
    dataDict["metric"].append(metric)
    dataDict["repeat"].append(0)

    for repeat in range(numRepeats):

        # generate a random sample of "closest" watersheds
        catsToClosestRandom = getRandomCatsToClosest(catchments)

        # measure similarity
        meanRange = scoreSimilarity(catsToClosestRandom, df, metric)

        # store the random result
        dataDict["mean_range"].append(meanRange)
        dataDict["category"].append("random")
        dataDict["metric"].append(metric)
        dataDict["repeat"].append(repeat)
        
        loop.update()

        if repeat % 100 == 0:
            print(repeat)

            outDf = pd.DataFrame.from_dict(dataDict)
            outDf.to_csv("slopes_empiricalDistributions.csv")

loop.close()
