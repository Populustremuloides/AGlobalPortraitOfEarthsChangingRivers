import pandas as pd
from scipy import stats
import numpy as np

numCatchments = 5
numRepeats = 100000


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib as mpl
import math

colorVar = "gords"

df = pd.read_csv("specific_discharge_vs_size.csv")
print(df)

cats = list(set(df["catchment"]))

def logData(data):
    loggedData = []
    for i in range(len(data)):
        if data[i] < 0:
            loggedData.append(-(math.sqrt(-data[i])))
        else:
            loggedData.append(math.sqrt(data[i]))
    return loggedData

dataDict = {
        "catchment":[],
        "slopes":[],
        "rVals":[],
        "pVals":[],
        "temps":[],
        "gords":[]
        }


for cat in cats:
    ldf = df[df["catchment"] == cat]
    years = list(ldf["year"])
    discharges = list(ldf["specific_discharge"])
    catTemps = list(ldf["temp"])
    catGords = list(ldf["gord"])
    if len(discharges) > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, discharges)
        
        dataDict["catchment"].append(cat)
        dataDict["slopes"].append(slope)
        dataDict["rVals"].append(r_value)
        dataDict["pVals"].append(p_value)
        dataDict["temps"].append(catTemps[0])
        dataDict["gords"].append(catGords[0])




oDf = pd.DataFrame.from_dict(dataDict)

df = df.merge(oDf, on="catchment")
for col in df.columns:
    print(col)
print("*****************************************")
df = df.groupby(["catchment"], as_index=False).mean()

for col in df.columns:
    print(col)

adf = pd.read_csv("alldata.csv")
cats = []
for cat in adf["grdc_no"]:
    try:
        cats.append(str(int(cat)))
    except:
        cats.append(None)

adf["catchment"] = cats
cats = []
for cat in df["catchment"]:
    try:
        cats.append(str(int(cat)))
    except:
        cats.append(None)
df["catchment"] = cats
df = df.merge(adf, on="catchment")
xCats = []
for cat in df["catchment"]:
    xCats.append("X" + cat)
df["catchment"] = xCats
print(df)

#df.to_csv("allDataWithDayOfMeanFlow.csv", index=False)

#df = pd.read_csv("allDataWithDayOfMeanFlow.csv")
# Read in the distance between each catchment

distDf = pd.read_csv("distance_df.csv")
print(distDf)
#distDf.index = distDf[distDf.columns[0]]
#distDf = distDf.drop(distDf.columns[0], axis=1)
cols = distDf.columns
for col in cols:
    if np.sum(distDf[col].isna()) == len(distDf[col]):
        distDf = distDf.drop(col, axis=1)
distDf = distDf[distDf[distDf.columns[0]].isin(distDf.columns)]
print(distDf)

rows = list(distDf[distDf.columns[0]])
print(len(rows))
print(len(list(set(rows))))

# Get rid of repeats (not sure how they got in there, so let's just get rid of them)
catsToNumSeen = {}
for catchment in rows:
    catsToNumSeen[catchment] = 0
for catchment in rows:
    catsToNumSeen[catchment] += 1
keeperCats = []
for catchment in catsToNumSeen.keys():
    if catsToNumSeen[catchment] == 1:
        keeperCats.append(catchment)
print(len(keeperCats))
distDf = distDf[distDf[distDf.columns[0]].isin(keeperCats)]
distDf = distDf[keeperCats]
print(distDf)

catchments = np.asarray(distDf.columns)
catsToClosest = {}
for index, col in enumerate(distDf.columns):
    distancesCol = distDf[col]
    mask = np.asarray(~distancesCol.isna())
    distancesCol =  np.asarray(distancesCol)
    distancesCol = distancesCol[mask]
    indices = np.argsort(distancesCol)
    closestCats = catchments[indices[:numCatchments]] # keep only the n closest
    catsToClosest[col] = closestCats

catsToRemove = []
def scoreSimilarity(catsToClosest, df):
    ranges = []
    for catOG in catsToClosest.keys():
        closestCats = catsToClosest[catOG]

        ldf = df[df["catchment"].isin(closestCats)]
        rangeOfSlopes = np.max(ldf["slopes"]) - np.min(ldf["slopes"])
        if np.isnan(rangeOfSlopes):
            print("error calculating range for ", catOG)
            catsToRemove.append(catOG)
        else:
            ranges.append(rangeOfSlopes)
    return np.mean(ranges)

df = df[df["catchment"].isin(keeperCats)]
print(df)
dataDict = {"mean_range":[],"category":[]}
meanRange = scoreSimilarity(catsToClosest, df)
dataDict["mean_range"].append(meanRange)
dataDict["category"].append("closest")

df = df[~df["catchment"].isin(catsToRemove)]
distCols = list(distDf.columns)
indicesToKeep = []
for i in range(len(distCols)):
    if not distCols[i] in catsToRemove:
        indicesToKeep.append(i)
distDf = distDf.iloc[indicesToKeep] # drop the rows associated with the "bad" catchments
distDf = distDf[distDf.columns[indicesToKeep]] # drop the columns
print(distDf)

def getRandomCatsToClosest(distDf):
    catchments = np.asarray(distDf.columns)
    randomCatsToClosest = {}

    for index, col in enumerate(distDf.columns):
        mask = np.asarray(~distDf[col].isna())
        indices = np.arange(distDf[col].shape[0])
        indices = indices[mask] # only keep indices with a non-null comparison
        randomIndices = np.random.choice(indices, size=numCatchments, replace=False)
        randomCats = catchments[randomIndices]
        randomCatsToClosest[col] = randomCats
    return randomCatsToClosest 

for repeat in range(numRepeats):
    # generate a random sample of "closest" watersheds
    catsToClosestRandom = getRandomCatsToClosest(distDf)

    # measure similarity
    meanRange = scoreSimilarity(catsToClosestRandom, df)

    # store the random result
    dataDict["mean_range"].append(meanRange)
    dataDict["category"].append("random")

    if repeat % 100 == 0:
        print(repeat)

        outDf = pd.DataFrame.from_dict(dataDict)
        outDf.to_csv("masd_slopes_empiricalDistribution.csv")

