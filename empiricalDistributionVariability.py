import pandas as pd
from scipy import stats
import numpy as np

numCatchments = 5
numRepeats = 100000


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from scipy import stats

adf = pd.read_csv("alldata.csv")
cats = []
for cat in adf["grdc_no"]:
    try:
        cats.append("X" + str(int(cat)))
    except:
        cats.append(None)
adf["catchment"] = cats
print(adf["grdc_no"])
#quit()

df = pd.read_csv("ml_slope_encodings_1.csv")

dataDict = {"catchment":[],"year":[],"encoding":[]}

for index, row in df.iterrows():
    catchment, year = row["catchment"].split("_")
    dataDict["catchment"].append(catchment)
    dataDict["year"].append(int(year))
    dataDict["encoding"].append(row["x"])

df = pd.DataFrame.from_dict(dataDict)
df = df[df["year"] < 2016]

print(df)
plt.hist(df["encoding"], bins=300)
plt.show()

years = list(set(df["year"]))
years.sort()
means = []
for year in years:
    ldf = df[df["year"] == year]
    mean = np.mean(ldf["encoding"])
    means.append(mean)
plt.plot(years, means)
plt.show()

catchments = list(set(df["catchment"]))
catchments.sort()

catAndSlope = {"catchment":[], "spectral_slope":[]}
for cat in catchments:
    ldf = df[df["catchment"] == cat]
    lyears = list(ldf["year"])
    lencodings = list(ldf["encoding"])
    if len(lyears) > 8:
        slope, intercept, rValue, pValue, stdErr = stats.linregress(lyears,lencodings)
        catAndSlope["catchment"].append(cat)
        catAndSlope["spectral_slope"].append(slope)

df = pd.DataFrame.from_dict(catAndSlope)
df = df.merge(adf, on="catchment")

#quit()

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
        rangeOfSlopes = np.max(ldf["spectral_slope"]) - np.min(ldf["spectral_slope"])
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
        outDf.to_csv("spectral_slopes_empiricalDistribution.csv")

