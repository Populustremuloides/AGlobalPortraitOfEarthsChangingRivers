import pandas as pd
from scipy import stats
import numpy as np

numCatchments = 5
numRepeats = 100000

'''
df = pd.read_csv("dayOfMeanFlowThroughTime.csv", index_col=None)
print(df)

cats = list(set(df["catchment"]))
cats.sort()

catToSlope = {
        "catchment":[],
        "dayOfMeanFlow_slope":[]
        }
for cat in cats:
    ldf = df[df["catchment"] == cat]
    years = ldf["year"]
    domf = ldf["day_of_mean_flow"]
    slope, intercept, rValue, pValue, stdErr = stats.linregress(years, domf)
    catToSlope["catchment"].append(cat)
    catToSlope["dayOfMeanFlow_slope"].append(slope)

df = pd.DataFrame.from_dict(catToSlope)

alldata = pd.read_csv("alldata.csv", encoding="latin-1")
for column in alldata.columns:
    print(column)

# fix alldata
catchments = []
for cat in alldata["grdc_no"]:
    try:
        catchments.append("X" + str(int(cat)))
    except:
        catchments.append(None)
alldata["catchment"] = catchments

df = df.merge(alldata, on="catchment")
df.to_csv("allDataWithDayOfMeanFlow.csv", index=False)
'''
df = pd.read_csv("allDataWithDayOfMeanFlow.csv")
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
    
def scoreSimilarity(catsToClosest, df):
    ranges = []
    for catOG in catsToClosest.keys():
        closestCats = catsToClosest[catOG]

        ldf = df[df["catchment"].isin(closestCats)]
        rangeOfSlopes = np.max(ldf["dayOfMeanFlow_slope"]) - np.min(ldf["dayOfMeanFlow_slope"])
        ranges.append(rangeOfSlopes)
    return np.mean(ranges)

df = df[df["catchment"].isin(keeperCats)]
dataDict = {"mean_range":[],"category":[]}
meanRange = scoreSimilarity(catsToClosest, df)
dataDict["mean_range"].append(meanRange)
dataDict["category"].append("closest")

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
        outDf.to_csv("dayOfMeanFlow_slopes_empiricalDistribution.csv")

