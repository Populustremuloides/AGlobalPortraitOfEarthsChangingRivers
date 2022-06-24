import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

numCatchments = 5

distDf = pd.read_csv("distance_df.csv")
cols = distDf.columns
for col in cols:
    if np.sum(distDf[col].isna()) == len(distDf[col]):
        distDf = distDf.drop(col, axis=1)
distDf = distDf[distDf[distDf.columns[0]].isin(distDf.columns)]
print(distDf)

rows = list(distDf[distDf.columns[0]])

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
closestDistances = []
for index, col in enumerate(distDf.columns):
    distancesCol = distDf[col]
    mask = np.asarray(~distancesCol.isna())
    distancesCol =  np.asarray(distancesCol)
    distancesCol = distancesCol[mask]
    indices = np.argsort(distancesCol)
    closestDistances = closestDistances + list(distancesCol[indices[:numCatchments]])
    closestCats = catchments[indices[:numCatchments]] # keep only the n closest
    catsToClosest[col] = closestCats

print("mean distance to 5 closest catchments: ")
print(np.mean(closestDistances))

plt.hist(closestDistances, bins=100)
plt.vlines(np.mean(closestDistances), ymin=0, ymax=5000, colors="r", label="mean")
plt.title("Distances to 5 Closest Catchments")
plt.ylabel("count")
plt.xlabel("kilometers")
plt.legend()
plt.show()
quit()

df1 = pd.read_csv("dayOfMeanFlow_slopes_empiricalDistribution.csv", index_col=None)
df2 = pd.read_csv("masd_slopes_empiricalDistribution.csv", index_col=None)
df3 = pd.read_csv("spectral_slopes_empiricalDistribution.csv", index_col=None)
df4 = pd.read_csv("spectral_slopes_empiricalDistributionSign.csv", index_col=None)



def plotDf(df, title):
    print(df)
    ranges = list(df["mean_range"])
    true = ranges[0]
    simulated = ranges[1:]
    plt.hist(simulated, bins=100)
    plt.vlines(true, ymin=0, ymax=5000, colors="r", label="true mean range, 5 nearest catchments")
    plt.ylim(0,600)
    plt.title(title)
    plt.xlabel("simulated mean range in slopes, 5 random catchments")
    plt.ylabel("count")
    plt.legend()
    plt.show()

plotDf(df1, "Change in Day of Mean Flow")
plotDf(df2, "Change in Mean Annual Specific Discharge")
plotDf(df3, "Change in Spectral Number")
plotDf(df4, "Absolute Value of Change in Spectral Number")
