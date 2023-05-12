import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as ss
from sklearn.metrics import silhouette_samples
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os 
import copy
from ColorCatchments import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

varToTitle = {
        "masd_mean":"Mean of Mean Annual Specific Discharge",
        "masd_slope":"Slope of Mean Annual Specific Discharge",
        "masd_slope_normalized":"% Change in Mean Annual Specific Discharge",
        "domf_mean":"Mean Day of Mean Flow",
        "domf_slope":"Slope of Day of Mean Flow",
        "domf_slope_normalized":"% Change in Day of Mean Flow",
        "spectral_mean":"Mean Spectral Number",
        "spectral_slope":"Slope of Spectral Number",
        "spectral_slope_normalized":"% Change in Spectral Number",
        "spectral_full":"Full Spectral Number",
        "MeanTempAnn":"Mean Annual Temp.",
        "MeanPrecAnn":"Mean Annual Precip.",
        "gord":"stream order"
        }

#root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/"
root = os.getcwd()
resultsDir = os.path.join(root, "ClusteringResults")

minNumClusters = 2

if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

def numpyToLongDf(scores):
    dataDict = {"silhouette score":[],"repeat":[],"number of clusters":[]}

    for i in range(scores.shape[0]):
        for j in range(scores.shape[-1]):
            score = scores[i,j]
            dataDict["silhouette score"].append(score)
            dataDict["repeat"].append(i)
            dataDict["number of clusters"].append(j + minNumClusters)

    scoresDf = pd.DataFrame.from_dict(dataDict)
    return scoresDf

df = pd.read_csv("mergedData.csv")

xs = df["new_lon.x"].to_numpy()
ys = df["new_lat.x"].to_numpy()

#print(slopeVar)
predictors = ["domf_mean","masd_mean", "spectral_mean"]
X = df[predictors].to_numpy()

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

numRepeats = 20
maxClusters = 10

'''
nClusters = list(range(minNumClusters,maxClusters))
scores = np.zeros((numRepeats, len(nClusters)))
loop = tqdm(total=len(nClusters) * numRepeats, position=0)

for j, rep in enumerate(range(numRepeats)):
    for k, num in enumerate(nClusters):
        model = KMeans(n_clusters=num).fit(X)
        labels = model.labels_
        score = ss(X, labels, metric="euclidean")
        scores[j, k] = score
        loop.update()
loop.close()

meanScores = np.mean(scores, axis=0)
print()
print(meanScores)
print("optimal number of clusters: ", nClusters[np.argsort(meanScores)[-1]])
print()

scoreDf = numpyToLongDf(scores)

sns.lineplot(data=scoreDf, x="number of clusters", y="silhouette score")
plt.title("Optimal # of Clusters for Global Streamflow Data")
plt.savefig(os.path.join(resultsDir,"abstract_numClusters.png"))
plt.show()
'''

# generate the map plot

nClusters = 5 #nClusters[np.argsort(meanScores)[-1]] 

model = KMeans(n_clusters=nClusters).fit(X)
realLabels = model.labels_

plt.hist(realLabels)
plt.show()

realScore = ss(X, realLabels, metric="euclidean")

fig = plt.figure(figsize=(11 * 2,6 * 2))

ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-70,83], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)

labelTypes = list(set(realLabels))
labelTypes.sort()
for labelType in labelTypes:
    mask = realLabels == labelType
    lxs = xs[mask]
    lys = ys[mask]
    ax.scatter(lxs, lys, s=4, alpha=0.75, label=str(labelType + 1))

lgnd = plt.legend(title="cluster #", prop={"size":20}, title_fontsize=15)
lgnd.legendHandles[0]._sizes = [50]
lgnd.legendHandles[1]._sizes = [50]
lgnd.legendHandles[2]._sizes = [50]
lgnd.legendHandles[3]._sizes = [50]
lgnd.legendHandles[4]._sizes = [50]
plt.title("Clusters Generated Using Timing, Magnitude, and Variability of Flow", fontsize=30)
plt.savefig(os.path.join(resultsDir, "abstract_map.png"))
plt.show()


fig, ax1 = plt.subplots()

cluster_labels = model.predict(X)
sample_silhouette_values = silhouette_samples(X, cluster_labels)


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

y_lower = 10
for i in range(nClusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    #color = cm.nipy_spectral(float(i) / nClusters)
    ax1.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=colors[i],
        edgecolor=colors[i],
 
#        facecolor=colors[4 - i],
#        edgecolor=colors[4 - i],
        alpha=0.7,
    )

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(nClusters - i))
    #ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, 1 + i)#str(nClusters[np.argsort(meanScores)[-1]] - i))
    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("Silhouette Scores for Each Catchment")
ax1.set_xlabel("silhouette score")
ax1.set_ylabel("cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=realScore, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.savefig(os.path.join(resultsDir, "abstract_silhouette.png"))
plt.show()
