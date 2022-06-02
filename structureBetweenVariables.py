import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import math
import random
from scipy import stats
import matplotlib.pyplot as plt
import time

import matplotlib.cm as cm
import matplotlib as mpl
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib


removeSparse = True
copyNo = 0

df = pd.read_csv("throughTimeCombined.csv")
#df = df.drop(df.columns[-2], axis=1)
#df = df.drop(df.columns[-3], axis=1)
print(df)
#quit()
cdf = pd.read_csv("alldata_hemisphereCorrected.csv")


# keep track of which columns have a lot of NaN values
tooSparseColumns = []
for column in cdf:
    num = np.sum(cdf[column].isna())
    if num > 500:
        tooSparseColumns.append(column)
    #print(str(column) + " " + str(num))

for column in cdf:
    column1 = column.replace("_","")
    if type(cdf[column][0]) == type("string") and column1.isalpha():
        codes, uniques = pd.factorize(cdf[column])
        cdf[column] = codes

xcats = []
for cat in cdf["grdc_no"]:
    try:
        xcats.append("X" + str(int(cat))) 
    except:
        xcats.append(None)

cdf["catchment"] = xcats
cdf = cdf.drop("grdc_no", axis=1)
df = df.merge(cdf, on="catchment")
print(df)


for i in range(len(df.columns)):
    print(str(i) + " " + str(df.columns[i]))


print(df.columns[1:7])

#df = df[df["domf_slope"] > -0.004]
dims = [list(df["domf_mean"]), list(df["domf_slope"]), list(df["masd_mean"]), list(df["masd_slope"]), list(df["spectral_mean"]), list(df["spectral_slope"])]
#print(df.columns[2])
#print(df.columns[1])
#print(len(dims[0]))
#quit()
corrMatrix = []
pValMatrix = []
i = 0
for dim1 in dims:
    corrList = []    
    pValList = []
    j = 0
    for dim2 in dims:
        corr, pVal = stats.pearsonr(dim1, dim2)
        print("i: " + str(i) + ", j: " + str(j) + ": " + str(corr))
        corrList.append(corr)
        pValList.append(pVal)
        j = j + 1
        print(j)
    corrMatrix.append(corrList)
    pValMatrix.append(pValList)
    i = i + 1

corrMatrix = np.asarray(corrMatrix)
pValMatrix = np.asarray(pValMatrix)
print(corrMatrix)
print(corrMatrix.shape)
sns.heatmap(corrMatrix, annot=True, vmin=-1, vmax=1, center=0)
plt.show()

sns.heatmap(pValMatrix, annot=True, vmin=1, vmax=0)
plt.show()


minTemp = np.min(df["MeanTempAnn"])
maxTemp = np.max(df["MeanTempAnn"])

minGord = np.min(df["gord"])
maxGord =  np.max(df["gord"])

minPrecip = np.min(df["MeanPrecAnn"])
maxPrecip = np.max(df["MeanPrecAnn"])

tempNorm = mpl.colors.Normalize(vmin=minTemp, vmax=maxTemp)
gordNorm = mpl.colors.Normalize(vmin=minGord, vmax=maxGord)
precipNorm = mpl.colors.Normalize(vmin=minPrecip, vmax=maxPrecip)

tempCmap = cm.seismic
gordCmap = cm.PiYG
precipCmap = cm.RdYlGn

tempM = cm.ScalarMappable(norm=tempNorm, cmap=tempCmap)
gordM = cm.ScalarMappable(norm=gordNorm, cmap=gordCmap)
precipM = cm.ScalarMappable(norm=precipNorm, cmap=precipCmap)

xs = []
ys = []
zs = []

tempColor = []
gordColor = []
precipColor = []
for index, row in df.iterrows():
        xs.append(row["domf_slope"])
        ys.append(row["masd_slope"])
        zs.append(row["spectral_slope"])
        tempColor.append(tempM.to_rgba(row["MeanTempAnn"]))
        gordColor.append(gordM.to_rgba(row["gord"]))
        precipColor.append(precipM.to_rgba(row["MeanPrecAnn"]))


plt.scatter(x=df["domf_mean"], y=df["domf_slope"], c=tempColor)
plt.xlabel("day of mean flow")
plt.ylabel("day of mean flow slope")
plt.title("Timing of Flow")
plt.show()

plt.scatter(x=df["domf_mean"], y=df["domf_slope"], c=gordColor)
plt.xlabel("day of mean flow")
plt.ylabel("day of mean flow slope")
plt.title("Timing of Flow")
plt.show()

plt.scatter(df["masd_mean"], df["masd_slope"], c=tempColor)
plt.xlabel("mean annual specific discharge mean")
plt.ylabel("mean annual specific discharge slope")
plt.title("Amount of Flow")
plt.show()

plt.scatter(df["masd_mean"], df["masd_slope"], c=gordColor)
plt.xlabel("mean annual specific discharge mean")
plt.ylabel("mean annual specific discharge slope")
plt.title("Amount of Flow")
plt.show()

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs=xs,ys=ys,zs=zs, color=tempColor)#, label="Mean Annual Temperature")
ax.set_xlabel("day of mean flow slope")
#ax.set_xlabel("neural network encoding")
ax.set_ylabel("mean annual specific discharge slope")
ax.set_zlabel("spectral slope")
ax.set_title("Global Distribution of Flow Regime slopes")# - Mean Annual Temperature")# + str(predVar))
plt.show()

