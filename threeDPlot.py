# ASSUMING STATIC -- THIS IS NOT CHANGE THROUGH TIME

import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats as stats

def fixCats(cats):
    outCats = []
    for cat in cats:
        outCats.append("X" + str(int(cat)))
    return outCats

def meanDataFrame(df):
    df = df.drop("variable", axis=1)
    outDict = {}
    cols = list(df.columns)
    for col in cols:
        outDict[col] = [np.mean(df[col])]
    outDf = pd.DataFrame.from_dict(outDict)
    return outDf

# import the day of mean flow stuff
dayOfMeanFlowData = pd.read_csv("day_of_mean_flow_vs_size.csv")

cats = list(set(dayOfMeanFlowData["catchment"]))
cats.sort()
i = 0
for cat in cats:
    ldf = dayOfMeanFlowData[dayOfMeanFlowData["catchment"] == cat]
    ldf = meanDataFrame(ldf)
    if i == 0:
        meanDayOfMeanFlowData = ldf
    else:
        meanDayOfMeanFlowData = meanDayOfMeanFlowData.append(ldf)
    i = i + 1
dayOfMeanFlowData = meanDayOfMeanFlowData
dayOfMeanFlowData["catchment"] = fixCats(dayOfMeanFlowData["catchment"])

# import the mean annual specific discharge stuff
meanAnnualSDData = pd.read_csv("specific_discharge_vs_size.csv")
cats = list(set(meanAnnualSDData["catchment"]))
cats.sort()
i = 0
for cat in cats:
    ldf = meanAnnualSDData[meanAnnualSDData["catchment"] == cat]
    ldf = meanDataFrame(ldf)
    if i == 0:
        meanMeanAnnualSDData = ldf
    else:
        meanMeanAnnualSDData = meanMeanAnnualSDData.append(ldf)
    i = i + 1
meanAnnualSDData = meanMeanAnnualSDData
meanAnnualSDData["catchment"] = fixCats(meanAnnualSDData["catchment"])
meanAnnualSDData = meanAnnualSDData.drop("precip", axis=1)
meanAnnualSDData = meanAnnualSDData.drop("temp", axis=1)
meanAnnualSDData = meanAnnualSDData.drop("gord", axis=1)
meanAnnualSDData = meanAnnualSDData.drop("year", axis=1)

# import the single dimensional neural network encoding
#frequencyData = pd.read_csv("ml_encodings_1.csv")

frequencyDf = pd.read_csv("ml_slope_encodings_1.csv")
dataDict = {"catchment":[],"year":[],"encoding":[]}
for index, row in frequencyDf.iterrows():
    catchment, year = row["catchment"].split("_")
    dataDict["catchment"].append(catchment)
    dataDict["year"].append(int(year))
    dataDict["encoding"].append(row["x"])
frequencyData = pd.DataFrame.from_dict(dataDict)

dataDict = {"catchment":[],"mean_encoding":[]}
fcats= list(set(frequencyData["catchment"]))
fcats.sort()
for cat in fcats:
    lfdf = frequencyData[frequencyData["catchment"] == cat]
    meanEncoding = np.mean(lfdf["encoding"]) 
    dataDict["catchment"].append(cat)
    dataDict["mean_encoding"].append(meanEncoding)
frequencyData = pd.DataFrame.from_dict(dataDict)

#print(frequencyData)
#quit()
# USE THIS TO 
frequencyData = pd.read_csv("ml_encodings_1.csv")

#frequencyData = frequencyData.drop(frequencyData.columns[0], axis=1)
print(dayOfMeanFlowData)
print(meanAnnualSDData)
print(frequencyData)

totalDF = dayOfMeanFlowData.merge(meanAnnualSDData, on="catchment")
totalDF = totalDF.merge(frequencyData, on="catchment")

logPrecip = []
for prec in totalDF["precip"]:
    try:
        logPrecip.append(math.log(prec + 1))
    except:
        logPrecip.append(None)
totalDF["log_precip"] = logPrecip
totalDF = totalDF.dropna()
totalDF.to_csv("three_d_df.cvs", index=False)
print(totalDF)
#quit()
# collect all the catchments for each one

x = [] # neural network encoding
y = [] # day of mean flow
z = [] # mean annual specific discharge
tempColor = []
gordColor = []
precipColor = []

minTemp = np.min(totalDF["temp"])
maxTemp = np.max(totalDF["temp"])

minGord = np.min(totalDF["gord"])
maxGord =  np.max(totalDF["gord"])

minPrecip = np.min(totalDF["log_precip"])
maxPrecip = np.max(totalDF["log_precip"])

tempNorm = mpl.colors.Normalize(vmin=minTemp, vmax=maxTemp)
gordNorm = mpl.colors.Normalize(vmin=minGord, vmax=maxGord)
precipNorm = mpl.colors.Normalize(vmin=minPrecip, vmax=maxPrecip)

tempCmap = cm.seismic
gordCmap = cm.PiYG
precipCmap = cm.RdYlGn

tempM = cm.ScalarMappable(norm=tempNorm, cmap=tempCmap)
gordM = cm.ScalarMappable(norm=gordNorm, cmap=gordCmap)
precipM = cm.ScalarMappable(norm=precipNorm, cmap=precipCmap)

for index, row in totalDF.iterrows():
        #x.append(row["mean_encoding"])
        x.append(row["x"])
        y.append(row["day_of_mean_flow"])
        z.append(row["specific_discharge"])
        tempColor.append(tempM.to_rgba(row["temp"]))
        gordColor.append(gordM.to_rgba(row["gord"]))
        precipColor.append(precipM.to_rgba(row["log_precip"]))

print(x)
print(y)
print(z)



font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=x, y=y, color=gordColor)
plt.xlabel("spectral number")
plt.ylabel("day of mean flow")
plt.show()

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=x, y=z, color=gordColor)
plt.xlabel("spectral number")
plt.ylabel("mean annual specific discharge")
plt.show()

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=y, y=z, color=gordColor)
plt.xlabel("day of mean flow")
plt.ylabel("mean annual specific discharge")
plt.show()



fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs=x,ys=y,zs=z, color=gordColor)#, label="Mean Annual Temperature")
ax.set_xlabel("spectral number")
#ax.set_xlabel("neural network encoding")
ax.set_ylabel("day of mean flow")
ax.set_zlabel("mean annual specific discharge")
plt.title("Global Distribution of Flow Regimes")# - Mean Annual Temperature")# + str(predVar))
plt.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=gordCmap,
                                norm=gordNorm,
                                orientation='horizontal')
#cb1.set_label('Degrees Celsius')
cb1.set_label("stream order")
plt.show()




plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=x, y=y, color=precipColor)
plt.xlabel("spectral number")
plt.ylabel("day of mean flow")
plt.show()

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=x, y=z, color=precipColor)
plt.xlabel("spectral number")
plt.ylabel("mean annual specific discharge")
plt.show()

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=y, y=z, color=precipColor)
plt.xlabel("day of mean flow")
plt.ylabel("mean annual specific discharge")
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs=x,ys=y,zs=z, color=precipColor)#, label="Mean Annual Temperature")
ax.set_xlabel("spectral number")
#ax.set_xlabel("neural network encoding")
ax.set_ylabel("day of mean flow")
ax.set_zlabel("mean annual specific discharge")
plt.title("Global Distribution of Flow Regimes")# - Mean Annual Temperature")# + str(predVar))
plt.show()


# color legend
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=precipCmap,
                                norm=precipNorm,
                                orientation='horizontal')
#cb1.set_label('Degrees Celsius')
cb1.set_label("log precipitation")
plt.show()

# store in x, y, z values

# color according to temperature

# color according to gord
plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=x, y=y, color=tempColor)
plt.xlabel("spectral number")
plt.ylabel("day of mean flow")
plt.show()

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=x, y=z, color=tempColor)
plt.xlabel("spectral number")
plt.ylabel("mean annual specific discharge")
plt.show()

plt.title("Global Distribution of Flow Regimes")
plt.scatter(x=y, y=z, color=tempColor)
plt.xlabel("day of mean flow")
plt.ylabel("mean annual specific discharge")
plt.show()



fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xs=x,ys=y,zs=z, color=tempColor)#, label="Mean Annual Temperature")
ax.set_xlabel("spectral number")
#ax.set_xlabel("neural network encoding")
ax.set_ylabel("day of mean flow")
ax.set_zlabel("mean annual specific discharge")
plt.title("Global Distribution of Flow Regimes")# - Mean Annual Temperature")# + str(predVar))
plt.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=tempCmap,
                                norm=tempNorm,
                                orientation='horizontal')
#cb1.set_label('Degrees Celsius')
cb1.set_label("mean annual temperature")
plt.show()


