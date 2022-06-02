import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from scipy import stats
import os
import random
import seaborn as sns


sns.set_context("poster")

alldf = pd.read_csv("alldata.csv")
cats = []
for cat in alldf["grdc_no"]:
    try:
        cats.append("X" + str(int(cat)))
    except:
        cats.append(None)
alldf["catchment"] = cats
print(alldf["grdc_no"])
for col in alldf.columns:
    print(col)
#print(adf)
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
plt.title("Global Distribution of Spectral Numbers")
plt.xlabel("spectral number")
plt.ylabel("count (# catchments)")
plt.show()

# divide into groups:
adf = df[df["encoding"] < -1.5]
bdf = df[(df["encoding"] > 0) & (df["encoding"] < 0.3)]
cdf = df[(df["encoding"] > 0.5) & (df["encoding"] < 0.8)]
ddf = df[(df["encoding"] > 1.5) & (df["encoding"] < 2)]
edf = df[df["encoding"] > 4]

# open up the original frequency decompositions
path = "/home/sethbw/Documents/GlobFlow/localWaterYear/"
yearTosDf = {}
for i in range(1988,2017):
    for ifile in os.listdir(path):
        if str(i) in ifile:
            if "localWaterYear" in ifile:
                df = pd.read_csv(path + ifile)
                print(df)
                yearTosDf[str(i)] = df

# open up the original hydrographs
path = "/home/sethbw/Documents/GlobFlow/localWaterYearSpectralDecomposition/"
yearTofDf = {}
for i in range(1988,2017):
    for ifile in os.listdir(path):
        if str(i) in ifile:
            if ("localWaterYear" in ifile) and ("FlowPowersTranspose" in ifile):
                df = pd.read_csv(path + ifile)
                yearTofDf[str(i)] = df

scale = list(yearTofDf["1988"].loc[0])[1:]
scale = [float(x) for x in scale]
print(scale)

def convertToSpecificDsicharge(array, cat):
    lldf = alldf[alldf["catchment"] == cat]
    size = float(lldf["garea_sqkm"])
    array = [float(x) for x in array]
    array = [(x / size) for x in array]
    return array

def getSpectrals(df):
    spectral = []
    encodings = []
    flow = []
    for index, row in df.iterrows():
        cat = row["catchment"]
        year = row["year"]
        encoding = row["encoding"]

        ydf = yearTofDf[str(year)]
        ydf = ydf[ydf[ydf.columns[0]] == cat]
        data = list(ydf.iloc[0])
        data = data[1:]
        data = [float(x) for x in data]
            
        scat = cat.replace("X","")
        sydf = yearTosDf[str(year)]
        sdata = list(sydf[scat])
        try:
            sdata = convertToSpecificDsicharge(sdata, cat)
            spectral.append(data)
            encodings.append(encoding)
            flow.append(sdata)
        except:
            pass

    return spectral, encodings, flow

aSpectral, aEncodings, aFlow = getSpectrals(adf)
bSpectral, bEncodings, bFlow = getSpectrals(bdf)
cSpectral, cEncodings, cFlow = getSpectrals(cdf)
dSpectral, dEncodings, dFlow = getSpectrals(ddf)
eSpectral, eEncodings, eFlow = getSpectrals(edf)

import random 
random.seed(100)

def makeFlowImage(flows, encodings, spectral, label):
    indices = random.choices(list(range(len(flows))), k=1)

    flows = np.asarray(flows)[indices]
    encodings = np.asarray(encodings)[indices]
    spectral = np.asarray(spectral)[indices]

    for flow in flows:
        plt.plot(flow, c="b")
#    plt.title("Example Hydrograph")
#    plt.xlabel("day in water year")
#    plt.ylabel("specific discharge")
    plt.show()

    for spec in spectral:
        plt.plot(scale, spec, c="orange")
#    plt.title("Example Spectral Decomposition")
#    plt.xlabel("peirod length (days)")
#    plt.ylabel("spectral power")
    plt.show()


makeFlowImage(aFlow, aEncodings, aSpectral, "a")
makeFlowImage(bFlow, bEncodings, bSpectral, "b")
makeFlowImage(cFlow, cEncodings, cSpectral, "c")
makeFlowImage(dFlow, dEncodings, dSpectral, "d")
makeFlowImage(eFlow, eEncodings, eSpectral, "e")


