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

plt.hist(frequencyData["encoding"])
plt.show()

plt.scatter(frequencyData["year"], frequencyData["encoding"], alpha=0.3)
plt.title("encodings across years")
plt.show()

# brief through-time analysis

totalDf = dayOfMeanFlowData.merge(meanAnnualSDData, on="catchment")

categories = []
for index, row in totalDf.iterrows():
    if row["gord"] < 6:
        if row["temp"] > 24:
            categories.append("hot and small")
        elif row["temp"] > 10:
            categories.append("med and small")
        else:
            categories.append("cold and small")
    elif row["gord"] >= 6:
        if row["temp"] > 24:
            categories.append("hot and large")
        elif row["temp"] > 10:
            categories.append("med and large")
        else:
            categories.append("cold and large")

totalDf["category"] = categories
totalDf = totalDf.merge(frequencyData, on="catchment")
totalDf.to_csv("spectralNumber_acrossTime.csv")
