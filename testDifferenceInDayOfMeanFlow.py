import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math

waterYearPrefix = "/home/sethbw/Documents/GlobFlow/localWaterYear/globFlowData_localWaterYear_"

variable = "gord"
figureTitle = "Large vs. Small Streams"
c1 = "g"
c2 = "m"
smallLabel = "average order: "
largeLabel = "average order: "

numToPlot = 4

smallType = "small stream"
largeType = "large stream"

dataDict = {
        "variable":[],
        "gord":[],
        "temp":[],
        "precip":[],
        "log_precip":[],
        "year":[],
        "catchment":[],
        "day_of_mean_flow":[]
        }



def getCenterOfMass(flowData):
    dayInYear = list(range(1,len(flowData) + 1))

    numerator = np.sum(np.multiply(flowData, dayInYear))
    denominator = np.sum(flowData)

    if denominator == 0.0:
        centerOfMass = np.sum(dayInYear) / len(dayInYear)
    else:
        centerOfMass = numerator / denominator
    return centerOfMass



def getMaskAndVec(vec):
    vec = pd.Series(vec)
    mask = ~vec.isna()
    mask = np.asarray(mask)
    vec = np.asarray(vec)
    return mask, vec


def divideByArea(flowData, area):
    #print("flow data: " + str(flowData))
    #print("area: " + str(area))
    newFlowData = []
    for i in range(len(flowData)):
        flow = flowData[i]
        #print(flow)
        newFlowData.append(float(flow) / float(area))
    return newFlowData


df2 = pd.read_csv("FullDatabase.csv")


year = 2004

for year in range(1988,2016):
    df = pd.read_csv(waterYearPrefix + str(year) + ".csv")

    days = np.asarray(list(range(0, len(df[df.columns[0]]))))

    #for col in df2.columns:
    #    print(col)
    #quit()
    newCats = []
    cats = df2["grdc_no"]
    for cat in cats:
        try:
            newCats.append(str(int(cat)))
        except:
            newCats.append(None)
    df2["grdc_no"] = newCats

    waterCats = list(set(df.columns))
    df2 = df2[df2["grdc_no" ].isin(waterCats)]
    df2 = df2[~df2[variable].isna()]
    df2 = df2[~df2["MeanTempAnn"].isna()]
    df2 = df2[~df2["MeanPrecAnn"].isna()]

    indices = np.argsort(df2[variable])
    cats = np.asarray(df2["grdc_no"])
    areas = np.asarray(df2["garea_sqkm"])
    temps = np.asarray(df2["MeanTempAnn"])
    precips = np.asarray(df2["MeanPrecAnn"])
    variableVals = np.asarray(df2[variable])

    cats = cats[indices]
    areas = areas[indices]
    temps = temps[indices]
    precips = precips[indices]
    variableVals = variableVals[indices]

    numSmallRecorded = 0
    smallVars = []
    i = 0
    for i in range(len(cats)):
        cat = cats[i]
        area = areas[i]
        var = variableVals[i]
        temp = temps[i]
        precip = precips[i]

        mask, data = getMaskAndVec(df[cat])
        data = data[mask]
        data= divideByArea(data, area)

        if len(data) >= 365:
            try:
                dataDict["log_precip"].append(math.log(precip + 1))
                dataDict["variable"].append(variable)
                dataDict["gord"].append(var)
                dataDict["temp"].append(temp)
                dataDict["precip"].append(precip)
                dataDict["year"].append(year)
                dataDict["catchment"].append(cat)
                dataDict["day_of_mean_flow"].append(getCenterOfMass(data))
                numSmallRecorded += 1
            except:
                pass

        i = i + 1


outDf = pd.DataFrame.from_dict(dataDict)
outDf.to_csv("day_of_mean_flow_vs_size.csv", index=False)
quit()


plt.scatter(outDf["stream variable val"],outDf["specific discharge"], alpha=0.1)
plt.show()


sns.boxplot(data=outDf, x="stream variable val", y="specific discharge", hue="variable")
plt.show()
print(outDf)
quit()







