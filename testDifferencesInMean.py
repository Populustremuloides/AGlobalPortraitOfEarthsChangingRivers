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
        "cultivated_and_managed_vegetation":[],
        "urban":[],
        "dam_count":[],
        "year":[],
        "catchment":[],
        "specific_discharge":[]
        }


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

    df2 = df2[~df2["cls7"].isna()]
    df2 = df2[~df2["cls9"].isna()]
    df2 = df2[~df2["Dam_Count"].isna()]

    indices = np.argsort(df2[variable])
    cats = np.asarray(df2["grdc_no"])
    areas = np.asarray(df2["garea_sqkm"])
    temps = np.asarray(df2["MeanTempAnn"])
    precips = np.asarray(df2["MeanPrecAnn"])
    variableVals = np.asarray(df2[variable])

    cls7s = np.asarray(df2["cls7"]) 
    cls9s = np.asarray(df2["cls9"])    
    Dam_Counts = np.asarray(df2["Dam_Count"])    

    cats = cats[indices]
    areas = areas[indices]
    temps = temps[indices]
    precips = precips[indices]
    variableVals = variableVals[indices]
    
    cls7s = cls7s[indices]
    cls9s = cls9s[indices]
    Dam_Counts = Dam_Counts[indices]

    numSmallRecorded = 0
    smallVars = []
    i = 0
    for i in range(len(cats)):
        cat = cats[i]
        area = areas[i]
        var = variableVals[i]
        temp = temps[i]
        precip = precips[i]

        cls7 = cls7s[i]
        cls9 = cls9s[i]
        Dam_Count = Dam_Counts[i]

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
                dataDict["cultivated_and_managed_vegetation"].append(cls7)
                dataDict["urban"].append(cls9)
                dataDict["dam_count"].append(Dam_Count)
                dataDict["year"].append(year)
                dataDict["catchment"].append(cat)
                dataDict["specific_discharge"].append(np.mean(data))
                numSmallRecorded += 1
            except:
                pass

        i = i + 1


outDf = pd.DataFrame.from_dict(dataDict)
outDf.to_csv("specific_discharge_vs_size.csv", index=False)


