import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

waterYearPrefix = "/home/sethbw/Documents/GlobFlow/localWaterYear/globFlowData_localWaterYear_"

variable = "gord"
figureTitle = "Large vs. Small Streams"
c1 = "m"
c2 = "g"
smallLabel = "average order: "
largeLabel = "average order: "

numToPlot = 5
def getMaskAndVec(vec):
    vec = pd.Series(vec)
    mask = ~vec.isna()
    mask = np.asarray(mask)
    vec = np.asarray(vec)
    return mask, vec


def divideByArea(flowData, area):
    print("flow data: " + str(flowData))
    print("area: " + str(area))
    newFlowData = []
    for i in range(len(flowData)):
        flow = flowData[i]
        #print(flow)
        newFlowData.append(float(flow) / float(area))
    return newFlowData


df = pd.read_csv(waterYearPrefix + "2004.csv")
df2 = pd.read_csv("FullDatabase.csv")

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


indices = np.argsort(df2[variable])
cats = np.asarray(df2["grdc_no"])
areas = np.asarray(df2["garea_sqkm"])
temps = np.asarray(df2[variable])

cats = cats[indices]
areas = areas[indices]
temps = temps[indices]


numSmallPlotted = 0
smallVars = []
i = 10
while numSmallPlotted < numToPlot:
    cat = cats[i]
    area = areas[i]
    temp = temps[i]

    mask, data = getMaskAndVec(df[cat])
    data = data[mask]
    data= divideByArea(data, area)

    if len(data) > 100:
        if numSmallPlotted == numToPlot - 1:
            smallVars.append(temp)
            plt.plot(days[mask], data, c=c1, alpha=0.8, label=smallLabel + str(np.mean(smallVars))[:6])
        else:
            smallVars.append(temp)
            plt.plot(days[mask], data, c=c1, alpha=0.8)

        numSmallPlotted += 1

    i = i + 1


numLargePlotted = 0
largeVars = []
i = len(temps) - 100
while numLargePlotted < numToPlot:
    cat = cats[i]
    area = areas[i]
    temp = temps[i]

    mask, data = getMaskAndVec(df[cat])
    data = data[mask]
    data= divideByArea(data, area)

    if len(data) > 100:
        if numLargePlotted == numToPlot - 1:
            largeVars.append(temp)
            plt.plot(days[mask], data, c=c2, alpha=0.8, label=largeLabel + str(np.mean(largeVars))[:6])
        else:
            largeVars.append(temp)
            plt.plot(days[mask], data, c=c2, alpha=0.8)

        numLargePlotted += 1

    i = i - 1

plt.legend()
plt.xlabel("day in water year 2003-04 (hemisphere corrected)")
plt.ylabel("specific discharge")
plt.title(figureTitle)
plt.show()
quit()
