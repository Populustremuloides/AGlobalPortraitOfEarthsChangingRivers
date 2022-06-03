import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

waterYearPrefix = "/home/sethbw/Documents/GlobFlow/localWaterYear/globFlowData_localWaterYear_"
spectralYearPrefix = "/home/sethbw/Documents/GlobFlow/localWaterYearSpectralDecomposition/globFlowData_localWaterYear_"

variable = "gord"
figureTitle = "Large vs. Small Streams"
c1 = "m"
c2 = "g"
smallLabel = "average order: "
largeLabel = "average order: "

numToPlot = 1
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
dfs = pd.read_csv(spectralYearPrefix + "2004_FlowPeriodPowers.csv")
df2 = pd.read_csv("FullDatabase.csv")

days = np.asarray(list(range(0, len(df[df.columns[0]]))))
scale = dfs["scale"]
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

numSmallPlanted = 0
smallVars = []
i = 10
cat1 = cats[i]
area = areas[i]
temp1 = temps[i]

mask, data = getMaskAndVec(df[cat1])
data = data[mask]
data= divideByArea(data, area)
print(cat1)
plt.plot(days[mask], data, c=c1, label="stream order = " + str(int(temp1)))


j = len(temps) - 100
cat2 = cats[j]
area = areas[j]
temp2 = temps[j]

mask, data = getMaskAndVec(df[cat2])
data = data[mask]
data= divideByArea(data, area)

print(cat2)
plt.plot(days[mask], data, c=c2, label="stream order = " + str(int(temp2)))
plt.legend()
plt.xlabel("day in water year 2003-04 (hemisphere corrected)")
plt.ylabel("specific discharge")
plt.title("Example Hydrographs")
plt.show()


cat1 = "X" + cat1
cat2 = "X" + cat2

spectral1 = dfs[cat1]
spectral2 = dfs[cat2]
plt.plot(scale, spectral1, c=c1, label="stream order = " + str(int(temp1)))
plt.plot(scale, spectral2, c=c2, label="stream order = " + str(int(temp2)))
plt.legend()
plt.title("Example Spectral Decompositions (same streams)")
plt.xlabel("period length (days)")
plt.ylabel("mean spectral power across time")
plt.show()


