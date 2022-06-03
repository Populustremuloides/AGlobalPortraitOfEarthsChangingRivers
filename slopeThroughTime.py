import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib as mpl
import math

colorVar = "slopes"

df = pd.read_csv("specific_discharge_vs_size.csv")
cats = list(set(df["catchment"]))

def logData(data):
    loggedData = []
    for i in range(len(data)):
        if data[i] < 0:
            loggedData.append(-(math.sqrt(-data[i])))
        else:
            loggedData.append(math.sqrt(data[i]))
    return loggedData

dataDict = {
        "slopes":[],
        "rVals":[],
        "pVals":[],
        "temps":[],
        "gords":[]
        }


for cat in cats:
    ldf = df[df["catchment"] == cat]
    years = list(ldf["year"])
    discharges = list(ldf["specific_discharge"])
    catTemps = list(ldf["temp"])
    catGords = list(ldf["gord"])
    if len(discharges) > 10:
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, discharges)

        dataDict["slopes"].append(slope)
        dataDict["rVals"].append(r_value)
        dataDict["pVals"].append(p_value)
        dataDict["temps"].append(catTemps[0])
        dataDict["gords"].append((random.random() * 0.8)+ catGords[0])


oDf = pd.DataFrame.from_dict(dataDict)

logSlopes = logData(oDf["slopes"])
oDf["log_slopes"] = logSlopes

minSlope = -0.001 #-0.08 #np.min(oDf[colorVar])
maxSlope = 0.001  #0.08 #np.max(oDf[colorVar])
norm = mpl.colors.Normalize(vmin=minSlope, vmax=maxSlope)
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

colors = []
for index, row in oDf.iterrows():
    cVar = row[colorVar]
    color = m.to_rgba(cVar)
    colors.append(color)

plt.hist(oDf[colorVar], bins=100)
plt.title("Change in Mean Annual Specific Discharge")
plt.xlabel("slope")
plt.ylabel("count")
plt.show()
plt.scatter(oDf["gords"], oDf["temps"], c=colors, alpha=0.5)
plt.title("Change in Mean Annual Specific Discharge")
plt.ylabel("mean annual temperature (degress C)")
plt.xlabel("stream order (with jitter)")
plt.show()

fig, ax = plt.subplots(figsize=(1.7, 6))
#fig.subplots_adjust(bottom=0.5)
cmap = mpl.cm.seismic
norm = mpl.colors.Normalize(vmin=minSlope, vmax=maxSlope)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
#cb1.set_label('Degrees Celsius')
cb1.set_label("slope")
plt.show()
fig.show()
