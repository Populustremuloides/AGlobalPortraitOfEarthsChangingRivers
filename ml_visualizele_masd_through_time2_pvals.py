import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import matplotlib.cm as cm
import matplotlib as mpl
import math

colorVar = "gords"

df = pd.read_csv("specific_discharge_vs_size.csv")
print(df)

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
        "catchment":[],
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
        
        dataDict["catchment"].append(cat)
        dataDict["slopes"].append(slope)
        dataDict["rVals"].append(r_value)
        dataDict["pVals"].append(p_value)
        dataDict["temps"].append(catTemps[0])
        dataDict["gords"].append(catGords[0])


oDf = pd.DataFrame.from_dict(dataDict)

df = df.merge(oDf, on="catchment")
for col in df.columns:
    print(col)
print("*****************************************")
df = df.groupby(["catchment"], as_index=False).mean()

for col in df.columns:
    print(col)

adf = pd.read_csv("alldata.csv")
cats = []
for cat in adf["grdc_no"]:
    try:
        cats.append(str(int(cat)))
    except:
        cats.append(None)

adf["catchment"] = cats
cats = []
for cat in df["catchment"]:
    try:
        cats.append(str(int(cat)))
    except:
        cats.append(None)
df["catchment"] = cats
df = df.merge(adf, on="catchment")

print(df)

for col in df.columns:
    print(col)


#gdf = df[df["slopes"] < 0.001]
#gdf2 = df[df["slopes"] > -0.001]
#df = pd.merge(gdf, gdf2, how="inner", on=["slopes"])
plt.hist(df["pVals"], bins=100)
plt.title("Distribution of P-Values for Mean Annual Specific Discahrge Slopes")
print(np.mean(df["pVals"]))
print(np.min(df["pVals"]))
print(np.max(df["pVals"]))
plt.xlabel("p-value")
plt.ylabel("count")
plt.show()
#quit()
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

# keep the middle of the distribution separate
#ldf = df[df["slopes"] < 0.0005]
#ldf = ldf[ldf["slopes"] > -0.0001]
#bdf = df[df["slopes"] > 0.0005]
#bdf2 = df[df["slopes"] < -0.0001]
#bdf = bdf.append(bdf2)

#lcolors = []
#for index, row in ldf.iterrows():
#    cVar = row["slopes"]
#    color = m.to_rgba(-1 * cVar)
#    lcolors.append(color)

#bcolors = []
#for index, row in bdf.iterrows():
#    cVar = row["slopes"]
#    color = m.to_rgba(-1 * cVar)
#    bcolors.append(color)

colors = []
for index, row in df.iterrows():
    cVar = row["pVals"]
    color = m.to_rgba(cVar)
    colors.append(color)


fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label("pValue")
plt.show()

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)

print(df)
#quit()

plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, s=1, alpha=0.9)
plt.title("P-Values of Mean Annual Specific Discharge Slopes")
plt.show()

#cb1.set_label('Degrees Celsius')
#cb1.set_label(" slope")
plt.show()

quit()


