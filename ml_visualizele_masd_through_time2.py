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


gdf = df[df["slopes"] < 0.001]
gdf2 = df[df["slopes"] > -0.001]
df = pd.merge(gdf, gdf2, how="inner", on=["slopes"])
plt.hist(df["slopes"], bins=100)
print(np.mean(df["slopes"]))
print(np.min(df["slopes"]))
print(np.max(df["slopes"]))
plt.show()
#quit()
norm = mpl.colors.Normalize(vmin=(-1 * np.max(df["slopes"])), vmax=(1 * np.max(df["slopes"])))
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

# keep the middle of the distribution separate
ldf = df[df["slopes"] < 0.0005]
ldf = ldf[ldf["slopes"] > -0.0001]
bdf = df[df["slopes"] > 0.0005]
bdf2 = df[df["slopes"] < -0.0001]
bdf = bdf.append(bdf2)

lcolors = []
for index, row in ldf.iterrows():
    cVar = row["slopes"]
    color = m.to_rgba(-1 * cVar)
    lcolors.append(color)

bcolors = []
for index, row in bdf.iterrows():
    cVar = row["slopes"]
    color = m.to_rgba(-1 * cVar)
    bcolors.append(color)

fig, ax = plt.subplots(figsize=(2, 6))
#fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label("slope")
plt.show()

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)


print(bdf)
print(ldf)
#quit()

plt.scatter(x=ldf["new_lon.x_x"], y=ldf["new_lat.x_x"], c=lcolors, s=5, alpha=0.9)
plt.scatter(x=bdf["new_lon.x_x"], y=bdf["new_lat.x_x"], c=bcolors, s=5, alpha=0.9)
#plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, edgecolor="black", s=.1, alpha=1)
plt.title("Slopes of Mean Annual Specific Discharge")
plt.show()

#cb1.set_label('Degrees Celsius')
#cb1.set_label(" slope")
plt.show()

quit()

logSlopes = logData(oDf["slopes"])
oDf["log_slopes"] = logSlopes
#plt.scatter(df[])


minSlope = -0.08 #np.min(oDf[colorVar])
maxSlope = 0.08 #np.max(oDf[colorVar])
norm = mpl.colors.Normalize(vmin=minSlope, vmax=maxSlope)
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

print(oDf)
colors = []
for index, row in oDf.iterrows():
    cVar = row[colorVar]
    color = m.to_rgba(cVar)
    colors.append(color)

plt.hist(oDf[colorVar], bins=100)
plt.title("Change in Mean Annual Specific Discharge (sqrt slope)")
plt.xlabel("log slope")
plt.ylabel("count")
plt.show()
plt.scatter(oDf["temps"], oDf["log_slopes"], c=colors, alpha=0.5)
plt.title("Change in Mean Annual Specific Discharge (sqrt slope)")
plt.xlabel("temperature (degress C)")
plt.ylabel("log slope")
plt.show()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cmap = mpl.cm.seismic
norm = mpl.colors.Normalize(vmin=minSlope, vmax=maxSlope)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
#cb1.set_label('Degrees Celsius')
cb1.set_label("sqrt slope")
plt.show()
fig.show()


