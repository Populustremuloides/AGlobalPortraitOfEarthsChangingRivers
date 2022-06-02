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
        "mean_masd":[],
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
        meanMasd= np.mean(discharges)
        
        dataDict["catchment"].append(cat)
        dataDict["mean_masd"].append(meanMasd)
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


#bdf = df[df["mean_masd"] < 0.05]
#ldf = df[df["mean_masd"] >= 0.05]
#ldf["mean_masd"] = [0.05] * len(list(ldf["mean_masd"]))
#df = bdf.append(ldf)
#df = bdf
plt.hist(df["mean_masd"], bins=100)
print(np.mean(df["mean_masd"]))
print(np.min(df["mean_masd"]))
print(np.max(df["mean_masd"]))

plt.show()
#quit()
norm = mpl.colors.Normalize(vmin=-0.05, vmax=0)
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

# keep the middle of the distribution separate
#ldf = df[df["mean_masd"] < 0.0005]
#ldf = ldf[ldf["mean_masd"] > -0.0001]
#bdf = df[df["mean_masd"] > 0.0005]
#bdf2 = df[df["mean_masd"] < -0.0001]
#bdf = bdf.append(bdf2)

colors = []
colorVals = []
for index, row in df.iterrows():
    cVar = row["mean_masd"]
    color = m.to_rgba(-1 * cVar)
    colors.append(color)

fig, ax = plt.subplots(figsize=(1.5, 6))
#fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label("mean annual specific discharge")
plt.show()
print(df)
for col in df.columns:
    print(col)
#quit()
ndf = df[~df["gord_y"].isna()]
gords = list(ndf["gord_y"])
temps = list(ndf["MeanTempAnn"])
precips = list(ndf["MeanPrecAnn"])
noise = np.random.normal(size = len(gords)) * 0.17
gords = np.add(gords, noise)
ncolors = []
for index, row in ndf.iterrows():
    ncolors.append(m.to_rgba(-1 * row["mean_masd"]))

plt.scatter(precips, temps, color=ncolors, alpha=1, s=2)
plt.xlabel("mean annual precipitation")
plt.ylabel("mean annual temperature")
plt.title("Mean Annual Specific Discharge across Spatial Features")
plt.show()

plt.scatter(gords, temps, color=ncolors, alpha=0.5)
plt.xlabel("stream order (with jitter)")
plt.ylabel("mean annual temperature")
plt.title("Mean Annual Specific Discharge across Spatial Features")
plt.show()


import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)


plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, s=10, alpha=0.5)
plt.title("Mean Annual Specific Discharge")
plt.show()

#cb1.set_label('Degrees Celsius')
#cb1.set_label(" slope")

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
    color = m.to_rgba(-1 * cVar)
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


