import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from scipy import stats

adf = pd.read_csv("alldata.csv")
cats = []
for cat in adf["grdc_no"]:
    try:
        cats.append("X" + str(int(cat)))
    except:
        cats.append(None)
adf["catchment"] = cats
print(adf["grdc_no"])
#quit()

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
plt.show()

years = list(set(df["year"]))
years.sort()
means = []
for year in years:
    ldf = df[df["year"] == year]
    mean = np.mean(ldf["encoding"])
    means.append(mean)
plt.plot(years, means)
plt.show()

catchments = list(set(df["catchment"]))
catchments.sort()

catAndSlope = {"catchment":[], "spectral_mean":[]}
for cat in catchments:
    ldf = df[df["catchment"] == cat]
    lyears = list(ldf["year"])
    lencodings = list(ldf["encoding"])
    if len(lyears) > 8:
        #slope, intercept, rValue, pValue, stdErr = stats.linregress(lyears,lencodings)

        catAndSlope["catchment"].append(cat)
        catAndSlope["spectral_mean"].append(np.mean(lencodings))

df = pd.DataFrame.from_dict(catAndSlope)
df = df.merge(adf, on="catchment")

plt.hist(df["spectral_mean"], bins=100)
plt.show()
print(df)

# PLOT ON A WORLD MAP
# we don't go as negative as we could for the lower bound to keep contrast (the Amazon catchments really go negative)
norm = mpl.colors.Normalize(vmin=(-1 * np.max(df["spectral_mean"])), vmax=(1 * np.max(df["spectral_mean"])))
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

colors = []
for slope in df["spectral_mean"]:
    colors.append(m.to_rgba(slope))

fig, ax = plt.subplots(figsize=(1.5, 6))
#fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label("Spectral Number")
#plt.title("Slope")
plt.show()

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, s=5, alpha=0.9)
plt.title("Spectral Numbers")
plt.show()


colors = []
colorVals = []
for index, row in df.iterrows():
    cVar = row["spectral_mean"]
    color = m.to_rgba(-1 * cVar)
    colors.append(color)

fig, ax = plt.subplots(figsize=(1.5, 6))
#fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')

cb1.set_label("mean annual spectral number")
plt.show()
print(df)
for col in df.columns:
    print(col)
#quit()
ndf = df[~df["gord"].isna()]
gords = list(ndf["gord"])
temps = list(ndf["MeanTempAnn"])
precips = list(ndf["MeanPrecAnn"])
noise = np.random.normal(size = len(gords)) * 0.17
gords = np.add(gords, noise)
ncolors = []
for index, row in ndf.iterrows():
    ncolors.append(m.to_rgba(row["spectral_mean"]))
plt.scatter(precips, temps, color=ncolors, alpha=1, s=2)
plt.xlabel("mean annual precipitation")
plt.ylabel("mean annual temperature")
plt.title("Mean Spectral Number across Spatial Features")
plt.show()


