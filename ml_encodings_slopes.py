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

catAndSlope = {"catchment":[], "spectral_slope":[]}
for cat in catchments:
    ldf = df[df["catchment"] == cat]
    lyears = list(ldf["year"])
    lencodings = list(ldf["encoding"])
    if len(lyears) > 8:
        slope, intercept, rValue, pValue, stdErr = stats.linregress(lyears,lencodings)
        catAndSlope["catchment"].append(cat)
        catAndSlope["spectral_slope"].append(slope)

df = pd.DataFrame.from_dict(catAndSlope)
df = df.merge(adf, on="catchment")

plt.hist(df["spectral_slope"], bins=100)
plt.show()
print(df)

# PLOT ON A WORLD MAP

norm = mpl.colors.Normalize(vmin=(-1 * np.max(df["spectral_slope"])), vmax=(1 * np.max(df["spectral_slope"])))
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

colors = []
for slope in df["spectral_slope"]:
    colors.append(m.to_rgba(slope))

fig, ax = plt.subplots(figsize=(1.5, 6))
#fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
cb1.set_label("slope")
#plt.title("Slope")
plt.show()

import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, s=5, alpha=0.9)
plt.title("Slopes of Spectral Numbers")
plt.show()

