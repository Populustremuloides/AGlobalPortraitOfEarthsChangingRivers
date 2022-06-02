import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import math
import random
from scipy import stats

numDimensions = 2

df = pd.read_csv("dayOfMeanFlowThroughTime.csv")
cats = list(set(df["catchment"]))
cats.sort()

catToSlope = {
        "catchment":[],
        "dayOfMeanFlow_slope":[],
        "dayOfMeanFlow_pValue":[]
        }
for cat in cats:
    ldf = df[df["catchment"] == cat]
    years = ldf["year"]
    domf = ldf["day_of_mean_flow"]
    slope, intercept, rValue, pValue, stdErr = stats.linregress(years, domf)
    catToSlope["catchment"].append(cat)
    catToSlope["dayOfMeanFlow_slope"].append(slope)
    catToSlope["dayOfMeanFlow_pValue"].append(pValue)

df = pd.DataFrame.from_dict(catToSlope)

alldata = pd.read_csv("alldata.csv", encoding="latin-1")
for column in alldata.columns:
    print(column) # fix alldata
catchments = []
for cat in alldata["grdc_no"]:
    try:
        catchments.append("X" + str(int(cat)))
    except:
        catchments.append(None)
alldata["catchment"] = catchments

df = df.merge(alldata, on="catchment")
print(df)
#quit()
# make a figure plotting catchment size against meanTempAnn, colored according to one-d

#maxVal = np.max([-1 * np.min(df["dayOfMeanFlow_slope"]), np.max(df["dayOfMeanFlow_slope"])])

minVar = np.min(df["dayOfMeanFlow_pValue"])
maxVar = np.max(df["dayOfMeanFlow_pValue"])

norm = mpl.colors.Normalize(vmin=0, vmax=1) # keep it centered on 0
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

print(minVar)
print(maxVar)
#quit()
colors = []
xs = []
ys = []
#zs = []
reds = []
blues = []
redMeans = []
blueMeans = []
for index, row in df.iterrows():
    if not math.isnan(row["new_lat.x"]) and not math.isnan(row["new_lon.x"]):
        #r, g, b, a = m.to_rgba(row["dayOfMeanFlow_slope"])
        #if r == 0 and g == 0 and b == 0 and a == 0:
        #    print(row[colorVar])
        colors.append(m.to_rgba(row["dayOfMeanFlow_pValue"]))
        xs.append(row["new_lat.x"])
        ys.append(row["new_lon.x"])

plt.hist(df["dayOfMeanFlow_pValue"], bins=100)
plt.title("Distribution of P-Values for Day of Mean Flow Slopes")
plt.xlabel("p-value")
plt.ylabel("slope")
plt.show()
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, s=1, alpha=0.9)
plt.title("P-Values of Day of Mean Flow Slopes")
plt.show()

norm = mpl.colors.Normalize(vmin=minVar, vmax=maxVar)
cmap = cm.seismic
m = cm.ScalarMappable(norm=norm, cmap=cmap)

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
#cb1.set_label('Degrees Celsius')
cb1.set_label("p value")
plt.show()

