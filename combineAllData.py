

# Catchment size

# do an even number of the smallest, mediumest, and largest catchments

# for each metric
# on each subset
# run gp regression between temperature and the metric
# calculate size of maximum error
# plot uncertainty / relationship



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
from scipy import stats
from scipy.stats.mstats import theilslopes as ts

numSplits = 3

adf = pd.read_csv("alldata.csv", encoding="latin-1")
cats = []
for cat in adf["grdc_no"]:
    try:
        cats.append("X" + str(int(cat)))
    except:
        cats.append(None)
adf["catchment"] = cats
print(adf["grdc_no"])
#quit()

def convertFromCubicMetersToLiters(array):
    array = np.asarray(array)
    array = array * 1000
    return array

# Get the Spectral Slope Data *****************************************************8

df = pd.read_csv("ml_slope_encodings_1.csv")

dataDict = {"catchment":[],"year":[],"encoding":[]}

for index, row in df.iterrows():
    catchment, year = row["catchment"].split("_")
    dataDict["catchment"].append(catchment)
    dataDict["year"].append(int(year))
    dataDict["encoding"].append(row["x"])

df = pd.DataFrame.from_dict(dataDict)
df = df[df["year"] < 2016]

years = list(set(df["year"]))
years.sort()
means = []
for year in years:
    ldf = df[df["year"] == year]
    mean = np.mean(ldf["encoding"])
    means.append(mean)

catchments = list(set(df["catchment"]))
catchments.sort()

spectralDict = {"catchment":[], "spectral_mean":[], "spectral_slope":[], "spectral_slope_normalized":[]}
for cat in catchments:
    ldf = df[df["catchment"] == cat]
    lyears = list(ldf["year"])
    lencodings = list(ldf["encoding"])
    if len(lyears) > 8:
        meanVal = np.mean(lencodings)
        #slope, intercept, rValue, pValue, stdErr = stats.linregress(lyears, lencodings)       
        slope, intercept, lo, up = ts(x=lyears, y=lencodings)
       # xs = np.linspace(np.min(lyears), np.max(lyears), 100)
       # ys = slope * xs + intercept
       # plt.scatter(lyears, lencodings)
       # plt.plot(xs, ys)
       # plt.show()
        
        spectralDict["catchment"].append(cat)
        spectralDict["spectral_mean"].append(meanVal)
        spectralDict["spectral_slope"].append(slope)
        spectralDict["spectral_slope_normalized"].append(100 * (slope / meanVal))

# add the spectral data from the full timeseries encoding
sdf = pd.read_csv("ml_encodings_1.csv")
sdf["spectral_full"] = sdf["x"]
sdf = sdf.drop("x", axis=1)

# Get the Mean Annual Specific Discharge Data ********************************

masddf = pd.read_csv("specific_discharge_vs_size.csv")
print(masddf)

cats = list(set(masddf["catchment"]))

masdDict = {"catchment":[], "masd_mean":[], "masd_slope":[], "masd_slope_normalized":[]}

for cat in cats:
    ldf = masddf[masddf["catchment"] == cat]
    years = list(ldf["year"])
    discharges = list(ldf["specific_discharge"])
    catTemps = list(ldf["temp"])
    catGords = list(ldf["gord"])
    if len(discharges) > 10:
        meanMasd = np.mean(convertFromCubicMetersToLiters(discharges))
#        slope, intercept, rValue, pValue, stdErr = stats.linregress(years, discharges)
        slope, intercept, lo, up = ts(x=years, y=discharges)
        masdDict["catchment"].append("X" + str(cat))
        masdDict["masd_mean"].append(meanMasd)
        masdDict["masd_slope"].append(slope)
        masdDict["masd_slope_normalized"].append(100 * (slope / meanMasd))

domfdf = pd.read_csv("dayOfMeanFlowThroughTime.csv")
cats = list(set(domfdf["catchment"]))
cats.sort()

domfDict = {
        "catchment":[],
        "domf_mean":[],
        "domf_slope":[],
        "domf_slope_normalized":[]
        }
for cat in cats:
    ldf = domfdf[domfdf["catchment"] == cat]
    years = ldf["year"]
    domf = ldf["day_of_mean_flow"]
    mean = np.mean(domf)
    #slope, intercept, rValue, pValue, stdErr = stats.linregress(years, domf)
    slope, intercept, lo, up = ts(x=years, y=domf)
    domfDict["catchment"].append(cat)
    domfDict["domf_mean"].append(mean)
    domfDict["domf_slope"].append(slope)
    domfDict["domf_slope_normalized"].append(100 * (slope / mean))

#print(masddataDict["catchment"][:10])
#print(catAndSlope["catchment"][:10])
#quit()
# merge all the data together

# spectral
df = pd.DataFrame.from_dict(spectralDict)
df = df.merge(adf, on="catchment")
df = df.merge(sdf, on="catchment")

# masd
masddf = pd.DataFrame.from_dict(masdDict)
df = masddf.merge(df, on="catchment")

# domf
domfdf = pd.DataFrame.from_dict(domfDict)
df = domfdf.merge(df, on="catchment")

df.to_csv("mergedData.csv", index=False)
print(df)
quit()
