import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

sd = pd.read_csv("specificDischargeThroughTime.csv")
mf = pd.read_csv("dayOfMeanFlowThroughTime.csv")

# ***************************************************
# The following code is added to use the real spectral slopes

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

catAndSlope = {"catchment":[], "spectral_slope":[], "spectral_mean":[]}
for cat in catchments:
    ldf = df[df["catchment"] == cat]
    lyears = list(ldf["year"])
    lencodings = list(ldf["encoding"])
    if len(lyears) > 8:
        slope, intercept, rValue, pValue, stdErr = stats.linregress(lyears,lencodings)
        catAndSlope["catchment"].append(cat)
        catAndSlope["spectral_slope"].append(slope)
        catAndSlope["spectral_mean"].append(np.mean(lencodings))

xdf = pd.DataFrame.from_dict(catAndSlope)
print(xdf)
#quit()
# ***************************************************
print(sd)
print(mf)

sdcats = list(set(sd["catchment"]))
sdcats.sort()
sdDict = {"catchment":[],"masd_slope":[], "masd_mean":[]}
for cat in sdcats:
    lds = sd[sd["catchment"] == cat]
    years = lds["year"]
    masds = lds["mean_annual_specific_discharge"]
    slope, intercept, rValue, pValue, stdErr = stats.linregress(years, masds)
    
    sdDict["masd_mean"].append(np.mean(masds))
    sdDict["catchment"].append(cat)
    sdDict["masd_slope"].append(slope)

sd = pd.DataFrame.from_dict(sdDict)




mfcats = list(set(mf["catchment"]))
mfcats.sort()
mfDict = {"catchment":[],"domf_slope":[], "domf_mean":[]}
for cat in mfcats:
    lds = mf[mf["catchment"] == cat]
    years = lds["year"]
    domfs = lds["day_of_mean_flow"]
    slope, intercept, rValue, pValue, stdErr = stats.linregress(years, domfs)

    mfDict["domf_mean"].append(np.mean(domfs))
    mfDict["catchment"].append(cat)
    mfDict["domf_slope"].append(slope)

mf = pd.DataFrame.from_dict(mfDict)

totalDf = mf.merge(sd, on="catchment")
totalDf = totalDf.merge(xdf, on="catchment")
#totalDf = totalDf.merge(x_static, on="catchment")

totalDf.to_csv("throughTimeCombined.csv", index=False)

