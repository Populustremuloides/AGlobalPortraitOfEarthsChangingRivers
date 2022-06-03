import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math


outDir = "gif_figures/"

# get all catchments for all time (get all the dfs)
# assign each one a number
# for each year
    # for each day
        # for each catchment
            # plot the flow (color according to stream order, temperature, and log precip)


waterYearPrefix = "/home/sethbw/Documents/GlobFlow/localWaterYear/globFlowData_localWaterYear_"

df2 = pd.read_csv("FullDatabase.csv")
newCats = []
cats = df2["grdc_no"]
for cat in cats:
    try:
        newCats.append("X" + str(int(cat)))
    except:
        newCats.append(None)

df2["grdc_no"] = newCats
df2 = df2[~df2["gord"].isna()]
df2 = df2[~df2["MeanTempAnn"].isna()]
df2 = df2[~df2["MeanPrecAnn"].isna()]
df2 = df2[~df2["garea_sqkm"].isna()]

catsTotal = []
yearToFlowDf = {}
print("reading data frames")
for year in range(1988,2017):
    print(year)
    df = pd.read_csv(waterYearPrefix + str(year) + ".csv")
    yearToFlowDf[year] = df
    catsTotal = catsTotal + list(df.columns[1:])

catsTotal = list(set(catsTotal))
catsTotal.sort()
print(catsTotal)
xcats = []
for cat in catsTotal:
    xcats.append("X" + str(cat))
catsTotal = xcats

catsTotal = set(catsTotal)
catsData = set(df2["grdc_no"])
catsTotal = list(catsTotal.intersection(catsData))
catsTotal.sort()
print(len(catsTotal))

df2 = df2[df2["grdc_no"].isin(catsTotal)]

# get catToArea
catToArea = {}
for index, row in df2.iterrows():
    cat = row["grdc_no"]
    area = row["garea_sqkm"]
    catToArea[cat] = area

#print(df2)

tempSort = np.argsort(df2["MeanTempAnn"])
areaSort = np.argsort(df2["garea_sqkm"])
precipSort = np.argsort(df2["MeanPrecAnn"])


print("making figures")
for year in range(1988, 2017):
    print("year: " + str(year) + " ******************************")
    for day in range(0,365): # make an image for every day
        print(day)
        df = yearToFlowDf[year]
        dayFlows = df.iloc[day]
        
        xs = []
        ys = []
        colors = [] # FIXME: implement colors
        
        cats = np.asarray(df2["grdc_no"])
        cats = cats[areaSort]
        for i in range(len(cats)):
            cat = cats[i]
            #print(cat)
            #print(df.columns)
            #input("any key")
            if cat[1:] in df.columns:
                xs.append(i) 
                ys.append(float(dayFlows[cat[1:]]) / float(catToArea[cat]))
            else:
                xs.append(i)
                ys.append(0)
        plt.bar(xs, ys)
        plt.ylim(0,1)
        plt.ylabel("specific disharge")
        plt.xlabel("stream # (ordered by catchment area)")
        plt.title(str(year) + ", day " + str(1 + day))
        plt.savefig(outDir + str(year) + "_" + str(1 + day) + "_area")
        plt.clf()



# get all catchments for all time (get all the dfs)
# assign each one a number
# for each year
    # for each day
        # for each catchment
            # plot the flow (color according to stream order, temperature, and log precip)



