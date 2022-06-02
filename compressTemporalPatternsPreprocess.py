import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from tqdm import tqdm
from customDataset import *
import pandas as pd
import os

assert torch.cuda.is_available() # You need to request a GPU from Runtime > Change Runtime Type

# Write the boilerplate code from the video here

# Create a dataset class that extends the torch.utils.data Dataset class here

path = "/home/sethbw/Documents/GlobFlow/localWaterYearSpectralDecomposition/"
yearToDf = {}
yearToCatchments = {}
allCatchments = []
for i in range(1988,2017):
    for ifile in os.listdir(path):
        if str(i) in ifile:
            if "FlowPeriodPowers" in ifile:
                df = pd.read_csv(path + ifile)
                yearToDf[str(i)] = df

                catchments = df.columns
                cats = []
                for cat in catchments:
                    if cat.startswith("X"):
                        cats.append(cat)
                yearToCatchments[str(i)] = cats
                allCatchments = allCatchments + cats
                #print(ifile)
                #print(df)
    #print(cats)
print(yearToCatchments)

allCatchments = set(allCatchments)

# get catchments to years where we have data, and make sure the data are continuous

catchmentToYears = {}
for cat in allCatchments:
    for year in range(1988, 2017):
        if cat in yearToCatchments[str(year)]:
            if cat not in catchmentToYears.keys():
                catchmentToYears[cat]= []
            catchmentToYears[cat].append(year)

print(catchmentToYears)

deleteCats = []
for cat in catchmentToYears.keys():
    years = catchmentToYears[cat]
    j = 0
    for i in range(1,len(years)):
        if (int(years[i]) - int(years[i - 1])) != 1:
            break
        j = j + 1
    if j >=9:
        catchmentToYears[cat] = catchmentToYears[cat][:j]
    else:
        deleteCats.append(cat)
for cat in deleteCats:
    catchmentToYears.pop(cat,None)



for cat in catchmentToYears.keys():
    years = catchmentToYears[cat]
    for i in range(1,len(years)):
        if (int(years[i]) - int(years[i - 1])) != 1:
            print("ERROR " + str(cat))
            print(int(years[i]) - int(years[i - 1]))
            print(years)

def normalizeYears(years):
    newYears = []
    for year in years:
        newYears.append((float(year) - 1988.0) / (2017.0-1988.0))
    return newYears

# concatenate all teh data
dataDict = {"catchment":[]}
j = 0
for cat in catchmentToYears.keys():
    years = catchmentToYears[cat]
    years = years[:10] #keep only the first 10 years (for simplicity)

    if len(years) == 10:
        dataDict["catchment"].append(cat)
        fullData = []
        fullData = fullData + normalizeYears(years)

        for i in range(1,len(years)):
            data = list(yearToDf[str(years[i])][cat])
            fullData = fullData + data

        for i in range(len(fullData)):
            if j == 0:
                dataDict[i] = []
            dataDict[i].append(fullData[i])

    j = j + 1

dataDf = pd.DataFrame.from_dict(dataDict)
dataDf.to_csv("ml_all_years_data.csv", index=False)

