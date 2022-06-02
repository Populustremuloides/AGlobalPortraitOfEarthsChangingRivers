import pandas as pd


outDirLocal = "/home/sethbw/Documents/GlobFlow/localWaterYear/"
outDirSixMonths = "/home/sethbw/Documents/GlobFlow/localWaterYearSixMonthsApart/"


datadf = pd.read_csv("alldata.csv", encoding='latin-1')

latitudes = list(datadf[datadf.columns[2]])
catchments = list(datadf[datadf.columns[4]])
catchmentToLatitude = dict(zip(catchments, latitudes))

hCatchments = []
hemispheres = []
for catchment in catchmentToLatitude:
    try:
        if catchmentToLatitude[catchment] < 0:
            hemisphere = "southern"
        else:
            hemisphere = "northern"
        hCatchments.append(str(int(catchment)))
        hemispheres.append(hemisphere)
    except:
        pass

catchmentToHemisphere = dict(zip(hCatchments, hemispheres))

print(catchmentToHemisphere)


#print(catchments)

#maxLength = 11324
#catchmentToData = {}
#with open("concatenatedData.csv", "r+") as flowFile:
    
#    for line in flowFile:
#        line = line.replace("\n","")
#        line = line.split(",")
#        catchment = line[0]
#        data = line[1:]

#        diff = maxLength - len(data)
#        extra = [None] * diff

#        data = data + extra
        
#        catchmentToData[catchment] = data


# if the water year is the 2016-17 year, I list it as the 2017 year

def getNorthernWaterYear(date):
    year, month, date = date.split("-")  
    year = int(year)
    month = int(month)
    if month > 10:
        year = year + 1
    return str(year)

def getSouthernWaterYear(date):
    year, month, date = date.split("-") 
    year = int(year)
    month = int(month)
    if month > 6:
        year = year + 1
    return str(year)

def getSixMonthSouthernWaterYear(date): # here i opted to go for the year most covered by the water year
    year, month, date = date.split("-") 
    year = int(year)
    month = int(month)
    if month < 4:
        year = year - 1
    return str(year)

df = pd.read_csv("universallyAlignedGlobalFlow_DailyQ2_column.csv")

catchments = df.columns[4:]
dates = list(df[df.columns[2]])

# uncomment the following code the first time through to generate "universallyAlignedGlobalFlow_DailyQ2_column.csv"

#northernWaterYear = []
#southernWaterYear = []
#sixMonthSouthernWaterYear = []

#for dt in dates:
#    northernWaterYear.append(getNorthernWaterYear(dt))
#    southernWaterYear.append(getSouthernWaterYear(dt))
#    sixMonthSouthernWaterYear.append(getSixMonthSouthernWaterYear(dt))

#df["northernWaterYear"] = northernWaterYear
#df["southernWaterYear"] = southernWaterYear
#df["sixMonthSouthernWaterYear"] = sixMonthSouthernWaterYear
#df.to_csv("universallyAlignedGlobalFlow_DailyQ2_column.csv", index=False)
#dft = df.transpose()
#dft.to_csv("universallyAlignedGlobalFlow_DailyQ2_row.csv", index=False)

# add a southern hemisphere water year
# go through each catchment
# grab the water year appropriate for that catchment's lcoation
# store in a dataframe
# 

#dates = list(df[df.columns[1]])
#print(dates)
#years = []
#for dt in dates:
#    year, month, date = dt.split("-")
#    year = int(year)
#    years.append(year)

#df["year"] = years
#df[""]
#df = pd.DataFrame.from_dict(catchmentToData)
#df.to_csv("concatenatedColumnWithNones.csv")

print(df.columns)
for year in range(2015, 2017):
    catchmentToData = {}
    catchmentToDataSixMonths = {}

    for column in df:
        if column in catchmentToHemisphere.keys():
            hemisphere = catchmentToHemisphere[column]
            if hemisphere == "northern":
                dfN = df[column]
                data = list(dfN.loc[df["northernWaterYear"] == year])
                catchmentToData[column] = data[:365]

                data = list(dfN.loc[df["northernWaterYear"] == year])                
                catchmentToDataSixMonths[column] = data[:365]

            else:

                dfN = df[column]
                data = list(dfN.loc[df["southernWaterYear"] == year])                
                catchmentToData[column] = data[:365]

                data = list(dfN.loc[df["sixMonthSouthernWaterYear"] == year])                
                catchmentToDataSixMonths[column] = data[:365]
   
    print("lengths")
    for key in catchmentToData.keys():
        print(len(catchmentToData[key]))

    newDF = pd.DataFrame.from_dict(catchmentToData)
    newDFSixMonths = pd.DataFrame.from_dict(catchmentToDataSixMonths)
        
    print(newDF)
    print(newDFSixMonths)

    newDF.to_csv(outDirLocal + "globFlowData_localWaterYear_" + str(year) + ".csv", index=False)
    newDFSixMonths.to_csv(outDirSixMonths + "globFlowData_sixMonthWaterYear_" + str(year) + ".csv", index=False)
   
#year = 1987
#for i in range(maxLength // 365):
#    newDict = {}

#    start = i
#    end = i + 365
   
#    for catchment in catchmentToData.keys():
#        data = catchmentToData[catchment][start:end]
#        newDict[catchment] = data

#    df = pd.DataFrame.from_dict(newDict)

