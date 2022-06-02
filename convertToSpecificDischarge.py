import pandas as pd
import numpy as np

root = "/home/sethbw/Documents/GlobFlow/localWaterYear"

df = pd.read_csv(root + "/meanFlow.csv")
#df = pd.read_csv("universallyAlignedGlobFlow_DailyQ2_column.csv")

cdf = pd.read_csv("FullDatabase.csv")
print(cdf["grdc_no"])
newNo = []
for item in cdf["grdc_no"]:
    try:
        newNo.append("X" + str(int(float(item))))
    except:
        newNo.append(None)
cdf["grdc_no"] = newNo

#print(cdf["area_sqkm"])
#print(list(cdf.columns))


dataDict = {}
for col in df.columns:
    print(col)
    if col.isnumeric() or "(" in col:
        colx = "X" + str(col)
        row = cdf[cdf["grdc_no"] == colx]
        try:
            area = float(row["area_sqkm"])
            newCol = []
            for item in df[col]:
                try:
                    flow = float(item)
                    sd = flow / area
                    newCol.append(sd)
                except:
                    newCol.append(None)
            dataDict[col] = newCol
        except:
            print(row)
    else:
        dataDict[col] = df[col]
sdDf = pd.DataFrame.from_dict(dataDict)
sdDf.to_csv(root + "/specific_discharge_meanFlow.csv")



