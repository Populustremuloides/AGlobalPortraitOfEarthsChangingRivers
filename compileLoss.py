import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataDict = {"loss":[],"example":[],"batch":[]}

for i in range(1,10):
    print(i)
    fileName = "ml_example" + str(i) + "slope_encodings_loss_1.csv"
    df = pd.read_csv(fileName)    
    losses = list(df["loss"])
    for j in range(len(losses)):

        dataDict["loss"].append(losses[j])
        dataDict["example"].append(j)
        dataDict["batch"].append(j)

df = pd.DataFrame.from_dict(dataDict)
df.to_csv("compiledLosses", index=False)

sns.lineplot(data=df, x="batch",y="loss")
plt.title("Autoencoder Losses over Batches")
plt.ylabel("loss")
plt.xlabel("batch")
plt.show()


