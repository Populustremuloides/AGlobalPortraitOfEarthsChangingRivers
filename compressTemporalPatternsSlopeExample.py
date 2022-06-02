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
df2 = pd.read_csv("/home/sethbw/Documents/GlobFlow/localWaterYearSpectralDecomposition/globFlowData_localWaterYear_2013_FlowPeriodPowers.csv")
print(df2)
scale = list(df2["scale"])


print("reading in dataframe")
df = pd.read_csv("ml_all_years_data_separate.csv")
print("constructing dataset")
dataset = CustomImageDataset(df)

# Extend the torch.Module class to create your own neural network
class LinearNetwork(nn.Module):
    def __init__(self, in_dim = 752, out_dim = 752, compression_level=3):
        super(LinearNetwork, self).__init__()
        
        print("internal compression level: " + str(compression_level))

        self.l1 = nn.Linear(in_dim, 200)
        self.l2 = nn.Linear(200, 100)
        self.l3 = nn.Linear(100, compression_level)
        self.l4 = nn.Linear(compression_level, 100)
        self.l5 = nn.Linear(100, 200)
        self.l6 = nn.Linear(200, out_dim)

#        self.l1 = nn.Linear(in_dim, 500)
#        self.l2 = nn.Linear(500, 200)
#        self.l3 = nn.Linear(200, 100)
#        self.l4 = nn.Linear(100, 10)
#        self.l5 = nn.Linear(10, compression_level)
#        self.l6 = nn.Linear(compression_level, 10)
#        self.l7 = nn.Linear(10, 100)
#        self.l8 = nn.Linear(100, 500)
#        self.l9 = nn.Linear(500, out_dim)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        #n, f = x.size()
        # n is the batch size, ch and w are used to calculate the dimensions in the flattened output vector
        #flattened = x.view(n, f)
            #return self.net(flattened) 
        #print(x)
        out = self.relu(self.l1(x)) 
        out = self.relu(self.l2(out)) 

        # save the encoding
        out = self.l3(out)
        encoding = out
        out = self.relu(out)

        out = self.relu(self.l4(out)) 
        out = self.relu(self.l5(out)) 
        out = self.l6(out)

#        out = self.relu(self.l6(out)) 
#        out = self.relu(self.l7(out)) 
#        out = self.relu(self.l8(out)) 
#        out = self.l9(out)
        return out, encoding

# Instantiate the train and validation sets

loader = DataLoader(dataset, batch_size = 25, pin_memory=True, shuffle=True)
loader2 = DataLoader(dataset, batch_size = 1, pin_memory=True)

# Instantiate your model and loss and optimizer functions

for c_level in range(1,2):

    if c_level == 1:
        dataDict = {"catchment":[], "x":[]} 

    model = LinearNetwork(compression_level=c_level)
    model = model.cuda()

    objective = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run your training / validation loops
    print("compression_level: " + str(c_level))
    
    numEpochs = 10

    loss_list = []
    for epoch in range(numEpochs):
        loop = tqdm(total=len(loader), position=0)
        j = 0
        for x, catchment in loader:

            x = x.cuda()

            optimizer.zero_grad()
            y_hat, encoding = model(x)
            loss = objective(y_hat, x) # reconstruct the original input
            
            loss.backward()
            optimizer.step()
            
            #if j == 0:
            #    plt.plot(y_hat[0].detach().cpu().numpy(), label="yhat")
            #    plt.plot(x[0].detach().cpu().numpy(), label="x")
            #    plt.show()

            loop.set_description('loss:{:.4f}'.format(loss.item()))
            loop.update(1)
            loss_list.append(loss.item())
            j = j + 1
        loop.close()
     # save the losses if you would like to plot them
   
    final_losses = []
    j = 0
    for x, catchment in loader2:
        x = x.cuda()
        reconstruction, encoding = model(x)
        loss = objective(reconstruction, x) # reconstruct the original input
        loss = loss.detach().cpu().numpy()
        final_losses.append(loss) 
        encoding = encoding.detach().cpu().numpy()[0]
        if j < 10 or (j > 500 and j < 510) or (j > 1500 and j < 1510) or (j > 3000 and j < 3010):
            #print(loss)
            x = x.detach().cpu().numpy()[0]
            reconstruction = reconstruction.detach().cpu().numpy()[0]
            plt.plot(scale, x, label="original")
            plt.plot(scale, reconstruction, label="reconstruction")
            plt.legend()
            plt.title("Example Frequency Decomposition Reconstruction")
            plt.xlabel("period length (days)")
            plt.ylabel("spectral power")
            outRoot = "/home/sethbw/Documents/paper2figures/reconstructions/"
            plt.savefig(outRoot + str(j))
            plt.clf()
            #plt.show()

        #print(encoding)
        j = j + 1

    print("mean final loss: ") 
    print(np.mean(final_losses))

