import os
import numpy as np
import matplotlib.pyplot as plt
import imageio


root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/gif_figures/"

with imageio.get_writer('whole_dataset.gif', mode='I') as writer:
    for year in range(1988, 2017):
        for day in range(1,366):
            filename = root + str(year) + "_" + str(day) + "_area.png"
            image = imageio.imread(filename)
            writer.append_data(image)

