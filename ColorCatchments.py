import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
import string
import os

''' 
The purpose of this script is to make a set of functions that 
return colors used for figures in a way that is consistent
across figures.
'''

df = pd.read_csv("mergedData.csv")
#i = 0
#for col in df.columns:
#    print(col)
#    if i > 20:
#        quit()
#    i = i + 1

degreesC = "$^{\circ}$C"
delta = r"$\Delta$"

seismic = mpl.cm.get_cmap('seismic')
seismic_r = mpl.cm.get_cmap('seismic_r')
PiYG = mpl.cm.get_cmap('PiYG')
PiYG_r = mpl.cm.get_cmap('PiYG_r')
cool = mpl.cm.get_cmap('cool')
cool_r = mpl.cm.get_cmap('cool_r')
plasma = mpl.cm.get_cmap('plasma')
plasma_r = mpl.cm.get_cmap('plasma_r')
viridis = mpl.cm.get_cmap('viridis')
viridis_r = mpl.cm.get_cmap('viridis_r')
PuOr = mpl.cm.get_cmap("PuOr")
PuOr_r = mpl.cm.get_cmap("PuOr_r")

def getCmapFromString(cmapString):
    if cmapString == "seismic":
        cmap = mpl.cm.get_cmap('seismic')
    elif cmapString == "seismic_r":
        cmap = mpl.cm.get_cmap('seismic_r')
    elif cmapString == 'PiYG':
        cmap = mpl.cm.get_cmap('PiYG')
    elif cmapString == 'PiYG_r':
        cmap = mpl.cm.get_cmap('PiYG_r')
    elif cmapString == 'cool':
        cmap = mpl.cm.get_cmap('cool')
    elif cmapString == 'cool_r':
        cmap = mpl.cm.get_cmap('cool_r')
    elif cmapString == 'plasma':
        cmap = mpl.cm.get_cmap('plasma')
    elif cmapString == 'plasma_r':
        cmap = mpl.cm.get_cmap('plasma_r')
    elif cmapString == 'viridis':
        cmap = mpl.cm.get_cmap('viridis')
    elif cmapString == 'viridis_r':
        cmap = mpl.cm.get_cmap('viridis_r')
    elif cmapString == "PuOr":
        cmap = mpl.cm.get_cmap("PuOr")
    elif cmapString == "PuOr_r":
        cmap = mpl.cm.get_cmap("PuOr_r")
    elif cmapString == "temp":
        cmap = mpl.cm.get_cmap('seismic')
    elif cmapString == "gord":
        cmap = mpl.cm.get_cmap('PiYG')
    elif cmapString == "precip":
        cmap = sns.diverging_palette(330, 250, s=100, as_cmap=True)
    else:
        print(cmapString + " not a recognized cmap")
    return cmap


superscriptDict = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ",
    "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "i": "ᶦ", "j": "ʲ", "k": "ᵏ",
    "l": "ˡ", "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "q": "۹", "r": "ʳ",
    "s": "ˢ", "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ",
    "z": "ᶻ", "A": "ᴬ", "B": "ᴮ", "C": "ᶜ", "D": "ᴰ", "E": "ᴱ", "F": "ᶠ",
    "G": "ᴳ", "H": "ᴴ", "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ",
    "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "Q": "Q", "R": "ᴿ", "S": "ˢ", "T": "ᵀ",
    "U": "ᵁ", "V": "ⱽ", "W": "ᵂ", "X": "ˣ", "Y": "ʸ", "Z": "ᶻ", "+": "⁺",
    "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾"}
trans = str.maketrans(
    ''.join(superscriptDict.keys()),
    ''.join(superscriptDict.values()))

seismic_r = mpl.cm.get_cmap('seismic_r')

def _printTruncation(var, lowerBound, upperBound, transform=None):
    if transform != None:
        numTruncated = np.sum(transform(df[var]) < lowerBound) + np.sum(transform(df[var]) > upperBound)
    else:
        numTruncated = np.sum(df[var] < lowerBound) + np.sum(df[var] > upperBound)

    print("*******************") 
    print("number truncated for " + var + " " + str(numTruncated))
    print(" = " + str(100 * (numTruncated / np.sum(~df[var].isna()))) + " % of the data")
    print("*******************")

# *********************************************************************************
# log mean annual precipitation detrended
# *********************************************************************************
#minPrecAnnDetrendedVal = np.min(df["MeanPrecAnnDetrended"])

#def transform_MeanPrecAnnDetrendedLog(array):
#    array = np.asarray(array)
#    posArray = array + 1.1 - minPrecAnnDetrendedVal
#    result = np.log(posArray)
#    return result

#transformedVals = transform_MeanPrecAnnDetrendedLog(df["MeanPrecAnnDetrended"]) # calculate once for speed later on
#plt.hist(transformedVals)
#plt.show()

def getNorm_MeanPrecAnnDetrendedLog(printTruncation=False):
    var = "MeanPrecAnnDetrended"
    lowerBound = 2.5 #np.min(transformedVals)
    upperBound = np.max(transformedVals)
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound, transform=transform_MeanPrecAnnDetrendedLog)

    return norm

def getM_MeanPrecAnnDetrendedLog(cmap):
    norm = getNorm_MeanPrecAnnDetrendedLog()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m
'''
def colorbar_MeanPrecAnnDetrendedLog(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_MeanPrecAnnDetrendedLog(printTruncation=True)
    m = getM_MeanPrecAnnDetrendedLog(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("log detrended mean annual precipitation (mm)", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_MeanPrecAnnDetrendedLog.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_MeanPrecAnnDetrendedLog.png"))
    plt.show()
'''
# *********************************************************************************
# log mean annual precipitation
# *********************************************************************************

def transform_MeanPrecAnnLog(array):
    array = np.asarray(array)
    return np.log(array + 1)

def getNorm_MeanPrecAnnLog(printTruncation=False):
    norm = mpl.colors.Normalize(vmin=2, vmax=np.max(np.log(df["MeanPrecAnn"] + 1)))
    var = "MeanPrecAnn"
    lowerBound = 2
    upperBound = np.inf
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
        
    return norm

def getM_MeanPrecAnnLog(cmap):
    norm = getNorm_MeanPrecAnnLog()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_MeanPrecAnnLog(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_MeanPrecAnnLog(printTruncation=True)
    m = getM_MeanPrecAnnLog(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("log mean annual precipitation (mm)", size=15, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_MeanPrecAnnLog.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_MeanPrecAnnLog.png"))
    plt.show()

# *********************************************************************************
# mean annual temperature
# *********************************************************************************

def getNorm_MeanTempAnn(printTruncation=False):
    norm = mpl.colors.Normalize(vmin=np.min(df["MeanTempAnn"]), vmax=np.max(df["MeanTempAnn"]))
    return norm

def getM_MeanTempAnn(cmap):
    norm = getNorm_MeanTempAnn()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_MeanTempAnn(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_MeanTempAnn()
    m = getM_MeanTempAnn(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean annual temperature (" + degreesC + ")", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_MeanTempAnn.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_MeanTempAnn.png"))
    plt.show()

# *********************************************************************************
# stream order (gord)
# ********************************************************************************* 
def getNorm_gord(printTruncation=False):
    norm = mpl.colors.Normalize(vmin=np.min(df["gord"]), vmax=np.max(df["gord"]))
    return norm

def getM_gord(cmap):
    norm = getNorm_gord()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_gord(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_gord()
    m = getM_gord(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("stream order", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_gord.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_gord.png"))
    plt.show()


# *********************************************************************************
# mean annual specific discahrge - fourth rooted (makes the data approximately normal)
# *********************************************************************************

# show how this makes the data distributed
#plt.hist(np.sqrt(np.sqrt(df["masd_mean"])))
#plt.show()

def transform_masdMean4thRoot(array):
    return np.sqrt(np.sqrt(array))

def getNorm_masdMean4thRoot(printTruncation=False):
    masds = np.sqrt(np.sqrt(df["masd_mean"]))
    norm = mpl.colors.Normalize(vmin=np.min(masds), vmax=np.max(masds))
    return norm

def getM_masdMean4thRoot(cmap):
    norm = getNorm_masdMean4thRoot()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdMean4thRoot(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdMean4thRoot()
    m = getM_masdMean4thRoot(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("transformed mean annual specific discharge" + "(L/s/km" + "2".translate(trans) + ")" + "1/4".translate(trans), size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_masdMean4thRoot.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_masdMean4thRoot.png"))
    plt.show()

# *********************************************************************************
# mean annual specific discharge - mean
# *********************************************************************************
#plt.hist(df["masd_mean"])
#plt.show()

def getNorm_masdMean(printTruncation=False): 
    var = "masd_mean"
    lowerBound = np.min(df["masd_mean"])
    upperBound = 100
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
    return norm

def getM_masdMean(cmap):
    norm = getNorm_masdMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdMean(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdMean(printTruncation=True)
    m = getM_masdMean(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean annual specific discharge (L/s/km" + "2".translate(trans) + ")", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_masdMean.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_masdMean.png"))
    plt.show()


# *********************************************************************************
# mean annual specific discharge - slope
# *********************************************************************************
#plt.hist(df["masd_slope"])
#plt.show()

def getNorm_masdSlope(printTruncation=False): 
    var = "masd_slope"
    lowerBound = -0.0011
    upperBound = 0.0011
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
        
    return norm

def getM_masdSlope(cmap):
    norm = getNorm_masdSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdSlope(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdSlope(printTruncation=True)
    m = getM_masdSlope(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(delta + " in mean annual specific discharge (L/s/km" + "2".translate(trans) + " / year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_masdSlope.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_masdSlope.png"))
    plt.show()


# *********************************************************************************
# mean annual specific discharge - slope normalized
# *********************************************************************************
#plt.hist(df["masd_slope_normalized"])
#plt.show()

def getNorm_masdSlopeNormalized(printTruncation=False): 
    var = "masd_slope_normalized"
    lowerBound = -0.011
    upperBound = 0.011
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
        
    return norm

def getM_masdSlopeNormalized(cmap):
    norm = getNorm_masdSlopeNormalized()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_masdSlopeNormalized(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_masdSlopeNormalized(printTruncation=True)
    m = getM_masdSlopeNormalized(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("% change in mean annual specific discharge / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_masdSlopeNormalized.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_masdSlopeNormalized.png"))
    plt.show()


# *********************************************************************************
# day of mean flow - mean
# *********************************************************************************
#plt.hist(df["domf_mean"])
#plt.show()

def getNorm_domfMean(): 
    domf_means = df["domf_mean"] 
    norm = mpl.colors.Normalize(vmin=np.min(domf_means), vmax=np.max(domf_means))
    return norm

def getM_domfMean(cmap):
    norm = getNorm_domfMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfMean(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfMean()
    m = getM_domfMean(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("day of mean flow (day in water year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_domfMean.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_domfMean.png"))
    plt.show()


# *********************************************************************************
# day of mean flow - slope
# *********************************************************************************
#plt.hist(df["domf_slope"])
#plt.show()

def getNorm_domfSlope(printTruncation=False): 
    var = "domf_slope"
    lowerBound = -4
    upperBound = 4
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound) 
        
    return norm

def getM_domfSlope(cmap):
    norm = getNorm_domfSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfSlope(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfSlope(printTruncation=True)
    m = getM_domfSlope(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(delta + " in day of mean flow (days / year)", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_domfSlope.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_domfSlope.png"))
    plt.show()

# *********************************************************************************
# day of mean flow - slope normalized
# *********************************************************************************
#plt.hist(df["domf_slope_normalized"])
#plt.show()

def getNorm_domfSlopeNormalized(printTruncation=False): 
    var = "domf_slope_normalized"
    lowerBound = -3
    upperBound = 3
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
       
    return norm

def getM_domfSlopeNormalized(cmap):
    norm = getNorm_domfSlopeNormalized()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_domfSlopeNormalized(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_domfSlopeNormalized(printTruncation=True)
    m = getM_domfSlopeNormalized(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("% change in day of mean flow / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_domfSlopeNormalized.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_domfSlopeNormalized.png"))
    plt.show()


# *********************************************************************************
# spectral number (annual) - mean
# *********************************************************************************
#plt.hist(df["spectral_mean"])
#plt.show()

def getNorm_spectralMean(printTruncation=False): 
    var = "spectral_mean"
    lowerBound = -2.5
    upperBound = 4
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
    
    return norm

def getM_spectralMean(cmap):
    norm = getNorm_spectralMean()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_spectralMean(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_spectralMean(printTruncation=True)
    m = getM_spectralMean(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("mean annual spectral number", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)
    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_spectralMean.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_spectralMean.png"))
    plt.show()


# *********************************************************************************
# spectral number (annual) - slope
# *********************************************************************************
#plt.hist(df["spectral_slope"])
#plt.show()

def getNorm_spectralSlope(printTruncation=False): 
    var = "spectral_slope"
    lowerBound = -0.11
    upperBound = 0.11
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
       _printTruncation(var, lowerBound, upperBound) 
        
    return norm

def getM_spectralSlope(cmap):
    norm = getNorm_spectralSlope()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_spectralSlope(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_spectralSlope(printTruncation=True)
    m = getM_spectralSlope(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(delta + " in spectral number / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20) 
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_spectralSlope.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_spectralSlope.png"))
    plt.show() 

# *********************************************************************************
# spectral number (annual) - slope normalized
# *********************************************************************************
#vals = df["spectral_slope_normalized"]
#mask = np.asarray(vals < 20)
#vals = np.asarray(vals)
#vals = vals[mask]
#plt.hist(vals)
#plt.show()

def getNorm_spectralSlopeNormalized(printTruncation=False): 
    var = "spectral_slope_normalized"
    lowerBound = -12.5
    upperBound = 12.5
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    if printTruncation:
        _printTruncation(var, lowerBound, upperBound)
       
    return norm

def getM_spectralSlopeNormalized(cmap):
    norm = getNorm_spectralSlopeNormalized()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_spectralSlopeNormalized(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_spectralSlopeNormalized(printTruncation=True)
    m = getM_spectralSlopeNormalized(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("% change spectral number / year", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_spectralSlopeNormalized.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_spectralSlopeNormalized.png"))
    plt.show()


# *********************************************************************************
# spectral number (full)
# *********************************************************************************
#plt.hist(df["spectral_full"])
#plt.show()

def getNorm_spectralFull(): 
    var = "spectral_full"
    lowerBound = np.min(df[var])
    upperBound = np.max(df[var])
    norm = mpl.colors.Normalize(vmin=lowerBound, vmax=upperBound)
    
    return norm

def getM_spectralFull(cmap):
    norm = getNorm_spectralFull()
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    return m

def colorbar_spectralFull(cmap, save=False, saveDir=None, pLeft=False):
    fig, ax = plt.subplots(figsize=(3, 10))

    norm = getNorm_spectralFull()
    m = getM_spectralFull(cmap)    
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label("full spectral number", size=20, weight='bold')
    if pLeft:
        fig.subplots_adjust(left=0.775)
        ax.yaxis.set_label_position('left')
        ax.yaxis.set_ticks_position('left')
    else:
        fig.subplots_adjust(right=0.25)

    cb1.ax.tick_params(labelsize=20)
    if save:
        if pLeft:
            plt.savefig(os.path.join(saveDir,"colorbarL_spectralFull.png"))
        else:
            plt.savefig(os.path.join(saveDir,"colorbar_spectralFull.png"))

    plt.show()

#cmap = cm.seismic
#colorbar_spectralFull(cmap)

#def getColors(var, m, transform=None):
#    vals = df[var]
    
#    if transform != None:
#        vals = transform(vals)
#    plt.hist(vals)
#    plt.title("after")
#    plt.show()

#    colors = []
#    for val in vals:
#        c = m.to_rgba(val)
#        colors.append(c)
#    return colors


def getColors(var, m, df, transform=None):
    vals = df[var]
    lower, upper = m.get_clim()
    if transform != None:
        vals = transform(vals)

    colors = []
    for val in vals:
        c = m.to_rgba(val)
        colors.append(c)
    return colors

def getM(variable, cmap):
    if variable == "MeanPrecAnnDetrendedLog":
        function = getM_MeanPrecAnnDetrendedLog
    elif variable == "MeanPrecAnnLog":
        function = getM_MeanPrecAnnLog
    elif variable == "MeanTempAnn":
        function = getM_MeanTempAnn
    elif variable == "gord":
        function = getM_gord
    elif variable == "masd_mean4throot":
        function = getM_masdMean4thRoot
    elif variable == "masd_mean":
        function = getM_masdMean
    elif variable == "masd_slope":
        function = getM_masdSlope
    elif variable == "masd_slope_normalized":
        function = getM_masdSlopeNormalized
    elif variable == "domf_mean":
        function = getM_domfMean
    elif variable == "domf_slope":
        function = getM_domfSlope
    elif variable == "domf_slope_normalized":
        function = getM_domfSlopeNormalized
    elif variable == "spectral_mean":
        function = getM_spectralMean
    elif variable == "spectral_slope":
        function = getM_spectralSlope
    elif variable == "spectral_slope_normalized":
        function = getM_spectralSlopeNormalized
    elif variable == "spectral_full":
        function = getM_spectralFull
    else:
        print(variable, " not recognized as a variable that can be used to color catchments")
    
    if type(cmap) == type("string"):
        cmap = getCmapFromString(cmap)

    m = function(cmap)
    return m

def plotColorbar(variable, cmap, save=False, saveDir=None, pLeft=False):
    if variable == "MeanPrecAnnDetrendedLog":
        colorbar_MeanPrecAnnDetrendedLog(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "MeanPrecAnnLog":
        colorbar_MeanPrecAnnLog(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "MeanTempAnn":
        colorbar_MeanTempAnn(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "gord":
        colorbar_gord(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "masd_mean4throot":
        colorbar_masdMean4thRoot(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "masd_mean":
        colorbar_masdMean(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "masd_slope":
        colorbar_masdSlope(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "masd_slope_normalized":
        colorbar_masdSlopeNormalized(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "domf_mean":
        colorbar_domfMean(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "domf_slope":
        colorbar_domfSlope(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "domf_slope_normalized":
        colorbar_domfSlopeNormalized(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "spectral_mean":
        colorbar_spectralMean(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "spectral_slope":
        colorbar_spectralSlope(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "spectral_slope_normalized":
        colorbar_spectralSlopeNormalized(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    elif variable == "spectral_full":
        colorbar_spectralFull(cmap, save=save, saveDir=saveDir, pLeft=pLeft)
    else:
        print(variable, " not recognized as a variable that can be used to color catchments")

def getTransform(variable):
    if variable == "MeanPrecAnnLog":
        transform = transform_MeanPrecAnnLog
    elif variable == "MeanPrecAnnDetrendedLog":
        transform = transform_MeanPrecAnnDetrendedLog
    elif variable == "masd_mean4throot":
        transform = transform_masdMean4thRoot
    else:
        transform = None 
    return transform

