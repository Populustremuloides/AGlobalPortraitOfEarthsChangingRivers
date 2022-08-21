import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from ColorCatchments import *

df = pd.read_csv("mergedData.csv")

root = "/home/sethbw/Documents/GlobFlow/spectralAnalysis/maps_try_2"

if not os.path.exists(root):
    os.mkdir(root)

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

varToTitle = {
        "masd_mean":"Mean Annual Specific Dicharge",
        "masd_mean_transformed":"Mean Annual Specific Dicharge (4" + "th".translate(trans) + "root transformed)",
        "masd_slope":"Change in Mean Annual Specific Discharge",
        "masd_slope_normalized":"Percent Change in Mean Annual Specific Discharge",
        "domf_mean":"Day of Mean Flow",
        "domf_slope":"Change in Day of Mean Flow",
        "domf_slope_normalized":"Percent Change in Day of Mean Flow",
        "spectral_mean":"Mean Annual Spectral Number",
        "spectral_slope":"Change in Mean Annual Spectral Number",
        "spectral_slope_normalized":"Percent Change in Mean Annual Spectral Number",
        "spectral_full":"Full Spectral Number"
        }

varToTitleS = {
        "masd_mean":"MAP_MeanAnnualSpecificDicharge.png",
        "masd_mean_transformed":"MAP_MeanAnnualSpecificDichargeTransformed.png",
        "masd_slope":"MAP_ChangeinMeanAnnualSpecificDischarge.png",
        "masd_slope_normalized":"MAP_PercentChangeinMeanAnnualSpecificDischarge.png",
        "domf_mean":"MAP_DayofMeanFlow.png",
        "domf_slope":"MAP_ChangeinDayofMeanFlow.png",
        "domf_slope_normalized":"MAP_PercentChangeinDayofMeanFlow.png",
        "spectral_mean":"MAP_MeanAnnualSpectralNumber.png",
        "spectral_slope":"MAP_ChangeinMeanAnnualSpectralNumber.png",
        "spectral_slope_normalized":"MAP_PercentChangeinMeanAnnualSpectralNumber.png",
        "spectral_full":"MAP_FullSpectralNumber.png"
        }


def plotVar(var, m, transform=None, save=False, saveDir=None):
    # width, height
    fig = plt.figure(figsize=(11 * 2,6 * 2))

    ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    ax.set_extent([-180,180,-90,90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    
    colors = getColors(var, m, df, transform)
    plt.scatter(x=df["new_lon.x"], y=df["new_lat.x"], c=colors, s=5, alpha=0.9)
    
    if transform == None:
        plt.title(varToTitle[var], fontsize=40)
    else:
        plt.title(varToTitle[var + "_transformed"], fontsize=40)
    
    if save:
        if transform == None:
            plt.savefig(os.path.join(saveDir, varToTitleS[var]), dpi=300)
        else:
            plt.savefig(os.path.join(saveDir, varToTitleS[var + "_transformed"]), dpi=300)
    plt.show()

cmap = PuOr_r

m = getM_spectralMean(cmap)
plotVar("spectral_mean", m, save=True, saveDir=root)
colorbar_spectralMean(cmap, save=True, saveDir=root)
colorbar_spectralMean(cmap, save=True, saveDir=root, pLeft=True)

m = getM_spectralSlope(cmap)
plotVar("spectral_slope", m, save=True, saveDir=root)
colorbar_spectralSlope(cmap, save=True, saveDir=root)
colorbar_spectralSlope(cmap, pLeft=True, save=True, saveDir=root)

# % change in spectral number doesn't make any sense
#m = getM_spectralSlopeNormalized(cmap)
#plotVar("spectral_slope_normalized", m)
#colorbar_spectralSlopeNormalized(cmap)

#cmap = plasma_r
cmap = PuOr_r
m = getM_spectralFull(cmap)
plotVar("spectral_full", m, save=True, saveDir=root)
colorbar_spectralFull(cmap, save=True, saveDir=root)
colorbar_spectralFull(cmap, pLeft=True, save=True, saveDir=root)

cmap = seismic_r

# keep as supplementary material (not very informative)
m = getM_masdMean(cmap)
plotVar("masd_mean", m, save=True, saveDir=root)
colorbar_masdMean(cmap, save=True, saveDir=root)
colorbar_masdMean(cmap, pLeft=True, save=True, saveDir=root)

m = getM_masdMean4thRoot(cmap)
plotVar("masd_mean", m, transform_masdMean4thRoot, save=True, saveDir=root)
colorbar_masdMean4thRoot(cmap, save=True, saveDir=root)
colorbar_masdMean4thRoot(cmap, pLeft=True, save=True, saveDir=root)

# keep as supplemental material
m = getM_masdSlope(cmap)
plotVar("masd_slope", m, save=True, saveDir=root)
colorbar_masdSlope(cmap, save=True, saveDir=root)
colorbar_masdSlope(cmap, pLeft=True, save=True, saveDir=root)

m = getM_masdSlopeNormalized(cmap)
plotVar("masd_slope_normalized", m, save=True, saveDir=root)
colorbar_masdSlopeNormalized(cmap, save=True, saveDir=root)
colorbar_masdSlopeNormalized(cmap, pLeft=True, save=True, saveDir=root)

cmap = PiYG

m = getM_domfMean(cmap)
plotVar("domf_mean", m, save=True, saveDir=root)
colorbar_domfMean(cmap, save=True, saveDir=root)
colorbar_domfMean(cmap, pLeft=True, save=True, saveDir=root)

m = getM_domfSlope(cmap)
plotVar("domf_slope", m, save=True, saveDir=root)
colorbar_domfSlope(cmap, save=True, saveDir=root)
colorbar_domfSlope(cmap, pLeft=True, save=True, saveDir=root)

# % change in domf doesn't make sense either
#m = getM_domfSlopeNormalized(cmap)
#plotVar("domf_slope_normalized", m)
#colorbar_domfSlopeNormalized(cmap)
