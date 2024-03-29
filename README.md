# A Global Portrait Of Earth's Changing Rivers
Analysis Code for the manuscript, Global river flow shows dramatic shifts since 1990’s, especially in dry, warm catchments.


## Data Availability

The raw data files used in this analysis are currently published on Research Gate [here](https://doi.org/10.13140/RG.2.2.31696.84487) and [here](https://doi.org/10.13140/RG.2.2.24985.95842). The methods used for generating these data can be found in [this preprint](https://doi.org/10.1002/essoar.10507854.1).

## Dependencies

The Python code was executed using version 3.7.1. To run the code, you will need the following packages installed in your environment:

- pandas 1.2.3
- scipy 1.6.2
- numpy 1.19.2
- matplotlib 3.3.4
- seaborn 0.11.1
- pytorch 1.3.1
- tqdm 4.5.0
- sklearn 0.24.1
- cartopy 0.20.0

In addition, the code uses the following native libraries:
- math
- os

R code was executing using version 3.6.1. R libraries used include the following:
- WaveletComp 1.1


## Code Organization

The following table represents the dependency graph structure between each figure in the manuscript and the code and data files used to generate it, as well as the parent code and data files all the way back to the original data files. Be sure to scroll to the right as your monitor may not be large enough to view all the columns at once.

code file | input file | output file | figure produced (if any) | Notes
-------------- | ---- | -------- | ------ | -----
getExampleStream.py | FullDatabase.csv |  | 1 | all panels
getExampleStream.py | localWaterYear |  |  |
getExampleStream.py | localWaterYearSpectralDecomposition |  |  |
abstractClusters.py | mergedData.csv |  | 2 | all panels
plotChanges.py | mergedData.csv | | 3 | all panels
mapAll.py | mergedData.csv | | 4 | all panels
lassoRegressionHeatMaps.py | mergedData.csv | | 5 | all panels
plotColdLikeHot.py | mergedData.csv | | 6 | all panels
colorCatchments.py | | | | 
buildGif.py | gif_figures |  | movie1 |
makeFinalFlowGif.py | localWaterYear | gif_figures |  |
compileLoss.py | ml_exampleXslope_encodings_loss_.csv |  | S1 |
compressTemporalPatternsSlopeExample.py | ml_all_years_data_separate.csv | ml_exampleXslope_encodings_loss_.csv | S2 |
example_spectral_properties_hydrographs.py | alldata.csv |  | S3 | All frames
mapAll | mergedData.csv | | S4 | frames A and B
plotRelationships.py | mergedData.csv | | S4 | frame C
autocorrelation.py | mergedData.csv | | S5 | all frames
plotChanges.py | mergedData.csv | | S6 | all frames
mapAll | mergedData.csv | | S7 | all frames
spatialSimilarityFigure.py | slopes_empiricalDistributions.csv | | S8 | all frames
spatialSimilarity.py | distancces.csv | slopes_empiricalDistributions.csv | |
spatialSimilarity.py | mergedData.csv | | |
plotRelationships.py | mergedData.csv | | S9 |
lassoRegressionHeatmaps.py | mergedData.csv | | S10 |
plotRelationships.py | mergedData.csv | | S11-S16 | all figures/panels
combineAllData.py | allData.csv | mergedData.csv | |
combineAllData.py | ml_slope_encodings_1.csv | mergedData.csv | |
combineAllData.py | ml_encodings_1.csv | mergedData.csv | |
combineAllData.py | specific_discharge_vs_size.csv | mergedData.csv | |
combineAllData.py | dayOfMeanFlowThroughTime.csv | mergedData.csv | |
getDistanceDf.csv | alldata.csv | distance_df.csv | | 
empiricalDistributionTiming.py | alldata.csv | dayOfMeanFlow_slopes_empiricalDistribution.csv | | 
empiricalDistributionTiming.py | dayOfMeanFlowThroughTime.csv | | | 
empiricalDistributionVolume.py | alldata.csv | dayOfMeanFlow_slopes_empiricalDistribution.csv | | 
empiricalDistributionVolume.py | specific_discharge_vs_size.csv |  | | 
empiricalDistributionVariability.py | alldata.csv | dayOfMeanFlow_slopes_empiricalDistribution.csv | | 
empiricalDistributionVariability.py | ml_slope_encodings1.csv |  | | 
dayOfMeanFlowThroughTime.py | alldata.csv | dayOfMeanFlowThroughTime.csv |  |
spectralThroughTime.py | alldata.csv | spectralPowersThroughTime.csv |  |
splitIntoNonGlobalWaterYears.py | alldata.csv | localWaterYear |  |
plotSpectralNumberThroughTime.py | day_of_mean_flow_vs_size.csv | spectralNumber_acrossTime.csv |  |
combineThroughTime.py | dayOfMeanFlowThroughTime.csv | throughTimeCombined.csv |  |
testDifferenceInDayOfMeanFlow.py | FullDatabase.csv | day_of_mean_flow_vs_size.csv |  |
testDifferencesInMean.py | FullDatabase.csv | specific_discharge_vs_size.csv |  | Convert discharge to specific disharge data
convertToSpecificDischarge.py | localWaterYear | localWaterYear |  | converts to specific discharge
dayOfMeanFlowThroughTime.py | localWaterYear | dayOfMeanFlowThroughTime.csv |  |
example_spectral_properties_hydrographs.py | localWaterYear |  |  |
specificDischargeThroughTime.py | localWaterYear | specificDischargeThroughTime.csv |  |
splitIntoNonGlobalWaterYears.py | localWaterYear | universallyAlignedGlobalFlow_DailyQ2_column.csv |  |
WaveletAnalysisLocalWaterYear.R | localWaterYear | localWaterYearSpectralDecomposition |  |
compressTemporalPatternsPreprocess.py | localWaterYearSpectralDecomposition | ml_all_years_data_separate.csv |  | create dataset for dimensionality compression
example_spectral_properties_hydrographs.py | localWaterYearSpectralDecomposition |  |  |
spectralThroughTime.py | localWaterYearSpectralDecomposition | spectralPowersThroughTime.csv |  |
testDifferenceInDayOfMeanFlow.py | localWaterYearSpectralDecomposition | day_of_mean_flow_vs_size.csv |  |
testDifferencesInMean.py | localWaterYearSpectralDecomposition | specific_discharge_vs_size.csv |  |
compressTemporalPaternsSlope.py | ml_all_years_data_separate.csv | ml_slope_encodings1.csv |  |
combineThroughTime.py | ml_slope_encodings1.csv | throughTimeCombined.csv |  |
example_spectral_properties_hydrographs.py | ml_slope_encodings1.csv |  |  |
plotSpectralNumberThroughTime.py | ml_slope_encodings1.csv | spectralNumber_acrossTime.csv |  |
plotSpectralNumberThroughTime.py | specific_discharge_vs_size.csv | spectralNumber_acrossTime.csv |  |
combineThroughTime.py | specificDischargeThroughTime.csv | throughTimeCombined.csv |  |
dimensionalityCompression.py | universallyAligned_powers.csv | ml_encodings1.csv |  |
WaveletAnalysisGlobal.R | universallyAlignedGlobalFlow_DailyQ2_column.csv | universallyAligned_powers.csv |  |
WaveletAnalysisGlobal.R | universallyAlignedGlobalFlow_DailyQ2_column.csv | universallyAligned_powersTranspose.csv |  |
