# A Global Portrait Of Earths Changing Rivers
Analysis Code for the manuscript, A Global Portrait of Earth's Changing Rivers: Smaller, Warmer Catchments are Most at Risk.

## Data Availability

The intermediary data files generated by this repository will be made available following publication. During the review process, these intermediary data files will be made available to reviewers. In the meantime, the raw data files used in this analysis are currently published on Research Gate [here](https://doi.org/10.13140/RG.2.2.31696.84487) and [here](https://doi.org/10.13140/RG.2.2.24985.95842). The methods used for generating these data can be found in [this preprint](https://doi.org/10.1002/essoar.10507854.1). The code in this repository should be sufficient to recreate any of the figures in the manuscript starting from the raw data files just mentioned. But as a convenience, as mentioned before, the intermediary data files will be made available following publication of the manuscript. 

## Dependencies

The Python code was executed using version 3.7.1. To run the code, you will need the following packages installed in your environment:

- pandas 1.2.3
- scipy 1.6.2
- numpy 1.19.2
- matplotlib 3.3.4
- seaborn 0.11.1
- pytorch 1.3.1
- tqdm 4.5.0

In addition, the code uses the following native libraries:
- math
- os


R code was executing using version 3.6.1. R libraries used include the following:
- ggplot2 3.1.1
- tidyverse 1.2.1
- scales 1.0.0
- WaveletComp 1.1


## Code Organization

The following table represents the dependency graph structure between each figure in the manuscript and the code and data files used to generate it, as well as the parent code and data files all the way back to the original data files. Be sure to scroll to the right as your monitor may not be large enough to view all the columns at once.

code file | input file | output file | figure produced (if any) | Notes
-------------- | ---- | -------- | ------ | -----
getExampleStream.py | FullDatabase.csv |  | 1 | all panels
getExampleStream.py | localWaterYear |  |  |
getExampleStream.py | localWaterYearSpectralDecomposition |  |  |
threeDPlot.py | day_of_mean_flow_vs_size.csv |  | 2 | All frames
threeDPlot.py | ml_slope_encodings1.csv |  | 2 |
threeDPlot.py | specific_discharge_vs_size.csv |  | 2 |
structureBetweenVariables.py | throughTimeCombined.csv |  | 3 | All frames
ml_encodings_means.py | alldata.csv |  | 4 |
ml_encodings_slopes.py  | alldata.csv |  | 4 |
ml_visualize_dayOfMeanFlow_mean_map.py | alldata.csv |  | 4 |
ml_visualize_dayOfMeanFlow_throgh_time_map.py | alldata.csv |  | 4 |
ml_visualizele_masd_through_time2.py | alldata.csv |  | 4 |
ml_visualize_dayOfMeanFlow_mean_map.py | dayOfMeanFlowThroughTime.csv |  | 4 | Panel D
ml_visualize_dayOfMeanFlow_throgh_time_map.py | dayOfMeanFlowThroughTime.csv |  | 4 | Panel C
ml_encodings_means.py | ml_slope_encodings1.csv |  | 4 | Panel F
ml_encodings_slopes.py  | ml_slope_encodings1.csv |  | 4 | Panel E
ml_masd_map.py | specific_discharge_vs_size.csv |  | 4 | Panel B
ml_visualizele_masd_through_time2.py | specific_discharge_vs_size.csv |  | 4 | Panel A
buildGif.py | gif_figures |  | movie1 |
makeFinalFlowGif.py | localWaterYear | gif_figures |  |
plotExampleHydrographs.py | universallyAligned_powersTranspose.csv |  | S1 | Panels b and d
getExampleStreams2.py | FullDatabase.csv |  | S1 | Panels a and c
getExampleStreams2.py | localWaterYear |  |  |
compileLoss.py | ml_exampleXslope_encodings_loss_.csv |  | S2 |
compressTemporalPatternsSlopeExample.py | ml_all_years_data_separate.csv | ml_exampleXslope_encodings_loss_.csv | S3 |
example_spectral_properties_hydrographs.py | alldata.csv |  | S4 | All frames
threeDPlot.py | ml_encodings_1.csv |  | S5 | All frames
threeDPlot.py | ml_encodings_1.csv |  | S6 | All frames
structureBetweenVariables.py | throughTimeCombined.csv |  | S7 | All frames
structureBetweenVariables.py | throughTimeCombined.csv |  | S8 | All frames
plotDayOfMeanFlowThroughTimeTogether.py | dayOfMeanFlowThroughTime.csv |  | S9 | "Panels D, E, and F"
plotSpecificDischargeThroughTimeTogether.R | specificDischargeThroughTime.csv |  | S9 | "Panels A, B, and C"
plotSpectralNumThroughTimeTogether.R | spectralNumber_acrossTime.csv |  | S9 | "Panels G, H, and I"
plotSpectralThroughTime.R | spectralPowersThroughTime.csv |  | S10 |
ml_encodings_means_temp.py | alldata.csv |  | S11 | Panel C
ml_encodings_slopes_temp.py | alldata.csv |  | S11 | Panel F
ml_masd_map.py | alldata.csv |  | S11 |
ml_visualize_dayOfMeanFlow_through_time2.py | alldata.csv |  | S11 |
ml_visualize_dayOfMeanFlow_through_time2.py | dayOfMeanFlowThroughTime.csv |  | S11 | Panel E
ml_encodings_slopes_temp.py | ml_slope_encodings1.csv |  | S11 |
ml_masd_map.py | specific_discharge_vs_size.csv |  | S11 | Panel A
ml_visualize_dayOfMeanFlow_mean_map.py | dayOfMeanFlowThroughTime.csv |  | S11 | Panel B
slopeThroughTime.py | specific_discharge_vs_size.csv |  | S11 | Panel D
ml_encodings_means.py | alldata.csv |  | S12 |
ml_visualize_dayOfMeanFlow_mean_map.py | alldata.csv |  | S12 |
ml_visualize_dayOfMeanFlow_mean_map.py | dayOfMeanFlowThroughTime.csv |  | S12 | Panel B
ml_encodings_means.py | ml_slope_encodings1.csv |  | S12 | Panel C
ml_masd_map.py | specific_discharge_vs_size.csv |  | S12 | Panel A
plotDayOfMeanFlow.R | day_of_mean_flow_vs_size.csv |  | S13 | Panel B
plotSpecificDischarge.R  | specific_discharge_vs_size.csv |  | S13 | Panel A
ml_encodings_slopes_pvalues.py | alldata.csv |  | S14 | Panels E and F
ml_visualize_dayOfMeanFlow_through_time_map_pvalues.py | alldata.csv |  | S14 |
ml_visualizele_masd_through_time2_pvals.py | alldata.csv |  | S14 |
ml_visualize_dayOfMeanFlow_through_time_map_pvalues.py | dayOfMeanFlowThroughTime.csv |  | S14 | Panels C and D
ml_encodings_slopes_pvalues.py | ml_slope_encodings1.csv |  | S14 |
ml_visualizele_masd_through_time2_pvals.py | specific_discharge_vs_size.csv |  | S14 | panels A and B
dayOfMeanFlowThroughTime.py | alldata.csv | dayOfMeanFlowThroughTime.csv |  |
spectralThroughTime.py | alldata.csv | spectralPowersThroughTime.csv |  |
splitIntoNonGlobalWaterYears.py | alldata.csv | localWaterYear |  |
structureBetweenVariables.py | alldata_hemisphereCorrected.csv |  |  |
structureBetweenVariables.py | alldata_hemisphereCorrected.csv |  |  |
structureBetweenVariables.py | alldata_hemisphereCorrected.csv |  |  |
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
