# A Global Portrait Of Earths Changing Rivers
Analysis Code for the paper, A Global Portrait of Earth's Changing Rivers: Smaller, Warmer Catchments are Most at Risk.

The data used and generated by this repository will be made available following publication. During the review process, data will be made available to reviewers
while the code listed here will continue to be public.

The Python code was executed using version 3.7. To run the code, you will need the following packages installed in your environment:

The following table represents the dependency graph between each figure in the manuscript and the code and data files used to generate it, as well as the parent code and data files all the way back to the original data files. The intermediary processing files will be made available following publication, but the raw files used in this analysis are currently published on Research Gate [here](https://www.researchgate.net/publication/354085237_Daily_Streamflow_Data_hemisphere_corrected#fullTextFileContent) and [here](https://www.researchgate.net/publication/354080229_Streamflow_Metrics_and_Catchment_Characteristics_for_Global_Streamflow_Dataset). The methods used for generating these data can be found in [this preprint](https://www.researchgate.net/publication/354197150_The_Music_of_Rivers_How_the_Mathematics_of_Waves_Reveals_Global_Drivers_of_Streamflow_Regime).


code file | input file | output file | figure produced (if any) | Notes
-------------- | ---- | -------- | ------ | -----
threeDPlot.py | day_of_mean_flow_vs_size.csv |  | 1 | All panels
threeDPlot.py | ml_slope_encodings1.csv |  | 1 | 
threeDPlot.py | specific_discharge_vs_size.csv |  | 1 | 
structureBetweenVariables.py | throughTimeCombined.csv |  | 2 | All panels
ml_encodings_means.py | alldata.csv |  | 3 | 
ml_encodings_slopes.py  | alldata.csv |  | 3 | 
ml_visualize_dayOfMeanFlow_mean_map.py | alldata.csv |  | 3 | 
ml_visualize_dayOfMeanFlow_throgh_time_map.py | alldata.csv |  | 3 | 
ml_visualizele_masd_through_time2.py | alldata.csv |  | 3 | 
ml_visualize_dayOfMeanFlow_mean_map.py | dayOfMeanFlowThroughTime.csv |  | 3 | Panel D
ml_visualize_dayOfMeanFlow_throgh_time_map.py | dayOfMeanFlowThroughTime.csv |  | 3 | Panel C
ml_encodings_means.py | ml_slope_encodings1.csv |  | 3 | Panel F
ml_encodings_slopes.py  | ml_slope_encodings1.csv |  | 3 | Panel E
ml_masd_map.py | specific_discharge_vs_size.csv |  | 3 | Panel B
ml_visualizele_masd_through_time2.py | specific_discharge_vs_size.csv |  | 3 | Panel A
compileLoss.py | ml_exampleXslope_encodings_loss_.csv |  | S1 | 
ml_encodings_means_temp.py | alldata.csv |  | S10 | Panel C
ml_encodings_slopes_temp.py | alldata.csv |  | S10 | Panel F
ml_masd_map.py | alldata.csv |  | S10 | 
ml_visualize_dayOfMeanFlow_through_time2.py | alldata.csv |  | S10 | 
ml_visualize_dayOfMeanFlow_through_time2.py | dayOfMeanFlowThroughTime.csv |  | S10 | Panel E
ml_encodings_slopes_temp.py | ml_slope_encodings1.csv |  | S10 | 
ml_masd_map.py | specific_discharge_vs_size.csv |  | S10 | Panel A
|  |  |  | S10 | Panel B
|  |  |  | S10 | Panel D
ml_encodings_means.py | alldata.csv |  | S11 | 
ml_visualize_dayOfMeanFlow_mean_map.py | alldata.csv |  | S11 | 
ml_visualize_dayOfMeanFlow_mean_map.py | dayOfMeanFlowThroughTime.csv |  | S11 | Panel B
ml_encodings_means.py | ml_slope_encodings1.csv |  | S11 | Panel C
ml_masd_map.py | specific_discharge_vs_size.csv |  | S11 | Panel A
plotDayOfMeanFlow.R | day_of_mean_flow_vs_size.csv |  | S12 | Panel B
plotSpecificDischarge.R  | specific_discharge_vs_size.csv |  | S12 | Panel A
ml_encodings_slopes_pvalues.py | alldata.csv |  | S13 | Panels E and F
ml_visualize_dayOfMeanFlow_through_time_map_pvalues.py | alldata.csv |  | S13 | 
ml_visualizele_masd_through_time2_pvals.py | alldata.csv |  | S13 | 
ml_visualize_dayOfMeanFlow_through_time_map_pvalues.py | dayOfMeanFlowThroughTime.csv |  | S13 | Panels C and D
ml_encodings_slopes_pvalues.py | ml_slope_encodings1.csv |  | S13 | 
ml_visualizele_masd_through_time2_pvals.py | specific_discharge_vs_size.csv |  | S13 | panels A and B
compressTemporalPatternsSlopeExample.py | ml_all_years_data_separate.csv | ml_exampleXslope_encodings_loss_.csv | S2 | 
example_spectral_properties_hydrographs.py | alldata.csv |  | S3 | All panels
threeDPlot.py | ml_encodings_1.csv |  | S4 | All panels
threeDPlot.py | ml_encodings_1.csv |  | S5 | All panels
structureBetweenVariables.py | throughTimeCombined.csv |  | S6 | All panels
structureBetweenVariables.py | throughTimeCombined.csv |  | S7 | All panels
plotDayOfMeanFlowThroughTimeTogether.py | dayOfMeanFlowThroughTime.csv |  | S8 | "Panels D, E, and F"
plotSpecificDischargeThroughTimeTogether.R | specificDischargeThroughTime.csv |  | S8 | "Panels A, B, and C"
plotSpectralNumThroughTimeTogether.R | spectralNumber_acrossTime.csv |  | S8 | "Panels G, H, and I"
plotSpectralThroughTime.R | spectralPowersThroughTime.csv |  | S9 | 
dayOfMeanFlowThroughTime.py | alldata.csv | dayOfMeanFlowThroughTime.csv |  | 
spectralThroughTime.py | alldata.csv | spectralPowersThroughTime.csv |  | 
splitIntoNonGlobalWaterYears.py | alldata.csv | localWaterYear |  | 
structureBetweenVariables.py | alldata_hemisphereCorrected.csv |  |  | 
structureBetweenVariables.py | alldata_hemisphereCorrected.csv |  |  | 
structureBetweenVariables.py | alldata_hemisphereCorrected.csv |  |  | 
plotSpectralNumberThroughTime.py | day_of_mean_flow_vs_size.csv | spectralNumber_acrossTime.csv |  | 
combineThroughTime.py | dayOfMeanFlowThroughTime.csv | throughTimeCombined.csv |  | 
testDifferenceInDayOfMeanFlow.py | FullDatabase.csv | day_of_mean_flow_vs_size.csv |  | 
testDifferencesInMean.py | FulLDatabase.csv | specific_discharge_vs_size.csv |  | Convert discharge to specific disharge data
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
