#install.packages("WaveletComp", repos='http://cran.us.r-project.org')
library("WaveletComp")

rootDir = "/home/sethbw/Documents/GlobFlow/spectralAnalysis"
outDir = "/home/sethbw/Documents/GlobFlow/spectralAnalysis"





dataFile = "universallyAlignedGlobalFlow_DailyQ2_column.csv"

dataFolder = paste(rootDir, dataFile, sep="/")
print(dataFolder)
data = read.csv(dataFolder, sep=",")
#print(data)
originalData = data.frame(data)

sites = names(data)
sites


getStartingIndex = function(siteData) {
	startingIndex = 0
	for (indx in 1:(length(siteData))) {
		if (!is.na(siteData[indx])) { 
			startingIndex = indx
		        break           
		}
	}
	return(startingIndex)
}

getStoppingIndex = function(startingIndex, siteData) {
	stoppingIndex = 0
	for (indx in startingIndex:(length(siteData))) {
		if (is.na(siteData[indx])) { 
			stoppingIndex = indx - 1
		        break           
		}
	}
	    if (stoppingIndex == 0) {
        	stoppingIndex = length(siteData)
    	}
	return(stoppingIndex)
}


getTrimmedData = function(originalData, site) {
	siteData = originalData[,site]
	startingIndex = getStartingIndex(siteData)
	stoppingIndex = getStoppingIndex(startingIndex, siteData)
	print(startingIndex)
	print(stoppingIndex)
	return(originalData[startingIndex:stoppingIndex,])
}


site = sites[[5]] # test site
trimmedData = getTrimmedData(originalData, site)

wvlt = analyze.wavelet(trimmedData, site,loess.span = 0,
               dt = 1, dj = 1/100,lowerPeriod = 2,
               upperPeriod = 4096,make.pval = F)

wt.image(wvlt, color.key = "quantile", 
        n.levels = 100,legend.params = list(lab = "wavelet power levels", mar = 4.7))

wt.avg(wvlt)

scale = wvlt$Scale
power = data.frame(scale)
power
pval = data.frame(scale)
pval

for (indx in 4:length(sites)) {
	currentSite = sites[indx]
	trimmedData = getTrimmedData(originalData, currentSite)
    
	tryCatch ({
		wvlt = analyze.wavelet(trimmedData, currentSite,loess.span = 0,
                dt = 1, dj = 1/100,lowerPeriod = 2,
                upperPeriod = 4096,make.pval = F)

		power[[currentSite]] = wvlt$Power.avg
		#pval[[currentSite]] = wvlt$Power.avg.pval
	    	print(indx)         
	}, error=print)
}


outFile = "universallyAligned_powers.csv"
write.csv(power,paste(outDir, outFile, sep="/"), row.names=FALSE)
#outFile = "universallyAligned_pVals.csv"
#write.csv(pval,paste(outDir, outFile, sep="/"), row.names=FALSE)


tPower = t(power)
#tPval = t(pval)

outFile = "universallyAligned_powersTranpose.csv"
write.csv(tPower, paste(outDir, outFile, sep="/"), row.names=TRUE)
#outFile = "universallyAligned_pValsTranspose.csv"
#write.csv(tPval,paste(outDir, outFile, sep="/"), row.names=FALSE)
