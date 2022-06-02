#install.packages("WaveletComp", repos='http://cran.us.r-project.org')
library("WaveletComp")

outDir = "/home/sethbw/Documents/GlobFlow/localWaterYearSpectralDecomposition"

rootDir = "/home/sethbw/Documents/GlobFlow/localWaterYear"

prefix = "globFlowData_localWaterYear_"

for (fix in 2016:2017) {
    	postFix = toString(fix)

	dataFile = paste(prefix, postFix, sep="")
	dataFile = paste(dataFile, ".csv", sep="")

	dataFolder = paste(rootDir, dataFile, sep="/")
	print(dataFolder)
	data = read.csv(dataFolder, sep=",")
	#print(data)
	originalData = data.frame(data) 
	print(originalData[,3:5])
	sites = names(data)
	sites

	getStartingIndex = function(siteData) {
		startingIndex = 1
		for (indx in 1:(length(siteData))) {
			if (!is.na(siteData[indx])) { 
				startingIndex = indx
			        break      
			}
		}
		return(startingIndex)
	}

	getStoppingIndex = function(startingIndex, siteData) {
		stoppingIndex = 1
		for (indx in startingIndex:(length(siteData))) {
			if (is.na(siteData[indx])) { 
				stoppingIndex = indx
			        break           
			}
		}
		if (stoppingIndex == 1) {
	        	stoppingIndex = length(siteData)
	    	}
		return(stoppingIndex)
	}

	getTrimmedData = function(originalData, site) {
		siteData = originalData[,site]
		startingIndex = getStartingIndex(siteData)
		stoppingIndex = getStoppingIndex(startingIndex, siteData)
		return(originalData[startingIndex:stoppingIndex,])
	}
	
	site = sites[[3]] # test site
	trimmedData = getTrimmedData(originalData, site)

	wvlt = analyze.wavelet(trimmedData, site,loess.span = 0,
               dt = 1, dj = 1/100,lowerPeriod = 2,
               upperPeriod = 365,make.pval = F)

	wt.image(wvlt, color.key = "quantile", 
	        n.levels = 100,legend.params = list(lab = "wavelet power levels", mar = 4.7))

	wt.avg(wvlt)

	scale = wvlt$Scale
	power = data.frame(scale)
	power
	pval = data.frame(scale)
	pval

	for (indx in 1:length(sites)) {
		currentSite = sites[indx]
		trimmedData = getTrimmedData(originalData, currentSite)
		tryCatch ({
			wvlt = analyze.wavelet(trimmedData, currentSite,loess.span = 0,
	                      dt = 1, dj = 1/100,lowerPeriod = 2,
	                      upperPeriod = 365,make.pval = F, n.sim = 1)

			power[[currentSite]] = wvlt$Power.avg
			pval[[currentSite]] = wvlt$Power.avg.pval
	    		print(indx)         
	    	}, error=print)
	}

	outFile = paste(prefix, postFix, sep="")
	outFile = paste(outFile, "_FlowPeriodPowers.csv", sep="")
	write.csv(power,paste(outDir, outFile, sep="/"), row.names=FALSE)
	outFile = paste(prefix, postFix, sep="")
	outFile = paste(outFile, "_FlowPeriodPowerPvals.csv", sep="")
	write.csv(pval,paste(outDir, outFile, sep="/"), row.names=FALSE)

	power

	tPower = t(power)
	tPval = t(pval)

	outFile = paste(prefix, postFix, sep="")
	outFile = paste(outFile, "_FlowPowersTranspose.csv", sep="")
	write.csv(tPower, paste(outDir, outFile, sep="/"), row.names=TRUE)
	outFile = paste(prefix, postFix, sep="")
	outFile = paste(outFile, "_FlowPeriodPowerPvalsTranspose.csv", sep="")
	write.csv(tPval,paste(outDir, outFile, sep="/"), row.names=FALSE)

}
