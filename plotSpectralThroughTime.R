library(ggplot2)
library(tidyverse)
library(scales)

df = read_csv("spectralPowersThroughTime.csv")
df$mean_period = as.factor(df$mean_period)
df = filter(df, year <= 2010)
# ************************************************

coldDf = filter(df, category == "cold and small")
ggplot(coldDf, aes(x=year,y=mean_power,color=mean_period)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("cold and small, max n = 781") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralThroughTime_coldSmall.png")


coldDf = filter(df, category == "cold and large")
ggplot(coldDf, aes(x=year,y=mean_power,color=mean_period)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("cold and large, max n = 903") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralThroughTime_coldLarge.png")


# ************************************************


medDf = filter(df, category == "med and small")
ggplot(medDf, aes(x=year,y=mean_power,color=mean_period)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("med and small, max n = 552") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralThroughTime_medSmall.png")

medDf = filter(df, category == "med and large")
ggplot(medDf, aes(x=year,y=mean_power,color=mean_period)) +
	geom_smooth() +
	theme_bw() +
	ggtitle("med and large, max n =  739") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralThroughTime_medLarge.png")


# ************************************************

hotDf = filter(df, category == "hot and small")
print(hotDf)
ggplot(hotDf, aes(x=year,y=mean_power,color=mean_period)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("hot and small, max n = 71") +
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralThroughTime_hotSmall.png")

hotDf = filter(df, category == "hot and large")
print(hotDf)
ggplot(hotDf, aes(x=year,y=mean_power,color=mean_period)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("hot and large, max n = 215") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralThroughTime_hotLarge.png")



#ggsave("combinedImportances.png", width=10,height=4)





