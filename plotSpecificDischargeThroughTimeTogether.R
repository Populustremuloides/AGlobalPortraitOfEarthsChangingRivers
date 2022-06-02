library(ggplot2)
library(tidyverse)
library(scales)

df = read_csv("specificDischargeThroughTime.csv")
df = filter(df, year <= 2010)
coldDf = filter(df, category == "cold and small" | category == "cold and large")
print(coldDf)
ggplot(coldDf, aes(x=year,y=mean_annual_specific_discharge, color=category)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("cold, max small n = 781, max large n = 903") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("specificDischargeThroughTime_cold.png")

# ************************************************

medDf = filter(df, category == "med and small" | category == "med and large")
ggplot(medDf, aes(x=year,y=mean_annual_specific_discharge, color=category)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("med, max small n = 552, max large n = 739") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("specificDischargeThroughTime_med.png")


# ************************************************

hotDf = filter(df, category == "hot and small" | category == "hot and large")
print(hotDf)
ggplot(hotDf, aes(x=year,y=mean_annual_specific_discharge, color=category)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("hot, max small n = 71, max large n = 215") +
	theme(plot.title = element_text(hjust = 0.5))
ggsave("specificDischargeThroughTime_hot.png")

