library(ggplot2)
library(tidyverse)
library(scales)

df = read_csv("spectralNumber_acrossTime.csv")
df = filter(df, year_y <= 2010)
coldDf = filter(df, category == "cold and small" | category == "cold and large")
print(coldDf)
ggplot(coldDf, aes(x=year_y,y=encoding, color=category)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("cold, max small n = 781, max large n = 903") +
	ylim(0,1) +
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralNumThroughTime_cold.png")

# ************************************************

medDf = filter(df, category == "med and small" | category == "med and large")
ggplot(medDf, aes(x=year_y,y=encoding, color=category)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("med, max small n = 552, max large n = 739") + 
	ylim(0, 1) +
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralNumThroughTime_med.png")


# ************************************************

hotDf = filter(df, category == "hot and small" | category == "hot and large")
print(hotDf)
ggplot(hotDf, aes(x=year_y,y=encoding, color=category)) +
	geom_smooth() +
	theme_bw() + 
	ggtitle("hot, max small n = 71, max large n = 215") +
	ylim(0,1) +
	theme(plot.title = element_text(hjust = 0.5))
ggsave("spectralNumThroughTime_hot.png")

