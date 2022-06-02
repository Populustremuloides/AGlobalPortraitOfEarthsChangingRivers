library(ggplot2)
library(tidyverse)
library(scales)

df = read_csv("dayOfMeanFlowThroughTime.csv")
df = filter(df, year <= 2010)

coldDf = filter(df, category == "cold and small" | category == "cold and large")
print(coldDf)
ggplot(coldDf, aes(x=year,y=day_of_mean_flow, color=category)) +
	geom_smooth() +
	theme_bw() + 
        ylim(100,300) +
	ggtitle("cold, max n small = 781, max n large = 903") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("dayOfMeanFlowThroughTime_cold.png")

# ************************************************

medDf = filter(df, category == "med and small" | category == "med and large")
ggplot(medDf, aes(x=year,y=day_of_mean_flow, color=category)) +
	geom_smooth() +
	theme_bw() + 
        ylim(100,300) +
	ggtitle("med, max n small = 552, max n large = 739") + 
	theme(plot.title = element_text(hjust = 0.5))
ggsave("dayOfMeanFlowThroughTime_med.png")

# ************************************************

hotDf = filter(df, category == "hot and small" | category == "hot and large")
print(hotDf)
ggplot(hotDf, aes(x=year,y=day_of_mean_flow, color=category)) +
	geom_smooth() +
	theme_bw() + 
        ylim(100,300) +
	ggtitle("hot, max n small = 71, max n large = 215") +
	theme(plot.title = element_text(hjust = 0.5))
ggsave("dayOfMeanFlowThroughTime_hot.png")




