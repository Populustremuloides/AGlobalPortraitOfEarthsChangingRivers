library(ggplot2)
library(tidyverse)

df = read_csv("specific_discharge_vs_size.csv")
print(df)


ggplot(df, aes(x=gord,y=specific_discharge, color=precip, alpha=0.7)) +
	geom_jitter() + 
	geom_smooth() +
	theme_bw() +
	ylab("mean annual specific discharge") + 
	xlab("year") +
        scale_color_gradient(low="red",high="blue")

ggsave("specific_discharge_vs_order_precip.png")


# two figures left:
# the gif
# 3d plot mean annual specific discharge, day of mean flow, single-dimensional neural network encoding

