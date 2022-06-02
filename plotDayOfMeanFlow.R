library(ggplot2)
library(tidyverse)

df = read_csv("day_of_mean_flow_vs_size.csv")
print(df)

ggplot(df, aes(x=gord,y=day_of_mean_flow, color=temp, alpha=0.5)) +
	geom_jitter() + 
	geom_smooth() +
	theme_bw() +
	ylab("day of mean flow") +
	xlab("stream order") +
        scale_color_gradient(low="blue", high="red", space ="Lab")

ggsave("day_of_mean_flow_vs_order_temp.png")



