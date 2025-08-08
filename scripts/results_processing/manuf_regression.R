library(tidyverse)

manuf_emissions <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/raw/lca/manufacture.csv")

manuf_emissions %>% ggplot(aes(x = Capacity, y = Emissions, color = Region)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal()


model <- lm(Emissions ~ Region + Weight, data = manuf_emissions)
summary(model)
