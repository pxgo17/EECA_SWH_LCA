library(tidyverse)

manuf_emissions <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/raw/lca/manufacture.csv")

manuf_emissions %>% ggplot(aes(x = Capacity, y = Emissions, color = Region)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal()


model <- lm(Emissions ~ Region + Weight, data = manuf_emissions)
summary(model)


manuf_energy <- read_csv("D:/test_data/embodiedEnergyAppliances.csv")

model_energy_Asia <- lm(Manu_MJ ~ Weight, data = manuf_energy %>% filter(Region == 'Asia'))

model_energy_Europe <- lm(Manu_MJ ~ Weight, data = manuf_energy %>% filter(Region == 'Europe'))


#summary(model_energy)

manuf_energy <- manuf_energy %>%
  mutate(Manu_MJ_pred_Asia = predict(model_energy_Asia, newdata = .)) %>%
  mutate(Manu_MJ_pred_Europe = predict(model_energy_Europe, newdata = .))
  

manuf_energy %>% ggplot(aes(x=Weight, y=Manu_MJ, color=Region)) +
  geom_point() + 
  geom_line(aes(x=Weight,y=Manu_MJ_pred_Asia), color  = 'red') +
  geom_line(aes(x=Weight,y=Manu_MJ_pred_Europe), color = 'blue') +
  scale_x_continuous(limits = c(0, NA))
  

manuf_energy %>% summary()

library(readxl)
library(openxlsx)

df_core <- read.xlsx("D:/test_data/FCU393.xlsx", rows = 323:336)
df_core <- df_core %>%
  mutate(across(c(3:7, 13:20), ~as.numeric(gsub(",", ".", gsub("[‐–−]", "-", .)))))

df_nr <- read.xlsx("D:/test_data/FCU393.xlsx", rows = 348:358)
df_nr <- df_nr %>%
  mutate(across(c(3:7, 13:20), ~as.numeric(gsub(",", ".", gsub("[‐–−]", "-", .)))))

df_core %>% write_csv("D:/test_data/FCU393_core.csv")
df_nr %>% write_csv("D:/test_data/FCU393_nr.csv")

df_core_2 <- read.xlsx("D:/test_data/Saga_200.xlsx", rows = 79:92)
df_core_2 <- df_core_2 %>%
  mutate(across(c(3:17), ~as.numeric(gsub(",", ".", gsub("[‐–−]", "-", .)))))

df_nr_2 <- read.xlsx("D:/test_data/Saga_200.xlsx", rows = 104:114)
df_nr_2 <- df_nr_2 %>%
  mutate(across(c(3:17), ~as.numeric(gsub(",", ".", gsub("[‐–−]", "-", .)))))

df_core_2 %>% write_csv("D:/test_data/Saga200_core.csv")
df_nr_2 %>% write_csv("D:/test_data/Saga200_nr.csv")
