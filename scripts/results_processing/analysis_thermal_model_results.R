library(tidyverse)
library(patchwork)
library(ggpattern)

# Read csv file with results for SWH technologies
# Files can be read from file (commented line or from github)
# If reading from file, please make sure to change directories accordingly
#raw_technologies <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/thermal/technologies.csv")
raw_technologies <- read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/processed/thermal/technologies.csv")
# Files can be read from file (commented line or from github)
#embodied <- read_csv("C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP/SWH_capacity_processing/embodied.csv")
embodied <- read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/processed/lca/embodied.csv")

# Filters by SH_code
embodied_SH_nonHYD <- embodied %>% filter(SH_code %in% c('R','Rb','Rl','HP','HPl')) %>%
  # New column for schedule is needed for non-hydronic SH subset
  mutate(schedule = word(Hshld_code, 3, sep = fixed("-")))
embodied_SH_HYD <- embodied %>% filter(SH_code %in% c('HYD200','HYD500','HYD1000')) 
# Filter by DW_code
embodied_DW <- embodied %>% filter(DW_code %in% c('R','Ro','HP','HPo'))

# Processing starts from now, first generating new columns
raw_technologies_processed <- raw_technologies %>%
  separate(col = Hshld_code, # Column to split
           into = c("house_type", "insulation", "schedule"), # New column names
           sep = "-") %>% # Separator character
  separate(col = Grid_code, # Column to split
           into = c("Grid_year", "Hydro_resource"), # New column names
           sep = "-")

# Add DW embodied emissions; join only based on DW_code and tank volume
# as these (capacity technology combinations) will be later added to all house types
raw_technologies_processed_full <- raw_technologies_processed %>%
  # Note only three columns are selected before join
  left_join(embodied_DW %>% select(DW_code,`Tank_Volume (L)`,Embodied), 
            by = c("DW_code", "Tank_Volume (L)"))

# Add non hydronic SH embodied emissions; join includes multiple attributes
# as capacities change depending on house type, insulation, location
raw_technologies_processed_full <- raw_technologies_processed_full %>%
  # HP_cap in included as original technologies dataset was missing this information for ASHP
  # R_cap is introduced here (missing in technologies dataset)
  left_join(embodied_SH_nonHYD %>% select(house_type,insulation,Loc_code,schedule,SH_code,HP_cap,R_cap,Embodied),
            by=c("house_type","insulation","Loc_code","schedule","SH_code"), suffix = c("_left", "_right")) %>%
  # Use coalesce(embodied.y, embodied.x) to prefer the right table’s value when there’s a match, 
  # otherwise keep the left table’s value (left table is technologies dataset which only
  # includes capacities for hydronic).
  mutate(Embodied = coalesce(Embodied_right, Embodied_left)) %>%
  select(-Embodied_left, -Embodied_right) %>%
  # apply coalesce approach to address HP_cap as well
  mutate(HP_cap = coalesce(HP_cap_right, HP_cap_left)) %>%
  select(-HP_cap_left, -HP_cap_right)

# Add hydronic SH embodied emissions; similar to previous join,
# but in this case HP_cap is included as was previously defined in technologies
# dataset. New columns (embodied HP, FC, tank) will be introduced with this join
# to disaggregate components for Hydronic scenarios. A Hyd_Cap_kW_Adj
# is also introduced, which documents nominal capacity used from
# embodied emissions table.
# Also, note that tank volume was provided in original technologies 
# dataset (consistent with SH_code and DW_code)

raw_technologies_processed_full <- raw_technologies_processed_full %>%
  left_join(embodied_SH_HYD %>% select(house_type,insulation,Loc_code,SH_code,HP_cap,Embodied,
                                       Hyd_Cap_kW_Adj, Hyd_Embodied_HP,
                                       Hyd_Embodied_Tank, Hyd_Embodied_FC),
            by=c("house_type","insulation","Loc_code","SH_code","HP_cap"), suffix = c("_left", "_right")) %>%
  # Use coalesce(embodied.y, embodied.x) to prefer the right table’s value when there’s a match, 
  # otherwise keep the left table’s value.
  mutate(Embodied = coalesce(Embodied_right, Embodied_left)) %>%
  select(-Embodied_left, -Embodied_right) 

##### On a technology basis, interpolation will be applied to work out
##### cumulative energy demand and emissions (a similar approach is later
##### implemented on a household basis)

# Set up the years you want to interpolate for
# Same period applies for household interpolation
desired_years <- 2025:2045

# Interpolating 'value' for each group (defined by scenario)
technology_year <- raw_technologies_processed_full %>% 
  mutate(year = as.integer(Grid_year)) %>%
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_code,DW_code,Occ_code,
           R_cap, HP_cap, `Tank_Volume (L)`) %>%
  complete(year = desired_years) %>% # creates rows for all years
  arrange(year) %>%
  mutate(kgCO2_oper_year = approx(year, kgCO2, xout = year, rule = 2)$y) %>%
  # Note that Embodied are divided over period of analysis (20 years)
  # to work out annual share
  mutate(kgCO2_embod_year = approx(year, Embodied/20, xout = year, rule = 2)$y) %>%
  # Electricity energy demand should be constant, average power may change 
  # with control signal
  mutate(P_annual_kWh_year = approx(year, `P_annual (kWh)`, xout = year, rule = 2)$y) %>%
  mutate(P_avg_peak_kW_year = approx(year, `P_avg_peak (kW)`, xout = year, rule = 2)$y) %>%
  ungroup() 

# Please make sure you change to desired directory (latest version was saved in github repo)
technology_year %>% write_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/thermal/technology_year.csv")

# Calculate operational and embodied emissions over lifespan
technology_lifetime <- technology_year %>% 
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_code,DW_code,Occ_code,
           R_cap, HP_cap, `Tank_Volume (L)`) %>%
  summarise(kgCO2_oper_life = sum(kgCO2_oper_year),
            P_annual_kWh_life = sum(P_annual_kWh_year),
            kgCO2_embod_life = kgCO2_embod_year
            ) %>% ungroup() %>%
  mutate(kgCO2_total_life = kgCO2_oper_life + kgCO2_embod_life)
  

# Please make sure you change to desired directory (latest version was saved in github repo)
technology_lifetime %>% write_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/thermal/technology_lifetime.csv")

####### COMBINATIONS FOR  development of household dataset  
# First SH combination for all resistive SH_code ('R')
# No grouping required as data is already aggregated
raw_technologies_processed_SH_R <- raw_technologies_processed_full %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code == 'R') %>% # Second filter only resistive (R)
  select(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource, SH_code, HP_cap, R_cap,
         `P_annual (kWh)`, `Q_annual (kWh)`, kgCO2, `P_avg_peak (kW)`,Embodied) %>%
  rename(
    P_annual_kWh_SH = `P_annual (kWh)`,
    Q_annual_kWh_SH = `Q_annual (kWh)`,
    P_avg_peak_kW_SH = `P_avg_peak (kW)`,
    kgCO2_SH =  kgCO2,
    Embodied_SH = Embodied)

# Second SH combination for all heat pump
# No grouping required as data is already aggregated
raw_technologies_processed_SH_HP <- raw_technologies_processed_full %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code == 'HP') %>% # Second filter only heat pump (HP)
  select(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource, SH_code, HP_cap, R_cap,
         `P_annual (kWh)`, `Q_annual (kWh)`, kgCO2, `P_avg_peak (kW)`,Embodied) %>%
  rename(
    P_annual_kWh_SH = `P_annual (kWh)`,
    Q_annual_kWh_SH = `Q_annual (kWh)`,
    P_avg_peak_kW_SH = `P_avg_peak (kW)`,
    kgCO2_SH =  kgCO2,
    Embodied_SH = Embodied)

# Third SH combination for Living heat pump, Bedroom Resistive
# Grouping required 
raw_technologies_processed_SH_HP_R <- raw_technologies_processed_full %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code %in% c('HPl','Rb')) %>% # Second filter 
  group_by(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource) %>%
  summarise(P_annual_kWh_SH = sum(`P_annual (kWh)`),
            Q_annual_kWh_SH = sum(`Q_annual (kWh)`),
            kgCO2_SH = sum(kgCO2),
            P_avg_peak_kW_SH = sum(`P_avg_peak (kW)`),
  # embodied table includes SH capacity for both resistive and HP,
  # meaning that in all cases there will two values, one for HP_cap,
  # and one for R_cap, and one will be NA.  
            HP_cap = sum(HP_cap, na.rm = TRUE), 
            R_cap = sum(R_cap, na.rm = TRUE),
            Embodied_SH = sum(Embodied)) %>% ungroup() %>%
  mutate(SH_code = 'HPl_Rb')

# Fourth combination is for hydronic (SH and WH),
# In this case occupancy matters because includes WH
# No grouping needed, hydronic heats entire house space and water
# Extra columns selected (occupancy and HP capacity)
raw_technologies_processed_Hydronic <- raw_technologies_processed_full %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code %in% c("HYD1000", "HYD200",  "HYD500")) %>% # Second filter only heat pump (HP)
  select(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource,
         SH_code, HP_cap, DW_code, `Tank_Volume (L)`, Occ_code, HP_cap,
         `P_annual (kWh)`, `Q_annual (kWh)`, kgCO2, `P_avg_peak (kW)`, Embodied) %>%
  rename(
    P_annual_kWh_total = `P_annual (kWh)`,
    Q_annual_kWh_total = `Q_annual (kWh)`,
    P_avg_peak_kW_total = `P_avg_peak (kW)`,
    kgCO2_total = kgCO2) %>%
  mutate(scenario = paste(house_type,insulation,schedule,
                          Loc_code,Occ_code, SH_code,
                          DW_code, 'C0',Grid_year,Hydro_resource, sep = "-"))

# Fifth combination is for HP WH,
# In this case occupancy matters because includes WH
raw_technologies_processed_WH_HP <- raw_technologies_processed_full %>%
  filter(!is.na(DW_code)) %>% # First filter only water heating (DW)
  filter(DW_code %in% c("HP", "HPo")) %>% # Second filter to leave only HP
  # Embodied is included in grouping variables to keep in final result
  group_by(Loc_code, Grid_year, Hydro_resource, Occ_code, DW_code, `Tank_Volume (L)`) %>% # Note location included as heat pump performace relies on weather
  summarise(P_annual_kWh_WH = sum(`P_annual (kWh)`),
            Q_annual_kWh_WH = sum(`Q_annual (kWh)`),
            kgCO2_WH = sum(kgCO2),
            P_avg_peak_kW_WH = sum(`P_avg_peak (kW)`),
            Embodied_WH = sum(Embodied)) %>% ungroup()

# Sixth combination is for R WH,
# In this case occupancy matters because includes WH
# Location code note required for resistive DWH
raw_technologies_processed_WH_R <- raw_technologies_processed_full %>%
  filter(!is.na(DW_code)) %>% # First filter only water heating (DW)
  filter(DW_code %in% c("R", "Ro")) %>% # Second filter to leave only HP
  group_by(Grid_year, Hydro_resource, Occ_code, DW_code, `Tank_Volume (L)`) %>% # Note location not included
  summarise(P_annual_kWh_WH = sum(`P_annual (kWh)`),
            Q_annual_kWh_WH = sum(`Q_annual (kWh)`),
            kgCO2_WH = sum(kgCO2),
            P_avg_peak_kW_WH = sum(`P_avg_peak (kW)`),
            Embodied_WH = sum(Embodied)) %>% ungroup()

# Build table for non-hydronic combinations with HP WH
# First need to combine all non-hydronic space heating
# Combine dataframes into a list
non_hydronic_SH_dataframes <- list(raw_technologies_processed_SH_R,
                                   raw_technologies_processed_SH_HP,
                                   raw_technologies_processed_SH_HP_R)

# Merge all SH dataframes into one
raw_technologies_processed_SH <- bind_rows(non_hydronic_SH_dataframes)

# Join by parts because HP technologies
# rely on weather/zone attributes
household_SH_WH_HP <- raw_technologies_processed_SH %>%
  left_join(raw_technologies_processed_WH_HP, 
            by = c("Loc_code", "Grid_year", "Hydro_resource")) %>%
  mutate(scenario = paste(house_type,insulation,schedule,
                          Loc_code,Occ_code, SH_code,
                          DW_code, 'C0',Grid_year,Hydro_resource, sep = "-"))

# Join for R WH does not include location attribute in join operation 
household_SH_WH_R <- raw_technologies_processed_SH %>%
  left_join(raw_technologies_processed_WH_R, 
  # Location attribute not required for resistive water heating
            by = c("Grid_year", "Hydro_resource")) %>%
  mutate(scenario = paste(house_type,insulation,schedule,
                          Loc_code,Occ_code, SH_code,
                          DW_code, 'C0',Grid_year,Hydro_resource, sep = "-"))

# Merge non-hydronic scenarios 
non_hydronic_SH_WH_dataframes <- list(household_SH_WH_HP,
                                      household_SH_WH_R)

household_SH_WH <- bind_rows(non_hydronic_SH_WH_dataframes) %>%
  mutate(P_annual_kWh_total = P_annual_kWh_SH + P_annual_kWh_WH,
         Q_annual_kWh_total = Q_annual_kWh_SH + Q_annual_kWh_WH,
         P_avg_peak_kW_total = P_avg_peak_kW_SH + P_avg_peak_kW_WH,
         kgCO2_total = kgCO2_SH + kgCO2_WH,
         Embodied = Embodied_SH + Embodied_WH)

# Merge all
household_dataframes <- list(household_SH_WH,raw_technologies_processed_Hydronic)

household <- bind_rows(household_dataframes) %>%
  mutate(SH_DW_code = paste(SH_code, DW_code, sep = "-"))
# Looking at final household dataset, there are 1728 rows with missing values for columns
# that end _SH or _WH, as these correspond to hydronic scenarios,
# meaning that only totals are reported.

# Please make sure you change to desired directory (latest version was saved in github repo)
household %>% write_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/thermal/households.csv")

# Exploratory plots

household %>% filter(schedule == 'Ideal', 
                     insulation == 'Nom',
                     house_type == 'Mod') %>% 
  ggplot(aes(x=Grid_year,y=kgCO2_total)) +
  geom_boxplot() + 
  facet_wrap(~ interaction(Loc_code,Hydro_resource))

household %>% filter(schedule == 'Ideal', 
                     insulation == 'Nom',
                     house_type == 'Mod') %>% 
  ggplot(aes(x=Grid_year,y=Embodied)) +
  geom_boxplot() + 
  facet_wrap(~ Loc_code)

household %>% filter(schedule == 'Ideal', 
                     insulation == 'Nom',
                     house_type == 'Mod') %>% 
  ggplot(aes(x=Grid_year,y=Embodied)) +
  geom_boxplot() + 
  facet_grid(Loc_code~SH_DW_code) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Aggregate household dataset to work out cumulative emissions/energy over the years
# First we need to address interpolation approach as data is given for 5 year periods

household <- household %>% mutate(year = as.integer(Grid_year))

# Interpolating 'value' for each group (defined by scenario)
household_interp <- household %>%
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_DW_code,Occ_code) %>%
  complete(year = desired_years) %>% # creates rows for all years
  arrange(year) %>%
  # These conditions are necesary to take care of hydronic scenarios
  # which do not specify how much is from DWH and SH
  mutate(kgCO2_SH_total_year = if (sum(!is.na(kgCO2_SH)) >=2) {
    approx(year, kgCO2_SH, xout = year, rule = 2)$y
    } else {
      rep(NA_real_, length(year))
    }) %>%
  mutate(kgCO2_WH_total_year = if (sum(!is.na(kgCO2_WH)) >=2) {
    approx(year, kgCO2_WH, xout = year, rule = 2)$y
  } else {
    rep(NA_real_, length(year))
  }) %>%
  mutate(kgCO2_total_year = approx(year, kgCO2_total, xout = year, rule = 2)$y) %>%
  # Electricity energy demand should be constant, average power may change 
  # with control signal
  mutate(P_annual_kWh_total_year = approx(year, P_annual_kWh_total, xout = year, rule = 2)$y) %>%
  mutate(P_avg_peak_kW_total_year = approx(year, P_avg_peak_kW_total, xout = year, rule = 2)$y) %>% 
  ungroup()

# Once interpolation allowed to get annual data (instead of 5 year periods)
# Aggregation can be obtained for lifetime attributes
household_interp_lifetime_operational <- household_interp %>% 
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_DW_code,Occ_code) %>%
  summarise(kgCO2_SH_life = sum(kgCO2_SH_total_year),
            kgCO2_WH_life = sum(kgCO2_WH_total_year),
            P_annual_kWh_life = sum(P_annual_kWh_total_year),
            kgCO2_op_life = sum(kgCO2_total_year)) %>% ungroup()

# Want to get only embodied emissions for all scenario combinations            
household_embodied <- household %>% 
  select(house_type,insulation,schedule,
         Loc_code,Hydro_resource,
         SH_DW_code,Occ_code,
         Embodied_SH, Embodied_WH, Embodied) %>%
  distinct()

# Merge operational and embodied emissions
household_lifetime <- household_interp_lifetime_operational %>%
  left_join(household_embodied, 
            by = c("house_type","insulation","schedule",
                   "Loc_code","Hydro_resource",
                   "SH_DW_code","Occ_code")) %>%
  mutate(kgCO2_total_life = kgCO2_op_life + Embodied)

# Please make sure you change to desired directory (latest version was saved in github repo)
household_lifetime %>% write_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/thermal/households_lifetime.csv")






################## Exploratory plots

household_lifetime <- household_lifetime %>%                    
  mutate(
    x_num = as.numeric(factor(SH_DW_code))   # 1, 2, 3, …
  )

stacked_household_lifetime <- household_lifetime %>% 
  group_by(SH_DW_code, x_num, insulation, Loc_code) %>%
  summarise(
    Operational = mean(kgCO2_op_life, na.rm = TRUE),
    Embodied    = mean(Embodied,        na.rm = TRUE),
    .groups = "drop"
  ) %>% 
  pivot_longer(
    cols      = c(Operational, Embodied),
    names_to  = "component",
    values_to = "value"
  ) %>% 
  group_by(SH_DW_code, x_num, insulation, Loc_code) %>% 
  mutate(
    ymin = cumsum(lag(value, default = 0)),
    ymax = ymin + value
  ) %>%
  ungroup()

### BOXPLOT
ggplot(household_lifetime, aes(x = as.factor(x_num), y = kgCO2_total_life, group = interaction(SH_DW_code, insulation), fill = insulation)) +
  geom_boxplot(width = 0.45, outlier.shape = NA, position = position_dodge(width = 0.7)) +
  scale_x_discrete(labels = unique(household_lifetime$SH_DW_code), name = "SH_DW_code") +
  labs(
    title = "Distribution of Lifecycle Emissions (Embodied + Operational) for SWH combinations",
    x = 'SWH setup',
    y = "kgCO2e",
    fill = "Insulation"
  ) +
  facet_grid(house_type ~ Loc_code) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(size = 12, face = "bold"),
    axis.title.x = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    axis.text.x = element_text(size = 8, angle = 45, vjust = 0.5, hjust = 1),
    axis.text.y = element_text(size = 8)
  )

##### STACKED AREAS
library(scales)

# Prepare the data
stacked_household_lifetime_plot <- stacked_household_lifetime %>%
  arrange(SH_DW_code, Loc_code, insulation, component) %>%
  group_by(Loc_code) %>%
  mutate(
    # Create numeric x-position for each SH_DW_code within each facet
    x_pos = as.numeric(factor(SH_DW_code, levels = unique(SH_DW_code))),
    # Adjust position based on insulation type
    x_adj = ifelse(insulation == "H1", x_pos - 0.2, x_pos + 0.2)
  ) %>%
  ungroup()

y_max <- max(stacked_household_lifetime_plot$value) * 1.1

# Create the plot with facet_wrap by Loc_code
ggplot(stacked_household_lifetime_plot, aes(x = x_adj, y = value, fill = component)) +
  geom_col(
    position = position_stack(),
    width = 0.35,
    color = "white",
    linewidth = 0.2
  ) +
  # Add insulation type labels
  geom_text(
    data = stacked_household_lifetime_plot %>% distinct(SH_DW_code, Loc_code, insulation, x_pos, x_adj),
    aes(x = x_adj, y = -0.05 * max(stacked_household_lifetime_plot$value), label = insulation),
    vjust = 1,
    size = 3,
    inherit.aes = FALSE
  ) +
  facet_wrap(~ Loc_code) +
  # Set x-axis breaks and labels
  scale_x_continuous(
    breaks = unique(stacked_household_lifetime_plot$x_pos),
    labels = unique(stacked_household_lifetime_plot$SH_DW_code)
  ) +
  scale_fill_manual(
    values = c("Operational" = '#ed6d63', "Embodied" = '#163f57'),
    name = "Component"
  ) +
  scale_y_continuous(
    labels = comma,
    limits = c(-0.1 * y_max, y_max),  # Fixed limits for all facets
    expand = c(0, 0)
  ) +
  labs(
    title = "Mean Lifecycle (Operational + Embodied) emissions for SWH setups",
    x = "SWH setup",
    y = "kG CO2e"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    panel.spacing = unit(1.5, "lines"),
    strip.background = element_rect(fill = "gray90", color = NA)
  )

# Plot annual technology annual series
technology_year %>% filter(year >= 2025) %>% 
  filter(!is.na(SH_code)) %>%
  group_by(year, SH_code, Loc_code, house_type, insulation) %>%
  summarize(
    mean_Operational_year = mean(kgCO2_oper_year, na.rm = TRUE),
    mean_Embodied_year = mean(kgCO2_embod_year, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = starts_with("mean_"),
    names_to = "Metric",
    values_to = "Value"
  ) %>% ggplot(aes(x = year, y = Value, fill = Metric)) +
  geom_col(position = "stack") +
  facet_grid(Loc_code + SH_code ~ house_type + insulation) +
  labs(
    x = "Year",
    y = "kGCO2e",
    fill = "Component"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))