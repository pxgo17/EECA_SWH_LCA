library(tidyverse)
library(patchwork)
library(ggpattern)
library(visdat)

unique_values <- function(column){
  column %>% as.factor() %>% levels()
}

# Read csv file with results for SWH technologies
# Files can be read from file (commented line or from github)
# If reading from file, please make sure to change directories accordingly
raw_technologies <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/thermal/technologies_finalv2.csv")
#raw_technologies <- read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/processed/thermal/technologies.csv")
# Files can be read from file (commented line or from github)
embodied <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/embodied_all.csv")
#embodied <- read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/processed/lca/embodied.csv")

# Filters by SH_code
embodied_SH_nonHYD <- embodied %>% filter(SH_code %in% c('R','Rb','Rl','HP','HPl')) %>%
  # New column for schedule is needed for non-hydronic SH subset
  mutate(schedule = word(Hshld_code, 3, sep = fixed("-")))
embodied_SH_HYD <- embodied %>% filter(SH_code %in% c('HYD200','HYD500','HYD1000')) 
# Filter by DW_code
embodied_DW <- embodied %>% filter(DW_code %in% c('R','Ro','HP','HPo'))

# Processing starts from now, first generating new columns and selecting control signal
raw_technologies_processed <- raw_technologies %>%
  separate(col = Hshld_code, # Column to split
           into = c("house_type", "insulation", "schedule"), # New column names
           sep = "-") %>% # Separator character
  separate(col = Grid_code, # Column to split
           into = c("Grid_year", "Hydro_resource"), # New column names
           sep = "-") 

# Define current control signal

CS <- "Crenew"

# Add DW embodied emissions; join only based on DW_code and tank volume
# as these (capacity technology combinations) will be later added to all house types
# Note that domestic water and hydronic cases have control signal dependency
raw_technologies_processed_CS_DW <- raw_technologies_processed %>%
  # Filter control signal - csv's will be generated for each control signal case
  filter(CS_code == CS) %>%
  # Note only three columns are selected before join
  left_join(embodied_DW %>% select(DW_code,`Tank_Volume (L)`,Manufacture,Distribution,Installation,EOL,Other), 
            by = c("DW_code", "Tank_Volume (L)"))

# Add non hydronic (SH) embodied emissions; join includes multiple attributes
# as capacities change depending on house type, insulation, location
raw_technologies_processed_CS_DW_SH <- raw_technologies_processed_CS_DW %>%
  # HP_cap in included as original technologies dataset was missing this information for ASHP
  # R_cap is introduced here (missing in technologies dataset)
  left_join(embodied_SH_nonHYD %>% select(house_type,insulation,Loc_code,schedule,SH_code,HP_cap,R_cap,
                                          Manufacture,Distribution,Installation,EOL,Other),
            by=c("house_type","insulation","Loc_code","schedule","SH_code"), suffix = c("_left", "_right")) %>%
  # Use coalesce(embodied.y, embodied.x) to prefer the right table’s value when there’s a match, 
  # otherwise keep the left table’s value (left table is technologies dataset which only
  # includes capacities for hydronic).
  # Note that column name HP_cap was changed (modified from Daniel's version) 
  mutate(HP_cap = coalesce(HP_cap_right, HP_cap_left),
         Manufacture = coalesce(Manufacture_right, Manufacture_left),
         Distribution = coalesce(Distribution_right, Distribution_left),
         Installation = coalesce(Installation_right, Installation_left),
         EOL = coalesce(EOL_right, EOL_left),
         Other = coalesce(Other_right, Other_left)) %>%
  select(-HP_cap_right, -HP_cap_left, -Manufacture_right,-Manufacture_left,
         -Distribution_right, -Distribution_left, -Installation_right,
         -Installation_left, -EOL_right, -EOL_left, -Other_right,
         -Other_left) %>% 
  # Next line is a fix, as some HP_cap have been defined for resistive WH tech
  mutate(R_cap = ifelse(DW_code %in% c('R','Ro'), HP_cap, R_cap),
         HP_cap = ifelse(DW_code %in% c('R','Ro'), NA, HP_cap))
  
# Add hydronic SH embodied emissions; similar to previous join,
# but in this case HP_cap is included as was previously defined in technologies
# dataset. New columns (embodied HP, FC, tank) will be introduced with this join
# to disaggregate components for Hydronic scenarios. A Hyd_Cap_kW_Adj
# is also introduced, which documents nominal capacity used from
# embodied emissions table.
# Also, note that tank volume was provided in original technologies 
# dataset (consistent with SH_code and DW_code)

raw_technologies_processed_CS_DW_SH_HYD <- raw_technologies_processed_CS_DW_SH %>%
  left_join(embodied_SH_HYD %>% select(house_type,insulation,Loc_code,SH_code,HP_cap,
                                       Manufacture,Distribution,Installation,EOL,Other,
                                       Hyd_Cap_kW_Adj, Hyd_Manufacture_HP,
                                       Hyd_Manufacture_Tank, Hyd_Manufacture_FC),
            by=c("house_type","insulation","Loc_code","SH_code","HP_cap"), suffix = c("_left", "_right")) %>%
  # Use coalesce to prefer the right table’s value when there’s a match, 
  # otherwise keep the left table’s value.
  mutate(Manufacture = coalesce(Manufacture_right, Manufacture_left),
         Distribution = coalesce(Distribution_right, Distribution_left),
         Installation = coalesce(Installation_right, Installation_left),
         EOL = coalesce(EOL_right, EOL_left),
         Other = coalesce(Other_right, Other_left)) %>%
  select(-Manufacture_right,-Manufacture_left,
         -Distribution_right, -Distribution_left, -Installation_right,
         -Installation_left, -EOL_right, -EOL_left, -Other_right,
         -Other_left)

##### On a technology basis, interpolation will be applied to work out
##### cumulative energy demand and emissions (a similar approach is later
##### implemented on a household basis)

# Set up the years you want to interpolate for
# Same period applies for household interpolation
desired_years <- 2025:2045

# Interpolating 'value' for each group (defined by scenario)
technology_year <- raw_technologies_processed_CS_DW_SH_HYD %>% filter(Grid_year>2024) %>%
  mutate(year = as.integer(Grid_year)) %>%
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_code,DW_code,Occ_code,
           R_cap, HP_cap, `Tank_Volume (L)`) %>%
  complete(year = desired_years) %>% # creates rows for all years
  arrange(year) %>%
  mutate(kgCO2_oper_year = approx(year, kgCO2, xout = year, rule = 2)$y) %>%
  # Note that Embodied are divided over period of analysis (21 years)
  # to work out annual share
  mutate(kgCO2_manuf_year = approx(year, Manufacture/21, xout = year, rule = 2)$y) %>%
  mutate(kgCO2_dist_year = approx(year, Distribution/21, xout = year, rule = 2)$y) %>%
  mutate(kgCO2_inst_year = approx(year, Installation/21, xout = year, rule = 2)$y) %>%
  mutate(kgCO2_eol_year = approx(year, EOL/21, xout = year, rule = 2)$y) %>%
  mutate(kgCO2_other_year = approx(year, Other/21, xout = year, rule = 2)$y) %>%
  # Electricity energy demand should be constant, average power may change 
  # with control signal
  mutate(P_annual_kWh_year = approx(year, `P_annual (kWh)`, xout = year, rule = 2)$y) %>%
  mutate(P_avg_peak_kW_year = approx(year, `P_avg_peak (kW)`, xout = year, rule = 2)$y) %>%
  ungroup() 

# Please make sure you change to desired directory (latest version was saved in github repo)
technology_year %>% write_csv(paste0("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/technology_year_",CS,".csv"))

### PLOT POWER
# technology_year %>% 
#   select(house_type, insulation, schedule, Loc_code, Occ_code, R_cap, HP_cap, `Tank_Volume (L)`, P_avg_peak_kW_year, year, DW_code,kgCO2_oper_year) %>%
#   filter(!is.na(DW_code)) %>%
#   filter(year > 2024) %>%
#   #filter(Loc_code == 'CH') %>%
#   #filter(DW_code %in% c('HYD200','HYD500','HYD1000')) %>%
#   filter(DW_code %in% c("HP","R")) %>%
#   ggplot(aes(x = as.factor(year), y = kgCO2_oper_year, fill = DW_code)) +
#   geom_bar(position = "dodge", stat = "summary", fun = "mean") +
#   facet_grid(Occ_code ~ `Tank_Volume (L)`) +
#   labs(x = "Year", fill = "Tech Type") +
#   theme_minimal()

# Calculate operational and embodied emissions over lifespan
technology_lifetime <- technology_year %>% 
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_code,DW_code,Occ_code,
           R_cap, HP_cap, `Tank_Volume (L)`) %>%
  summarise(kgCO2_oper_life = sum(kgCO2_oper_year),
            P_annual_kWh_life = sum(P_annual_kWh_year),
            kgCO2_manuf_life = sum(kgCO2_manuf_year),
            kgCO2_dist_life = sum(kgCO2_dist_year),
            kgCO2_inst_life = sum(kgCO2_inst_year),
            kgCO2_eol_life = sum(kgCO2_eol_year),
            kgCO2_other_life = sum(kgCO2_other_year)
            ) %>% ungroup() %>%
  mutate(kgCO2_total_life = kgCO2_oper_life + kgCO2_manuf_life +
           kgCO2_dist_life + kgCO2_inst_life + kgCO2_eol_life + kgCO2_other_life)
  

# Please make sure you change to desired directory (latest version was saved in github repo)
technology_lifetime %>% write_csv(paste0("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/technology_lifetime_",CS,".csv"))

####### COMBINATIONS FOR  development of household dataset  
# Need to run 'C0' first
# HP and R Space heating cases are not affected by control signal (do not include thermal storage for SH)

if (CS=="C0"){
  raw_technologies_processed_C0_DW_SH_HYD <- raw_technologies_processed_CS_DW_SH_HYD
  # Embodied emisssion in 'raw_technologies' datasets correspond to full lifetime
  # Need to get annual figures, hence division by 2021 for period 2025-2045
  raw_technologies_processed_C0_DW_SH_HYD <- raw_technologies_processed_C0_DW_SH_HYD %>%
    mutate(Manufacture = Manufacture/21,
           Distribution = Distribution/21,
           Installation = Installation/21,
           EOL= EOL/21,
           Other = Other/21)
}

raw_technologies_processed_CS_DW_SH_HYD <- raw_technologies_processed_CS_DW_SH_HYD %>%
  mutate(Manufacture = Manufacture/21,
         Distribution = Distribution/21,
         Installation = Installation/21,
         EOL= EOL/21,
         Other = Other/21)

# First SH combination for all resistive SH_code ('R')
# No grouping required as data is already aggregated
raw_technologies_processed_SH_R <- raw_technologies_processed_C0_DW_SH_HYD %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code == 'R') %>% # Second filter only resistive (R)
  select(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource, SH_code, HP_cap, R_cap,
         `P_annual (kWh)`, `Q_annual (kWh)`, kgCO2, `P_avg_peak (kW)`,
         Manufacture,Distribution,Installation,EOL,Other) %>%
  rename(
    P_annual_kWh_SH = `P_annual (kWh)`,
    Q_annual_kWh_SH = `Q_annual (kWh)`,
    P_avg_peak_kW_SH = `P_avg_peak (kW)`,
    Operation_SH =  kgCO2,
    Manufacture_SH = Manufacture,
    Distribution_SH = Distribution,
    Installation_SH = Installation,
    EOL_SH = EOL,
    Other_SH = Other)

# Second SH combination for all heat pump
# No grouping required as data is already aggregated
raw_technologies_processed_SH_HP <- raw_technologies_processed_C0_DW_SH_HYD %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code == 'HP') %>% # Second filter only heat pump (HP)
  select(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource, SH_code, HP_cap, R_cap,
         `P_annual (kWh)`, `Q_annual (kWh)`, kgCO2, `P_avg_peak (kW)`,
         Manufacture,Distribution,Installation,EOL,Other) %>%
  rename(
    P_annual_kWh_SH = `P_annual (kWh)`,
    Q_annual_kWh_SH = `Q_annual (kWh)`,
    P_avg_peak_kW_SH = `P_avg_peak (kW)`,
    Operation_SH =  kgCO2,
    Manufacture_SH = Manufacture,
    Distribution_SH = Distribution,
    Installation_SH = Installation,
    EOL_SH = EOL,
    Other_SH = Other)

# Third SH combination for Living heat pump, Bedroom Resistive
# Grouping required (hence SH_code not included in group_by and added at end)
raw_technologies_processed_SH_HP_R <- raw_technologies_processed_C0_DW_SH_HYD %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code %in% c('HPl','Rb')) %>% # Second filter 
  group_by(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource) %>%
  summarise(P_annual_kWh_SH = sum(`P_annual (kWh)`),
            Q_annual_kWh_SH = sum(`Q_annual (kWh)`),
            Operation_SH = sum(kgCO2),
            P_avg_peak_kW_SH = sum(`P_avg_peak (kW)`),
  # embodied table includes SH capacity for both resistive and HP,
  # meaning that in all cases there will two values, one for HP_cap,
  # and one for R_cap, and one will be NA.  
            HP_cap = sum(HP_cap, na.rm = TRUE), 
            R_cap = sum(R_cap, na.rm = TRUE),
            Manufacture_SH = sum(Manufacture),
            Distribution_SH = sum(Distribution),
            Installation_SH = sum(Installation),
            EOL_SH = sum(EOL),
            Other_SH = sum(Other)) %>% 
  ungroup() %>%
  mutate(SH_code = 'HPl_Rb')

# Fourth combination is for hydronic (SH and WH),
# In this case occupancy matters because includes WH
# No grouping needed, hydronic heats entire house space and water
# Extra columns selected (occupancy and HP capacity)
# Note name of base dataset includes CS (depends on control signal)
raw_technologies_processed_Hydronic <- raw_technologies_processed_CS_DW_SH_HYD %>%
  filter(!is.na(SH_code)) %>% # First filter only space heating (SH)
  filter(SH_code %in% c("HYD1000", "HYD200",  "HYD500")) %>% # Second filter only heat pump (HP)
  select(house_type,insulation,schedule,Loc_code, Grid_year, Hydro_resource,
         SH_code, HP_cap, DW_code, `Tank_Volume (L)`, Occ_code, 
         `P_annual (kWh)`, `Q_annual (kWh)`, kgCO2, `P_avg_peak (kW)`, 
         Manufacture,Distribution,Installation,EOL,Other, CS_code) %>%
  rename(
    P_annual_kWh_total = `P_annual (kWh)`,
    Q_annual_kWh_total = `Q_annual (kWh)`,
    P_avg_peak_kW_total = `P_avg_peak (kW)`,
    Operation = kgCO2) %>%
  mutate(scenario = paste(house_type,insulation,schedule,
                          Loc_code,Occ_code, SH_code,
                          DW_code, CS_code,Grid_year,Hydro_resource, sep = "-"))

# Fifth combination is for HP WH,
# In this case occupancy matters because includes WH
raw_technologies_processed_WH_HP <- raw_technologies_processed_CS_DW_SH_HYD %>%
  filter(!is.na(DW_code)) %>% # First filter only water heating (DW)
  filter(DW_code %in% c("HP", "HPo")) %>% # Second filter to leave only HP
  # Embodied is included in grouping variables to keep in final result
  group_by(Loc_code, Grid_year, Hydro_resource, Occ_code, DW_code, `Tank_Volume (L)`, CS_code) %>% # Note location included as heat pump performace relies on weather
  summarise(P_annual_kWh_WH = sum(`P_annual (kWh)`),
            Q_annual_kWh_WH = sum(`Q_annual (kWh)`),
            Operation_WH = sum(kgCO2),
            P_avg_peak_kW_WH = sum(`P_avg_peak (kW)`),
            Manufacture_WH = sum(Manufacture),
            Distribution_WH = sum(Distribution),
            Installation_WH = sum(Installation),
            EOL_WH = sum(EOL),
            Other_WH = sum(Other)) %>% 
  ungroup()

test_WH_HP <- raw_technologies_processed_CS_DW_SH_HYD %>%
  filter(!is.na(DW_code)) %>% # First filter only water heating (DW)
  filter(DW_code %in% c("HP", "HPo")) 

test_WH_HP_group <- raw_technologies_processed_CS_DW_SH_HYD %>%
  filter(!is.na(DW_code)) %>% # First filter only water heating (DW)
  filter(DW_code %in% c("HP", "HPo")) %>%
  group_by(Loc_code, Grid_year, Hydro_resource, Occ_code, DW_code, `Tank_Volume (L)`, CS_code) %>% # Note location included as heat pump performace relies on weather
  summarise(P_annual_kWh_WH = sum(`P_annual (kWh)`),
            Q_annual_kWh_WH = sum(`Q_annual (kWh)`),
            Operation_WH = sum(kgCO2),
            P_avg_peak_kW_WH = sum(`P_avg_peak (kW)`),
            Manufacture_WH = sum(Manufacture),
            Distribution_WH = sum(Distribution),
            Installation_WH = sum(Installation),
            EOL_WH = sum(EOL),
            Other_WH = sum(Other)) %>% 
  ungroup()

# Sixth combination is for R WH,
# In this case occupancy matters because includes WH
# Location code note required for resistive DWH
raw_technologies_processed_WH_R <- raw_technologies_processed_CS_DW_SH_HYD %>%
  filter(!is.na(DW_code)) %>% # First filter only water heating (DW)
  filter(DW_code %in% c("R", "Ro")) %>% # Second filter to leave only HP
  group_by(Grid_year, Hydro_resource, Occ_code, DW_code, `Tank_Volume (L)`, CS_code) %>% # Note location not included
  summarise(P_annual_kWh_WH = sum(`P_annual (kWh)`),
            Q_annual_kWh_WH = sum(`Q_annual (kWh)`),
            Operation_WH = sum(kgCO2),
            P_avg_peak_kW_WH = sum(`P_avg_peak (kW)`),
            Manufacture_WH = sum(Manufacture),
            Distribution_WH = sum(Distribution),
            Installation_WH = sum(Installation),
            EOL_WH = sum(EOL),
            Other_WH = sum(Other)) %>% 
  ungroup()

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
                          DW_code, CS_code,Grid_year,Hydro_resource, sep = "-"))
# Note that there will be missing values for HP_cap or R_cap, depending on SH_code 
# (i.e. all resistive will not have HP capacity)

# Join for R WH does not include location attribute in join operation 
household_SH_WH_R <- raw_technologies_processed_SH %>%
  left_join(raw_technologies_processed_WH_R, 
  # Location attribute not required for resistive water heating
            by = c("Grid_year", "Hydro_resource")) %>%
  mutate(scenario = paste(house_type,insulation,schedule,
                          Loc_code,Occ_code, SH_code,
                          DW_code, CS_code,Grid_year,Hydro_resource, sep = "-"))

# Merge non-hydronic scenarios 
non_hydronic_SH_WH_dataframes <- list(household_SH_WH_HP,
                                      household_SH_WH_R)

household_SH_WH <- bind_rows(non_hydronic_SH_WH_dataframes) %>%
  mutate(P_annual_kWh_total = P_annual_kWh_SH + P_annual_kWh_WH,
         Q_annual_kWh_total = Q_annual_kWh_SH + Q_annual_kWh_WH,
         P_avg_peak_kW_total = P_avg_peak_kW_SH + P_avg_peak_kW_WH,
         Operation = Operation_SH + Operation_WH,
         Manufacture = Manufacture_SH + Manufacture_WH,
         Distribution = Distribution_SH + Distribution_WH,
         Installation = Installation_SH + Installation_WH,
         EOL = EOL_SH + EOL_WH,
         Other = Other_SH + Other_WH)

# Merge all
household_dataframes <- list(household_SH_WH,raw_technologies_processed_Hydronic)

household <- bind_rows(household_dataframes) %>%
  mutate(SH_DW_code = paste(SH_code, DW_code, sep = "-"))
# Looking at final household dataset, there are 1728 rows with missing values for columns
# that end _SH or _WH, as these correspond to hydronic scenarios,
# meaning that only totals are reported.

# Aggregate household dataset to work out cumulative emissions/energy over the years
# First we need to address interpolation approach as data is given for 5 year periods

household <- household %>% mutate(year = as.integer(Grid_year))

# Note that rows for hydronic do not have SH / WH split 

# Interpolating 'value' for each group (defined by scenario minus year)
household_interp <- household %>% filter(year > 2024) %>%
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_DW_code,Occ_code,CS_code) %>%
  complete(year = desired_years) %>% # creates rows for all years
  arrange(year) %>%
  # These conditions are necessary to take care of hydronic scenarios
  # which do not specify how much is from DWH and SH
  mutate(Operation_SH_total_year = if (sum(!is.na(Operation_SH)) >=2) {
    approx(year, Operation_SH, xout = year, rule = 2)$y
    } else {
      rep(NA_real_, length(year))
    }) %>%
  mutate(Operation_WH_total_year = if (sum(!is.na(Operation_WH)) >=2) {
    approx(year, Operation_WH, xout = year, rule = 2)$y
  } else {
    rep(NA_real_, length(year))
  }) %>%
  mutate(Operation_total_year = if (sum(!is.na(Operation)) >=2) {
    approx(year, Operation, xout = year, rule = 2)$y
  } else {
    rep(NA_real_, length(year))
  }) %>%
  # Electricity energy demand should be constant, average power may change 
  # with control signal
  mutate(P_annual_kWh_total_year = approx(year, P_annual_kWh_total, xout = year, rule = 2)$y) %>%
  mutate(P_avg_peak_kW_total_year = approx(year, P_avg_peak_kW_total, xout = year, rule = 2)$y) %>%
  ungroup()

# Please make sure you change to desired directory (latest version was saved in github repo)
household_interp %>% write_csv(paste0("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/household_year_",CS,".csv"))


# Once interpolation allowed to get annual data (instead of 5 year periods)
# Aggregation can be obtained for lifetime operation attributes
household_operation <- household_interp %>% 
  group_by(house_type,insulation,schedule,Loc_code,Hydro_resource,SH_DW_code,Occ_code) %>%
  summarise(Operation_SH_life = sum(Operation_SH_total_year),
            Operation_WH_life = sum(Operation_WH_total_year),
            P_annual_kWh_life = sum(P_annual_kWh_total_year),
            Operation_life = sum(Operation_total_year),
            P_avg_peak_kW_year = mean(P_avg_peak_kW_total_year)) %>% ungroup() 

# Want to get only embodied emissions for all scenario combinations
# For building lifetime dataset, embodied need to be multiplied to 21
household_embodied <- household %>%
  select(house_type,insulation,
         Loc_code,
         SH_DW_code,Occ_code,
         Manufacture_SH, Manufacture_WH, Manufacture,
         Distribution_SH, Distribution_WH, Distribution,
         Installation_SH, Installation_WH, Installation,
         EOL_SH, EOL_WH, EOL,
         Other_SH, Other_WH, Other) %>%
  mutate(Manufacture_SH = Manufacture_SH*21,
         Distribution_SH = Distribution_SH*21,
         Installation_SH = Installation_SH*21,
         EOL_SH= EOL_SH*21,
         Other_SH = Other_SH*21,
         Manufacture_WH = Manufacture_WH*21,
         Distribution_WH = Distribution_WH*21,
         Installation_WH = Installation_WH*21,
         EOL_WH= EOL_WH*21,
         Other_WH = Other_WH*21,
         Manufacture = Manufacture*21,
         Distribution = Distribution*21,
         Installation = Installation*21,
         EOL= EOL*21,
         Other = Other*21) %>%
  distinct()

# Merge operational and embodied emissions
household_lifetime <- household_operation %>%
  left_join(household_embodied,
            by = c("house_type","insulation",
                   "Loc_code",
                   "SH_DW_code","Occ_code")) %>%
  mutate(Embodied_life = Manufacture + Distribution + Installation + EOL + Other,
         Total_life = Operation_life + Embodied_life)

# Please make sure you change to desired directory (latest version was saved in github repo)
household_lifetime %>% write_csv(paste0("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/household_lifetime_",CS,".csv"))

####################### RUN ALL LINES BEFORE THIS CHECKPOINT FIRST FOR EACH CONTROL SIGNAL###############################

# Merge datasets with different control signals
household_lifetime_C0 <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/household_lifetime_C0.csv") %>%
  mutate(CS_code = 'C0')
household_lifetime_Cpeak <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/household_lifetime_Cpeak.csv") %>%
  mutate(CS_code = 'Cpeak')
household_lifetime_Crenew <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/household_lifetime_Crenew.csv") %>%
  mutate(CS_code = 'Crenew')

# Merge all household_lifetime datasets
household_lifetime_dataframes <- list(household_lifetime_C0,household_lifetime_Cpeak,household_lifetime_Crenew)
household_lifetime_Call <- bind_rows(household_lifetime_dataframes) 

household_lifetime_Call %>% write_csv(paste0("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/household_lifetime_Call.csv"))

# Merge datasets for technology lifetime with different control signals
technology_lifetime_C0 <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/technology_lifetime_C0.csv") %>%
  mutate(CS_code = 'C0')
technology_lifetime_Cpeak <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/technology_lifetime_Cpeak.csv") %>%
  mutate(CS_code = 'Cpeak')
technology_lifetime_Crenew <- read_csv("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/technology_lifetime_Crenew.csv") %>%
  mutate(CS_code = 'Crenew')

# Merge all household_lifetime datasets
technology_lifetime_dataframes <- list(technology_lifetime_C0,technology_lifetime_Cpeak,technology_lifetime_Crenew)
technology_lifetime_Call <- bind_rows(technology_lifetime_dataframes) 
technology_lifetime_Call %>% write_csv(paste0("D:/EECA_SWH_LCA/EECA_SWH_LCA/EECA_SWH_LCA/data/processed/lca/technology_lifetime_Call.csv"))

######################### RUN UP TO HERE AFTER CONTROL SIGNAL SPECIFIC FILES ARE CREATED #################################

test <- raw_technologies %>% rowwise() %>%
  mutate(year=as.integer(strsplit(Grid_code,"-")[[1]][1])) %>% 
  mutate(hydro=strsplit(Grid_code,"-")[[1]][2]) %>% 
  mutate(house_type=strsplit(Hshld_code,"-")[[1]][1]) %>% 
  mutate(insulation=strsplit(Hshld_code,"-")[[1]][2]) %>%
  mutate(schedule=strsplit(Hshld_code,"-")[[1]][3]) %>%
  filter(SH_code %in% c("HYD200", "HYD500", "HYD1000")) %>% 
  group_by(house_type,insulation,schedule,Loc_code,SH_code,HP_cap, year, hydro, Occ_code) %>% 
  summarise(kWh_year = mean(`P_annual (kWh)`), kgCO2 = mean(kgCO2)) %>% ungroup()

test_cap <- test %>% filter(SH_code == "HYD1000", 
                             year ==2024, 
                             house_type == 'Mass',
                             Loc_code == 'AK')
test %>% 
  filter(hydro == 'Hhigh') %>% 
  filter(year == 2024) %>%
  filter(insulation == 'Nom') %>%
  filter(schedule == 'Real') %>%
  ggplot(aes(x=SH_code,y=kgCO2)) +
  geom_col() + 
  facet_grid(Loc_code + house_type ~ HP_cap ) +
  theme_minimal()


