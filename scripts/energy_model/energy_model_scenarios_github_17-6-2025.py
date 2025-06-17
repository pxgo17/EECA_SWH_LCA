# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:43:34 2024

@author: pxg11
"""
import pandas as pd
import numpy as np
import joblib
import gdown

# # Visualize historic dispatch distribution
eeca_palette = ['#41b496', '#447474', '#163f57', '#3c4c49', '#e94e24', '#ed6d63']

############## Load base years data
base_2022 = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/base_year_2022.csv")
base_2024 = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/base_year_2024.csv")
# Load base demand
base_demand = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/demand_base.csv")
# check annual output (EDGS: 40.575 TWh in 2024). Actual = 41.423
# print(base_demand['kWh'].sum())
# Load solar base (not accounted for in scenario data)
base_solar_distributed = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/solar_distributed.csv")
# Load estimated power output for utility size solar farms (Kohira, Edgecumbe, Lauriston)
base_solar_k = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/solar_kohira_pvlib.csv")
base_solar_r = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/solar_rangitaiki_pvlib.csv")
base_solar_l = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/solar_lauriston_pvlib.csv")
base_solar = pd.DataFrame()
base_solar['utility_kWh'] = base_solar_k['kWh'] + base_solar_r['kWh'] + base_solar_l['kWh']
base_solar['distributed_kWh'] = base_solar_distributed['kWh']
base_solar['total_kWh'] = base_solar['utility_kWh'] + base_solar['distributed_kWh']
# print(base_solar['distributed_kWh'].sum()/base_solar['utility_kWh'].sum())
# Load wind base
base_wind = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/wind_base.csv")
# Load base cogeneration
# cogen_gas updated to new rolling window series 'cogen_gas_base_rw.csv', profile fixed
# for all scenarios as no added capacity is expected
base_cogen_gas = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/cogen_gas_base_rw.csv")

# Wood capacity increase is expected, but approach based on capacity factor which
# remains fixed
cogen_wood_CF = 0.7378
# Geothermal capacity increase is expected, but approach based on capacity factor which
# remains fixed
geo_CF = 0.8113
# Capacity for coal and diesel based on transpower data https://www.transpower.co.nz/system-operator/live-system-and-market-data/generation-fuel-type
# No additional capacity increase from coal in the future, decommissioning expected
# Regimes are based on distribution of 4 year historical hourly series.
low_kWh_coal = 250000
mid_kWh_coal = 500000
max_kWh_coal = 750000
# Frequency keeping regime is based on 2.5th percentile
freq_kWh_coal = 34566
# Capacity ranges for CCGT gas power dispatch regimes
# Two sets are defined to account for Taranaki decomission before 2030
mid_kWh_ccgt = 394000
max_kWh_ccgt = 788000
mid_kWh_ccgt_new = 200000
max_kWh_ccgt_new = 415275

# Max dispatch for diesel power generation, remains fixed
max_kWh_diesel = 156000
# Define transmission and distribution losses ref: https://figure.nz/chart/IYj2FDmVXFCOJxSA
# Total 6.9% reference: https://www.mbie.govt.nz/assets/energy-in-new-zealand-2021.pdf
# Just distribution losses are applied to solar
t_losses = 0.069*(1406.08 / (1406.08 + 1546.62))
d_losses = 0.069*(1546.62 / (1406.08 + 1546.62))
td_losses = t_losses + d_losses

############## Load Machine learning models and scaler (for hydro regressor)
# Load trained hydro dispatch model and scaler
# hydro_model = joblib.load("C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//hydro_dispatch.pkl")
# hydro_scaler = joblib.load("C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//hydro_dispatch_scaler.pkl")

hydro_id = "114EXsk0lJRt0leANMdKWRlKjqanR6KHO"
url = f"https://drive.google.com/uc?export=download&id={hydro_id}"
output = "hydro_dispatch.pkl"
gdown.download(url, output, quiet=False)
hydro_model = joblib.load(open(output, "rb"))

scaler_id = "1mUbNorXR7IobSi-jFbMDEgCMEKkJCKfI"
url = f"https://drive.google.com/uc?export=download&id={scaler_id}"
output = "hydro_dispatch_scaler.pkl"
gdown.download(url, output, quiet=False)
hydro_scaler = joblib.load(open(output, "rb"))

# Approach has changed. New model predicts dispatch regime for each thermal fossil baseload CCGT and Coal.
# Load trained coal power dispatch model (classifier, no scaling needed)
# coal_model = joblib.load('C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//coal_dispatch.pkl')

coal_id = "1fr7o6EWzIFimhQs2IQt3VYyZCZ-GocDC"
url = f"https://drive.google.com/uc?export=download&id={coal_id}"
output = "coal_dispatch.pkl"
gdown.download(url, output, quiet=False)
coal_model = joblib.load(open(output, "rb"))

# Load trained gas CCGT power dispatch model (classifier, no scaling needed)
# ccgt_model = joblib.load('C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//ccgt_dispatch.pkl')

ccgt_id = "1JlsXfMe072h5h5BaIGWvkRShwB0ozdQf"
url = f"https://drive.google.com/uc?export=download&id={ccgt_id}"
output = "ccgt_dispatch.pkl"
gdown.download(url, output, quiet=False)
coal_model = joblib.load(open(output, "rb"))

############## Load hydro resource (does not change in all scenarios) 
resource_hydro_low = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/hydro_bad.csv")
resource_hydro_high = pd.read_csv("https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/hydro_good.csv")
# Calculate min and max storage capacity
# hydro_min_storage_kWh = 1000000*min(resource_hydro_low['storage_GWh'].min(), resource_hydro_high['storage_GWh'].min())
hydro_min_storage_kWh = 700000000
# Based on maximum controlled storage reported in June/2023
hydro_max_storage_kWh = 5000000000.0
#hydro_max_storage_kWh = 1000000*max(resource_hydro['storage_GWh'].max(), resource_hydro_high['storage_GWh'].max())
# Create an additional reference storage point (midmax) to shift hydro to dispatch
# to higher outputs and aim to reduce spill. This can be considered as a factor
# to moderate hydro dispatch 
midmax = 0.7
# Function to select which hydro resource to implement, low or high
def select_hydro_resource(hydro_resource):
    if hydro_resource=='high':
        return resource_hydro_high
    else:
        return resource_hydro_low

############## Load cumulative scenario data for periods
solar_utility_scenarios = pd.read_csv('https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/solar_scenario_results_cumulative.csv', index_col='hour_of_year')
wind_utility_scenarios = pd.read_csv('https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/wind_scenario_results_cumulative.csv', index_col='hour_of_year')
solar_distributed_scenarios = pd.read_csv('https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/solar_distributed_scenario_results.csv', index_col='hour_of_year')
demand_scenarios = pd.read_csv('https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/demand_scenario_results.csv', index_col='hour_of_year')
# Capacities for gas (peaker and ocgt) already account for decommisioning
# Note negative value for gas (considers net outcome for all periods)
capacity_scenarios = pd.read_csv('https://raw.githubusercontent.com/pxgo17/EECA_SWH_LCA/refs/heads/main/data/raw/cumulative_capacity_scenarios_periods_adj.csv')
# Reindex to keep compatibility with scenario data
solar_utility_scenarios.index = solar_utility_scenarios.index - 1
wind_utility_scenarios.index = wind_utility_scenarios.index - 1
solar_distributed_scenarios.index = solar_distributed_scenarios.index - 1

def energy_model(scenario, period, hydro_resource):    
    # Declare global variables that will be updated within function
    global low_kWh_coal, mid_kWh_coal, max_kWh_coal
    # Define hydro resource values
    resource_hydro = select_hydro_resource(hydro_resource)
    hydro_midmax_storage_kWh = 1000000*resource_hydro['storage_GWh'].quantile(midmax)
    # Define cumulative capacities (MW) for a specific scenario year combination
    cum_hydro = capacity_scenarios.loc[(capacity_scenarios['period'] == period) & (capacity_scenarios['scenario'] == scenario), 'hydro'].values[0]
    cum_geo = capacity_scenarios.loc[(capacity_scenarios['period'] == period) & (capacity_scenarios['scenario'] == scenario), 'geo'].values[0]
    cum_gas = capacity_scenarios.loc[(capacity_scenarios['period'] == period) & (capacity_scenarios['scenario'] == scenario), 'gas'].values[0]
    cum_wood = capacity_scenarios.loc[(capacity_scenarios['period'] == period) & (capacity_scenarios['scenario'] == scenario), 'wood'].values[0]
    # Update capacities (base + future cumulative)
    # Hydro Max base capacity updated from 4700 MW to transpower figures 5415
    hydro_max_dispatch_kWh = 1000*(5415 + cum_hydro) 
    cogen_wood_cap = 1000*(37.55 + cum_wood) 
    # Gas base capacity based on EA spreadsheet (DispatchedGenerationPlant.xlsb)
    max_kWh_ocgt_gas = 1000*(492 + cum_gas)    
    # Maximum base geo capacity based on EMI figure documented in report
    geo_cap = 1000*(1061 + cum_geo)
    # Create a DataFrame for cogen wood with the same number of rows as base_cogen_gas
    cogen_wood = pd.DataFrame({
        "cogen_wood_kWh": [cogen_wood_CF*cogen_wood_cap] * len(base_cogen_gas)})
    # Calculate total cogeneration, and calculate share of each technology
    # Merge the two DataFrames
    cogen = pd.concat([base_cogen_gas, cogen_wood], axis=1)
    # Create a new column representing the sum of the other two columns
    cogen['total_kWh'] = cogen['cogen_gas_kWh'] + cogen['cogen_wood_kWh']
    # Create columns for the total share in regards to the column total
    cogen['cogen_gas_share'] = cogen['cogen_gas_kWh'] / cogen['total_kWh']
    cogen['cogen_wood_share'] = cogen['cogen_wood_kWh'] / cogen['total_kWh']
    
    # Define column name to select series for specific scenario period combination
    scen_period = f"{scenario}_{period}"
    
    # Define dataframe to record estimates
    energy_results = pd.DataFrame()
    energy_results = pd.DataFrame({
        # Demand divided by number less than 1
        # to overcome transmission and distribution losses
        'demand': demand_scenarios[scen_period]/(1-td_losses),
        'geothermal_dispatched': 0,
        'geothermal_excess': 0,
        'wind_dispatched': 0,    
        'unmet_demand': 0
    })
       
    ##### Geothermal dispatch: Capacity factor approach
    # Given Maximum capacity and CF of 81.13%
    geothermal_available = geo_cap*geo_CF
    energy_results['geothermal_available'] = geothermal_available
    energy_results['geothermal_dispatched'] = energy_results[['demand', 'geothermal_available']].min(axis=1)
    energy_results['geothermal_excess'] = geothermal_available - energy_results['geothermal_dispatched']
    # Calculate unmet demand after geothermal dispatch
    energy_results['unmet_demand'] = energy_results['demand'] - energy_results['geothermal_dispatched']
    # Ensures that the column values are bounded by the specified lower limit (0).
    energy_results['unmet_demand'] = energy_results['unmet_demand'].clip(lower=0)
           
    ##### Wind dispatch: Availability based on windpowerlib estimates
    wind_available = base_wind['wind_kWh'] + wind_utility_scenarios[scen_period]
    energy_results['wind_available'] = wind_available
    # Vectorized-based approach to set conditional (check unmet demand) power dispatch
    energy_results['wind_dispatched'] = np.where(
        energy_results['unmet_demand'] > 0,  
        energy_results[['unmet_demand', 'wind_available']].min(axis=1),  # If True: calculate the minimum
        0)    
    energy_results['wind_excess'] = wind_available - energy_results['wind_dispatched']
    # Calculate remaining unmet demand after wind dispatch
    energy_results['unmet_demand'] -= energy_results['wind_dispatched']
    
    ##### Solar dispatch grid connected: Availability based on pvlib estimates for utility setup
    # Solar available corrects kohira output to account for transmission losses
    solar_grid_available = base_solar_k['kWh'] 
    energy_results['solar_grid_available'] = solar_grid_available
    # Vectorized-based approach to set conditional (check unmet demand) power dispatch
    energy_results['solar_grid_dispatched'] = np.where(
        energy_results['unmet_demand'] > 0,  
        energy_results[['unmet_demand', 'solar_grid_available']].min(axis=1),  # If True: calculate the minimum
        0)
    energy_results['solar_grid_net'] = (1 - td_losses)*energy_results['solar_grid_dispatched']
    energy_results['solar_grid_excess'] = solar_grid_available - energy_results['solar_grid_dispatched']
    # Calculate remaining unmet demand after solar grid dispatch
    energy_results['unmet_demand'] -= energy_results['solar_grid_dispatched']
    
    ##### Solar dispatch embedded: Utility setup
    solar_utility_available = (base_solar_r['kWh'] + base_solar_l['kWh']  
                                + solar_utility_scenarios[scen_period])
    energy_results['solar_utility_available'] = solar_utility_available
    # Unmet demand is corrected here as generation does not need to compensate
    # transmission losses
    energy_results['factored_unmet_demand_u'] = (1-t_losses)*energy_results['unmet_demand']
    # Vectorized-based approach to set conditional (check unmet demand) power dispatch
    energy_results['solar_utility_dispatched'] = np.where(
        energy_results['unmet_demand'] > 0,  
        energy_results[['factored_unmet_demand_u', 'solar_utility_available']].min(axis=1),  # If True: calculate the minimum
        0)
    energy_results['solar_utility_excess'] = solar_utility_available - energy_results['solar_utility_dispatched']
    # Calculate remaining unmet demand after solar embedded dispatch (correct for compensation of transmission losses)
    energy_results['unmet_demand'] -= (1 + t_losses)*energy_results['solar_utility_dispatched']
    
    # Need to start setting variables needed for classifier implementation 
    energy_results['utility_interm'] = (energy_results['wind_dispatched']
                                        + energy_results['solar_utility_dispatched'])
    
    ##### Solar dispatch embedded: Rooftop setup
    solar_rooftop_available = (base_solar['distributed_kWh'] 
                                + solar_distributed_scenarios[scen_period])
    energy_results['solar_rooftop_available'] = solar_rooftop_available
    # Unmet demand is corrected here as generation does not need to compensate
    # transmission AND distribution losses (power is consumed at site)
    energy_results['factored_unmet_demand_r'] = (1-td_losses)*energy_results['unmet_demand']
    # Vectorized-based approach to set conditional (check unmet demand) power dispatch
    energy_results['solar_rooftop_dispatched'] = np.where(
        energy_results['unmet_demand'] > 0,  
        energy_results[['factored_unmet_demand_r', 'solar_rooftop_available']].min(axis=1),  # If True: calculate the minimum
        0)
    energy_results['solar_rooftop_excess'] = solar_rooftop_available - energy_results['solar_rooftop_dispatched']
    # Calculate remaining unmet demand after solar embedded dispatch (correct for compensation of transmission AND distribution losses)
    energy_results['unmet_demand'] -= (1 + td_losses)*energy_results['solar_rooftop_dispatched']
    
    ##### Coal dispatch two stages
    # At this ppoint we calculate rolling window sums for variables that are 
    # needed to predict the regime (max capacity) for coal power generation
    resource_hydro['inflow_next_8'] = resource_hydro['inflow_MW'].shift(-7).rolling(window=8, min_periods=1).sum()
    energy_results['inflow_next_8'] = resource_hydro['inflow_next_8']
    energy_results['demand_next_8'] = energy_results['demand'].shift(-7).rolling(window=8, min_periods=1).sum()
    energy_results['geo_next_8'] = energy_results['geothermal_dispatched'].shift(-7).rolling(window=8, min_periods=1).sum()
       
    ##### Variables to support error tracing during model execution
    energy_results['hyd_pred_>_hyd_max'] = 0
    energy_results['hyd_pred_>_unmet'] = 0
    energy_results['hyd_post_stor_>_hyd_min_stor'] = 0
    energy_results['hyd_max_stor_>_hyd_post_stor_>_hyd_min_stor'] = 0
    energy_results['hyd_post_stor_>_hyd_max_stor'] = 0
    energy_results['hyd_max_disp_<_unmet'] = 0
    energy_results['hyd_pred_<_unmet'] = 0
    energy_results['hyd_post_stor_upd_>_hyd_max_stor'] = 0
    energy_results['hyd_post_stor_upd_<_hyd_max_stor'] = 0
    energy_results['unmet_>_0'] = 0
    energy_results['hydro_condition'] = 'default'
    energy_results['only_renewable'] = 0
    energy_results['hydro_contingency'] = 0
    energy_results['coal_regime'] = 0
    
    # Iterative hydro power dispatch loop over all hours
    for i in range(len(base_demand)):
        if i == 4380:
            print('50% scenario progress')
        # demand in this block does not account for losses
        # as models calibrated on supply side totals
        demand = demand_scenarios.loc[i, scen_period]
        # temporal attributes included in base demand dataframe        
        inflows = resource_hydro.loc[i, "inflow_MW"]
        unmet_demand = energy_results.loc[i,'unmet_demand']
        cogen_i = cogen.loc[i,'total_kWh']
        # Definition of storage. First value is based on historical levels
        # If i > 0 storage values are defined in previous iteration         
        if i==0:
            hydro_storage = 1000000*resource_hydro.loc[i, "storage_GWh"]
            # Assumed no coal generation took place at last hour of previous year
            lag_coal = 0.0
            lag_ccgt = 0.0
            energy_results.loc[i, "hydro_storage"] = hydro_storage
            energy_results.loc[i,'i_=_0_init_hydstor'] = 1
        else:
            lag_coal = energy_results.loc[i-1, "coal_dispatched"]
            lag_ccgt = energy_results.loc[i-1, "gas_ccgt_dispatched"]
            hydro_storage = energy_results.loc[i, "hydro_storage"]
            energy_results.loc[i,'i_>_0_set_hydstor'] = 1
        
        # Need to check if there is unmet demand, and fill values for 
        # processed to be skipped
        if unmet_demand == 0:
            energy_results.loc[i, "only_renewable"] = 1
            energy_results.loc[i, "hydro_dispatched"] = 0
            # Condition to avoid assigning value to non-existent row
            if i < 8759:
                energy_results.loc[i+1, "hydro_storage"] = hydro_storage + (1000*resource_hydro.loc[i + 1, "inflow_MW"])
            # Condition to ensure max storage is not surpassed
            if energy_results.loc[i, "hydro_storage"] >= hydro_max_storage_kWh:
                energy_results.loc[i, "hydro_spill"] = energy_results.loc[i, "hydro_storage"] - hydro_max_storage_kWh
                energy_results.loc[i, "hydro_storage"] = hydro_max_storage_kWh
                energy_results.loc[i,'hydro_condition'] = 'max_hydro'
            elif energy_results.loc[i, "hydro_storage"] >= hydro_midmax_storage_kWh:
                energy_results.loc[i,'hydro_condition'] = 'max_hydro'                
            elif energy_results.loc[i, "hydro_storage"] <= hydro_min_storage_kWh:
                energy_results.loc[i,'hydro_condition'] = 'low_hydro'
            elif energy_results.loc[i, "hydro_storage"] > hydro_min_storage_kWh and energy_results.loc[i, "hydro_storage"] < hydro_midmax_storage_kWh:
                energy_results.loc[i,'hydro_condition'] = 'mid_hydro'        
            # No further dispatch needed if above unmet_demand condition is TRUE
            energy_results.loc[i,'gas_ccgt_dispatched'] = 0 
            energy_results.loc[i,'coal_dispatched'] = 0 
            energy_results.loc[i, "cogen_gas"] = 0 
            energy_results.loc[i, "cogen_wood"] = 0
            energy_results.loc[i,'gas_ocgt_dispatched'] = 0            
            energy_results.loc[i,'diesel_dispatched'] = 0 
            continue
                       
        ##### Hydro Dispatch: ML Model 
        # Hydro dispatch based only on utility intermittent
        intermittent_utility = wind_available[i] + solar_grid_available[i] + solar_utility_available[i]
        # Prepare hydro model input
        model_input = pd.DataFrame([[demand - intermittent_utility, hydro_storage/1000000, inflows]], 
                                   columns=['diff', 'storage_GWh', 'inflow_MWh'])
        model_input_scaled = hydro_scaler.transform(model_input)
        # Predict hydro dispatch
        hydro_dispatch_pred = hydro_model.predict(model_input_scaled)[0]
        # Unlike other sections in the model where skipping iterations is determined by unmet
        # demand, this part has to implement other actions to manage hydro storage levels    
        # Condition to ensure dispatch is not higher than historical capacity
        if hydro_dispatch_pred > hydro_max_dispatch_kWh:
            hydro_dispatch_pred = hydro_max_dispatch_kWh
            energy_results.loc[i,'hyd_pred_>_hyd_max'] = 1        
        # Condition to ensure dispatch does not exceed unmet_demand
        if hydro_dispatch_pred > unmet_demand:
            hydro_dispatch_pred = unmet_demand      
            energy_results.loc[i,'hyd_pred_>_unmet'] = 1
        # calculate storage after estimated hydro power dispatch
        hydro_storage_post = hydro_storage - hydro_dispatch_pred
        # Check if 'post storage' goes below minimum required
        if hydro_storage_post <= hydro_min_storage_kWh:
            energy_results.loc[i,'hyd_post_stor_<_hyd_min_stor'] = 1
            energy_results.loc[i,'hydro_condition'] = 'low_hydro'
            # adjust dispatch to avoid dispatching more than allowed by minimum storage
            hydro_dispatch_adj = hydro_storage - hydro_min_storage_kWh
            energy_results.loc[i, "hydro_dispatched"] = hydro_dispatch_adj
            energy_results.loc[i, "hydro_storage"] = hydro_min_storage_kWh
            energy_results.loc[i, "hydro_spill"] = 0
            # condition to avoid error on last iteration
            if i < 8759:
                # new storage should be based on current plus historical inflows
                # from next row as current historical inflows are already included in storage.
                # a factor of 1000 is used for conversion from MW to kW
                new_storage = energy_results.loc[i, "hydro_storage"] + (1000*resource_hydro.loc[i + 1, "inflow_MW"])
                energy_results.loc[i+1, "hydro_storage"] = new_storage       
        # Check is dispatch is within min and max thresholds.
        elif hydro_storage_post > hydro_min_storage_kWh and hydro_storage_post < hydro_midmax_storage_kWh:
            energy_results.loc[i,'hyd_max_stor_>_hyd_post_stor_>_hyd_min_stor'] = 1
            energy_results.loc[i,'hydro_condition'] = 'mid_hydro'
            # all predicted hydro power can be dispatched
            energy_results.loc[i, "hydro_dispatched"] = hydro_dispatch_pred
            # hydro_storage_post already accounts for reduction due to predicted dispatch
            energy_results.loc[i, "hydro_storage"] = hydro_storage_post
            energy_results.loc[i, "hydro_spill"] = 0
            # condition to avoid error on last iteration
            if i < 8759:            
                # new storage should be based on current plus historical inflows
                # from next row as current historical inflows are already included in storage.
                # a factor of 1000 is used for conversion from MW to kW
                new_storage = energy_results.loc[i, "hydro_storage"] + (1000*resource_hydro.loc[i + 1, "inflow_MW"])
                energy_results.loc[i+1, "hydro_storage"] = new_storage     
        # Check if max hydro storage is exceeded
        # This block forces maximum allowed dispatch
        elif hydro_storage_post >= hydro_midmax_storage_kWh:
            energy_results.loc[i,'hyd_post_stor_>_hyd_max_stor'] = 1
            energy_results.loc[i,'hydro_condition'] = 'max_hydro'
            # Need to dispatch everything as lake levels are at maximum 
            # despite already implemented dispatch
            # Ensure max dispatch does not exceed demand requirements  
            hydro_dispatch_pred = min(hydro_max_dispatch_kWh, unmet_demand)
            # if (hydro_max_dispatch_kWh <= unmet_demand) and (hydro_dispatch_pred <= hydro_max_dispatch_kWh):
            #     energy_results.loc[i,'hyd_max_disp_<_unmet'] = 1
            #     hydro_dispatch_pred = hydro_max_dispatch_kWh
            # elif (hydro_dispatch_pred < unmet_demand):
            #     energy_results.loc[i,'hyd_pred_<_unmet'] = 1
            #     hydro_dispatch_pred = unmet_demand
            # hydro storage needs to be updated again as hydro dispatched was updated
            hydro_storage_post = hydro_storage - hydro_dispatch_pred
            energy_results.loc[i, "hydro_dispatched"] = hydro_dispatch_pred
            # Conditions to calculate storage based on updated dispatch to avoid
            # negative spill
            if hydro_storage_post > hydro_max_storage_kWh:
                energy_results.loc[i,'hyd_post_stor_upd_>_hyd_max_stor'] = 1
                # Maximum storage is set at maximum and everything else is spilled
                energy_results.loc[i, "hydro_storage"] = hydro_max_storage_kWh
                # Spill includes inflows included in previous iteration             
                energy_results.loc[i, "hydro_spill"] = hydro_storage_post - hydro_max_storage_kWh
            elif hydro_storage_post <= hydro_max_storage_kWh:
                energy_results.loc[i,'hyd_post_stor_upd_<_hyd_max_stor'] = 1
                # Storage is updated and No spill as storage does not exceed limit   
                energy_results.loc[i, "hydro_storage"] = hydro_storage_post          
                energy_results.loc[i, "hydro_spill"] = 0         
            # condition to avoid error on last iteration
            if i < 8759:    
                new_storage = energy_results.loc[i, "hydro_storage"] + (1000*resource_hydro.loc[i + 1, "inflow_MW"])
                energy_results.loc[i+1, "hydro_storage"] = new_storage
        unmet_demand -= energy_results.loc[i, "hydro_dispatched"]
        
        ##### CCGT dispatch: Highest merit order fossil thermal, ML predicts regime
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        if unmet_demand > 0 and period < 2040:
            ccgt_model_input = pd.DataFrame([[energy_results.loc[i,'geothermal_dispatched'],
                                              energy_results.loc[i,'utility_interm'],
                                              energy_results.loc[i,'hydro_storage']/1000000, # note that original model uses GWh instead of  kWh
                                              resource_hydro.loc[i,'inflow_MW'],
                                              energy_results.loc[i,'demand'],
                                              lag_ccgt,
                                              energy_results.loc[i,'demand_next_8'],
                                              energy_results.loc[i,'geo_next_8'],
                                              energy_results.loc[i,'inflow_next_8']
                                              ]],
                                            columns=['Geo_Geo', 'interm', 'storage_GWh', 'inflow_MW', 'demand', 
                                                      'lag_ccgt', 'demand_next_8','geo_next_8', 'inflow_next_8'])
            ccgt_regime = ccgt_model.predict(ccgt_model_input)[0]
            energy_results.loc[i, "ccgt_regime"] = ccgt_regime
            if ccgt_regime == 0:
                ccgt_gas_dispatched = 0
            elif ccgt_regime == 1:
                # if regime is one, we are in ramp up mode. Prediction
                # also accounts for lag generation, so previous power dispatch
                # at mid regime increase chances of having higher capacity in 
                # future dispatch
                if(period >= 2030):
                    ccgt_gas_dispatched = min(mid_kWh_ccgt, unmet_demand) 
                else:
                    ccgt_gas_dispatched = min(mid_kWh_ccgt_new, unmet_demand)                    
            else:
                # if regime is two, maximum capacity enabled  
                if(period >= 2030):
                    ccgt_gas_dispatched = min(max_kWh_ccgt, unmet_demand)
                else:
                    ccgt_gas_dispatched = min(max_kWh_ccgt_new, unmet_demand)
            energy_results.loc[i,'gas_ccgt_dispatched'] = ccgt_gas_dispatched
            unmet_demand -= ccgt_gas_dispatched
        # In this case condition checks for period as CCGT is fully removed by 2040
        # no 'continue' at this point because demand remains unmet
        elif unmet_demand > 0 and period >= 2040:
            energy_results.loc[i,'gas_ccgt_dispatched'] = 0 
            energy_results.loc[i,'coal_dispatched'] = 0
            energy_results.loc[i, "cogen_gas"] = 0 
            energy_results.loc[i, "cogen_wood"] = 0 
            energy_results.loc[i,'gas_ocgt_dispatched'] = 0              
            energy_results.loc[i,'diesel_dispatched'] = 0
        elif unmet_demand <= 0:
            energy_results.loc[i,'gas_ccgt_dispatched'] = 0 
            energy_results.loc[i,'coal_dispatched'] = 0
            energy_results.loc[i, "cogen_gas"] = 0 
            energy_results.loc[i, "cogen_wood"] = 0 
            energy_results.loc[i,'gas_ocgt_dispatched'] = 0              
            energy_results.loc[i,'diesel_dispatched'] = 0 
            continue
        
        ##### Coal dispatch stage 1: ML Model 
        # Stage 1 is implemented here as frequency keeping is needed 
        # when demand is high and hydro resource is low (inflows and storage); 
        # there is still a chance frequency keeping is not needed
        # when dispatch regime is zero (there are 4 dispatch regimes 0 - 3).
        # Frequency keeping can be applicable for all periods
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        if unmet_demand > 0:
            coal_model_input = pd.DataFrame([[energy_results.loc[i,'geothermal_dispatched'],
                                              energy_results.loc[i,'utility_interm'],
                                              energy_results.loc[i,'hydro_storage']/1000000, # note that original model uses GWh instead of  kWh
                                              resource_hydro.loc[i,'inflow_MW'],
                                              energy_results.loc[i,'demand'],
                                              lag_coal,
                                              energy_results.loc[i,'demand_next_8'],
                                              energy_results.loc[i,'geo_next_8'],
                                              energy_results.loc[i,'inflow_next_8']
                                             ]],
                                            columns=['Geo_Geo', 'interm', 'storage_GWh', 'inflow_MW', 'demand', 
                                                      'lag_coal', 'demand_next_8','geo_next_8', 'inflow_next_8'])
            coal_regime = coal_model.predict(coal_model_input)[0]
            energy_results.loc[i, "coal_regime"] = coal_regime
            if coal_regime == 0:
                coal_dispatched = 0
            else:
                # if other regimes are identified/required, coal generation must provide
                # minimum between frequency keeping and unmet_demand
                # provide frequency keeping. If more power is required, stage 2 is
                # implemented later; this bypasses full coal generation to enable
                # lower emission cogeneration dispatch
                coal_dispatched = min(freq_kWh_coal, unmet_demand)    
            energy_results.loc[i,'coal_dispatched'] = coal_dispatched
            energy_results.loc[i,'coal_regime'] = coal_regime
            unmet_demand -= coal_dispatched
        else:
            energy_results.loc[i,'coal_dispatched'] = 0
            energy_results.loc[i, "cogen_gas"] = 0 
            energy_results.loc[i, "cogen_wood"] = 0 
            energy_results.loc[i,'gas_ocgt_dispatched'] = 0              
            energy_results.loc[i,'diesel_dispatched'] = 0 
            energy_results.loc[i,'coal_regime'] = 0
            continue
        
        ###### Cogeneration Dispatch: Availability shape (gas) and Capacity factor (wood) 
        energy_results.loc[i, "unmet_demand"] = unmet_demand    
        # Demand remains unmet? Then cogen dispatch is next
        if unmet_demand > 0:
            energy_results.loc[i,'unmet_>_0'] = 1
            # cogen_i is sum of wood and gas based cogen at i; defined earlier
            if cogen_i <= unmet_demand:
                cogen_gas = cogen.loc[i,'cogen_gas_kWh']
                cogen_wood = cogen.loc[i,'cogen_wood_kWh']          
            # Condition when cogen is greater than demand, dispatch is 
            # determined in proportion to historical share
            else:
                # Proportionately dispatch the value of unmet_demand
                cogen_gas = unmet_demand * cogen.loc[i,'cogen_gas_share']
                cogen_wood = unmet_demand * cogen.loc[i,'cogen_wood_share'] 
            energy_results.loc[i, "cogen_gas"] = cogen_gas
            energy_results.loc[i, "cogen_wood"] = cogen_wood
            unmet_demand -= (cogen_wood + cogen_gas)            
        else:            
            energy_results.loc[i, "cogen_gas"] = 0 
            energy_results.loc[i, "cogen_wood"] = 0 
            energy_results.loc[i,'gas_ocgt_dispatched'] = 0              
            energy_results.loc[i,'diesel_dispatched'] = 0 
            continue
        
        ##### OCGT Dispatch 
        # It is assumed that OCGT plants can be activated on a 
        # demand basis without ramp up
        # max_kWh_ocgt_gas already accounts for projected capacity
        # additional and decommissioning (capacity_scenarios table)
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        if unmet_demand > 0:
            ocgt_dispatched = min(unmet_demand, max_kWh_ocgt_gas)
            energy_results.loc[i, "gas_ocgt_dispatched"] = ocgt_dispatched 
            unmet_demand -= ocgt_dispatched            
        else:             
            energy_results.loc[i,'gas_ocgt_dispatched'] = 0              
            energy_results.loc[i,'diesel_dispatched'] = 0
            continue
        
        ##### Coal dispatch stage 2
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        # Frequency keeping coal dispatch may have already happened,
        # so at this stage we top up with remaining capacity which depends
        # on predicted regime
        if unmet_demand > 0 and coal_regime != 0:
            # An additional condition at this point to reduce capacity
            # due to (cumulative) decommissioning of 400 MW by 2035
            if period >= 2035:
                low_kWh_coal = 115000
                mid_kWh_coal = 230000
                max_kWh_coal = 350000
            # Need to dispatch depending on coal regimes 0, 1, 2, or 3
            # coal_dispatched_2 is the additional power.
            # if regime is 1,2, or 3, it means frequency keeping
            # has already been dispatched (hence substraction from given capacity)
            if coal_regime == 1:
                coal_dispatched_2 = min(unmet_demand, low_kWh_coal - coal_dispatched)
            elif coal_regime == 2:
                coal_dispatched_2 = min(unmet_demand, mid_kWh_coal - coal_dispatched)
            elif coal_regime == 3:
                coal_dispatched_2 = min(unmet_demand, max_kWh_coal - coal_dispatched)
            # Update previous coal generation (set at coal_dispatched)
            energy_results.loc[i,'coal_dispatched'] += coal_dispatched_2
            unmet_demand -= coal_dispatched_2
        elif unmet_demand > 0 and coal_regime == 0:                          
            energy_results.loc[i,'diesel_dispatched'] = 0 
        elif unmet_demand <= 0:                          
            energy_results.loc[i,'diesel_dispatched'] = 0
            continue   
        
        ##### Diesel power Dispatch ---
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        # In this case we also check period as coal was expected to be decomissioned
        # in 2024 but likely before 2030
        if unmet_demand > 0 and period < 2030:
            # min function ensures dispatch does not exceed capacity
            # only transmission as diesel is likely consumed in distribution networks
            diesel_dispatch = min((1-t_losses)*unmet_demand, max_kWh_diesel)
        else:
            diesel_dispatch = 0
        energy_results.loc[i,'diesel_dispatched'] = diesel_dispatch
        # Diesel is also considered as embedded, hence
        # transmission losses are compensated to unmet demand
        unmet_demand -= (1+t_losses)*diesel_dispatch
        ##### Hydro contingency check
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        if unmet_demand > 0 and energy_results.loc[i,'hydro_condition'] == 'mid_hydro':
            energy_results.loc[i,'hydro_contingency'] = 1
            # dispatch considers difference of already dispatched and max dispatch (critical csase)
            contingency_hydro = min(hydro_max_dispatch_kWh - energy_results.loc[i, "hydro_dispatched"], unmet_demand)
            # Extra dispatch is added to already dispatched
            energy_results.loc[i, "hydro_dispatched"] += contingency_hydro 
            # Further reduction of storage, spillage was already set to zero (mid_hydro condition fulfilled by now)
            energy_results.loc[i, "hydro_storage"] -= contingency_hydro 
            unmet_demand -= contingency_hydro
            # condition to update storage and avoid error on last iteration
            if i < 8759:            
                # new storage should be based on current plus historical inflows
                # from next row as current historical inflows are already included in storage.
                # a factor of 1000 is used for conversion from MW to kW
                new_storage = energy_results.loc[i, "hydro_storage"] + (1000*resource_hydro.loc[i + 1, "inflow_MW"])
                energy_results.loc[i+1, "hydro_storage"] = new_storage
        ##### Coal contingency check
        energy_results.loc[i, "unmet_demand"] = unmet_demand
        # Condition checks that coal plant is already running (>0) and that
        # it has not dispatched at full capacity yet (3)
        if unmet_demand > 0 and energy_results.loc[i, 'coal_regime'] not in (0, 3):
            # New regime is added for emergency
            energy_results.loc[i,'coal_regime'] = 4
            # dispatch considers difference of already dispatched and max dispatch (critical case)
            contingency_coal = min(max_kWh_coal - energy_results.loc[i, "coal_dispatched"], unmet_demand)
            # Extra dispatch is added to already dispatched
            energy_results.loc[i, "coal_dispatched"] += contingency_coal 
            # Further reduction of storage, spillage was already set to zero (mid_hydro condition fulfilled by now)
            unmet_demand -= contingency_coal
        energy_results.loc[i, "unmet_demand"] = unmet_demand
    return energy_results

## Model can be executed for a specific scenario
# scenario_results = energy_model('Base', 2024, 'low')

#### Setup to test base year; recommend to run first
# scenarios = ['Base']
# periods = [2024]
# hydro_resource = ['low', 'high']

###### Define scenarios and periods for energy model
scenarios = ['Base', 'Reference', 'Environmental']  # Add 'Base'
periods = [2024, 2025, 2030, 2035, 2040, 2045]      # Include 2024
hydro_resource = ['low', 'high']


##### Run all scenarios
# Initialize the dictionary to store dataframes
scenario_results = {}
# Iterate over all combinations of arguments
for scenario in scenarios:
    for period in periods:
        # Skip periods > 2024 for 'Base' scenario
        if scenario == 'Base' and period != 2024:
            continue
        # Skip 2024 for 'Reference' and 'Environmental' scenarios
        if scenario in ['Reference', 'Environmental'] and period == 2024:
            continue
        for hydro in hydro_resource:
            # Call the energy_model function with the current combination of arguments
            scenario_period = energy_model(scenario, period, hydro)
            # Use the combination of arguments as the key and store the resulting DataFrame
            scenario_results[(scenario, period, hydro)] = scenario_period    
            print(scenario,period,hydro)

##### Save scenario data into csv's
# First set of results based on midmax parameter set to 70% (see hydro dispatch)
# Change folder name if running model for different midmax setup
# results_path =  'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont_ccgt'
# # Ensure the folder exists
# os.makedirs(results_path, exist_ok=True)    
# # Iterate through the dictionary
# for key, df in scenario_results.items():
#     # Convert the key (tuple) to a string to use as the filename
#     sanitized_key = '_'.join(map(str, key))
#     file_path = os.path.join(results_path, f"{sanitized_key}.csv")
#     # Save the DataFrame as a CSV file
#     df.to_csv(file_path, index=False)    
