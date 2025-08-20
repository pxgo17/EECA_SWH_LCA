# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:36:55 2025

@author: pxg11
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set path to folder with latest results
results_path =  'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont_ccgt'
# Ensure the folder exists
os.makedirs(results_path, exist_ok=True)  

# To read back dictionary for further analysis in this script
# This avoids re-running energy model (time consuming)

def load_csv_to_dict(folder_path):    
    # Initialize an empty dictionary to store the DataFrames
    loaded_dict = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Only process CSV files
        if filename.endswith('.csv'):
            # Remove the file extension and split the filename into components to reconstruct the key
            sanitized_key = os.path.splitext(filename)[0]
            key = tuple(sanitized_key.split('_'))

            # Construct the full file path
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Add the DataFrame to the dictionary with the reconstructed key
            loaded_dict[key] = df

    return loaded_dict

scenario_results = load_csv_to_dict(results_path)

# Aggregate solar generation in single column for further results processing
for key, df in scenario_results.items():       
    df['solar_dispatched'] = df['solar_grid_dispatched'] + df['solar_utility_dispatched'] + df['solar_rooftop_dispatched']

##### Plot scenarios
# Function to generate load duration curve plots. Note that depends on existence
# of scenario_results dataframe    
def load_duration_plot(scenario, period, hydro):      
    # Create a plot for each DataFrame
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    scenario_df = scenario_results[(scenario, period, hydro)]
    #scenario_df['solar_dispatched'] = scenario_df['solar_grid_dispatched'] + scenario_df['solar_utility_dispatched'] + scenario_df['solar_rooftop_dispatched']
    scenario_df_sorted = scenario_df.sort_values(by='demand', ascending=False).reset_index(drop=True)
    # Create a stacked area plot representing load duration curve
    plt.figure(figsize=(10, 6))
    plt.stackplot(range(len(scenario_df_sorted)), 
                  scenario_df_sorted['geothermal_dispatched']/1000,
                  scenario_df_sorted['hydro_dispatched']/1000,
                  scenario_df_sorted['gas_ccgt_dispatched']/1000,
                  scenario_df_sorted['coal_dispatched']/1000,                  
                  scenario_df_sorted['wind_dispatched']/1000,
                  scenario_df_sorted['solar_dispatched']/1000,
                  scenario_df_sorted['cogen_wood']/1000,
                  scenario_df_sorted['cogen_gas']/1000,              
                  scenario_df_sorted['gas_ocgt_dispatched']/1000,                 
                  scenario_df_sorted['diesel_dispatched']/1000,
                  labels=['Geothermal', 'Hydro', 'Gas_CCGT', 'Coal', 'Wind', 'Solar', 'Cogen Wood', 'Cogen Gas', 'Gas_OCGT', 'Diesel'],
                  colors=['#41b496','#163f57', '#447474', '#3c4c49', '#92c5de', 'yellow', '#f4a582', '#e94e24', '#447474', 'brown'])
    # colors=['#41b496', '#447474', '#163f57', '#3c4c49', '#e94e24', '#ed6d63', '#f4a582', '#92c5de']
    # Add labels and title
    plt.xlabel('Time (sorted by demand)')
    plt.ylabel('Power Dispatch (MW)')
    # Process data from key to include in plot and file names
    # key_clean = '_'.join(map(str, key))
    plt.title(f'Load Duration Curve with Power Dispatch by Technology - {scenario},{period},{hydro}')
    plt.legend(loc='upper right')
    # Show the plot
    plt.savefig(f'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont_ccgt//Plots//load_duration_curve_modeled_{scenario}_{period}_{hydro}.png')
    # Optionally, close the plot to free up memory
    plt.close()
    print(f"Plot saved for {scenario},{period},{hydro}")

# If scenario_results dictionary was created by reading from file 
# then keys (period) need to be processed (char to int)

scenario_results = {
    (key[0], int(key[1]), key[2]): value
    for key, value in scenario_results.items()
}

# Build plots iteratively

for key, df in scenario_results.items():
    # Unpack the key into scenario, period, and hydro
    scenario, period, hydro = key    
    # Pass the arguments to the load_duration_plot function
    load_duration_plot(scenario, period, hydro)


##### Analysis of results

# Function to calculate sums for each DataFrame in the scenario_results dictionary
def calculate_sums(df_dict, columns_to_sum):
    results = []

    for key, df in df_dict.items():
        scenario, year, resource = key  # Unpack the key
        row = {'Scenario': scenario, 'Year': year, 'Resource': resource}
        
        # Calculate sums for the specified columns
        for column in columns_to_sum:
            row[column] = df[column].sum()
        
        results.append(row)

    # Convert the results into a DataFrame
    return pd.DataFrame(results)

# columns to sum
columns_sum = ['demand','unmet_demand', 'wind_excess', 'solar_grid_excess', 
               'solar_utility_excess', 'solar_rooftop_excess','wind_dispatched', 
               'solar_grid_dispatched', 'solar_utility_dispatched', 
               'solar_rooftop_dispatched', 'hydro_spill', 'hydro_dispatched']

# Calculate sums
scenario_aggregate_results = calculate_sums(scenario_results, columns_sum)
scenario_aggregate_results['%_demand_deficit'] = 100*scenario_aggregate_results['unmet_demand'] / scenario_aggregate_results['demand']
scenario_aggregate_results['%_excess_interm_utility'] = 100*(scenario_aggregate_results['wind_excess'] + \
                                                     scenario_aggregate_results['solar_grid_excess'] + \
                                                         scenario_aggregate_results['solar_utility_excess']) / (
                                                             scenario_aggregate_results['wind_dispatched'] + scenario_aggregate_results['solar_grid_dispatched'] + \
                                                                 scenario_aggregate_results['solar_utility_dispatched'])
scenario_aggregate_results['%_excess_rooftop'] = 100*(scenario_aggregate_results['solar_rooftop_excess']) / \
                                                     scenario_aggregate_results['solar_rooftop_dispatched']   
scenario_aggregate_results['%_hydro_spill'] = 100*(scenario_aggregate_results['hydro_spill']) / \
                                                     scenario_aggregate_results['hydro_dispatched']     

# Plot aggregate results
                                                         
def save_aggregate_plots(dataframe, columns_to_plot, output_dir):
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get unique resources
    resources = dataframe['Resource'].unique()
    
    # Set your desired font sizes
    axis_title_fontsize = 16
    tick_label_fontsize = 13
    legend_fontsize = 13
    title_fontsize = 18

    for column in columns_to_plot:
        for resource in resources:
            # Filter data for the specific resource
            resource_data = dataframe[dataframe['Resource'] == resource]

            # Pivot data to prepare for plotting (Years as index, Scenarios as columns)
            pivot_df = resource_data.pivot(index='Year', columns='Scenario', values=column)

            # Create a plot for this resource and column
            fig, ax = plt.subplots(figsize=(8, 6))
            pivot_df.plot(kind='bar', ax=ax, width=0.8)
            
            ax.set_title(f'{resource} - {column}', fontsize=title_fontsize)
            ax.set_xlabel('Year', fontsize=axis_title_fontsize)
            ax.set_ylabel(column, fontsize=axis_title_fontsize)
            ax.legend(title='Scenario', fontsize=legend_fontsize, title_fontsize=legend_fontsize)

            # Set tick label font size
            ax.tick_params(axis='both', labelsize=tick_label_fontsize)

            # Adjust layout
            plt.tight_layout()

            # Save plot with a file name that includes the column and resource
            output_file = f"{output_dir}/{column}_{resource}_comparison.png"
            plt.savefig(output_file)
            print(f"Plot saved: {output_file}")

            # Close the figure to avoid memory issues
            plt.close(fig)                                    


# midmax70_env_2045_low = energy_model('Environmental', 2045, 'low')
# midmax70_env_2045_high = energy_model('Environmental', 2045, 'high')

# midmax70_ref_2025_low = energy_model('Reference', 2025, 'low')
# midmax70_ref_2025_high = energy_model('Reference', 2025, 'high')

# # Visualize predicted dispatch distribution for each hydro_condition
# # Function to format the x-axis labels
# def thousands(x, pos):
#     return f'{int(x / 1000)}'
# # Create the plot
# palette_three = {
#     "max_hydro": "#41b496",
#     "mid_hydro": "#447474",
#     "low_hydro": "#e94e24"
# }
# sns.histplot(data=midmax70_ref_2025_high, x="hydro_dispatched", hue="hydro_condition", palette=palette_three, kde=True)
# # Adding labels and title
# plt.xlabel('MWh')
# plt.ylabel('Frequency')
# plt.title('Distribution of Dispatch Regimes')
# # Format the x-axis
# plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
# # Display the plot
# plt.show()
# plt.close()

# # Visualize predicted hydro dispatch distribution
# sns.histplot(scenario_results["hydro_dispatched"], color='coral', kde=True)
# # Adding labels and title
# plt.xlabel('kWh')
# plt.ylabel('Frequency')
# plt.title('Distribution Dispatch Plot')
# # Display the plot
# plt.show()
# plt.close()

# # Visualize predicted dispatch distribution for each hydro_condition
# # Function to format the x-axis labels
# def thousands(x, pos):
#     return f'{int(x / 1000)}'
# # Create the plot
# sns.histplot(data=scenario_results, x="hydro_dispatched", hue="hydro_condition", palette=eeca_palette, kde=True)
# # Adding labels and title
# plt.xlabel('MWh')
# plt.ylabel('Frequency')
# # Format the x-axis
# plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands))
# # Display the plot
# plt.show()
# plt.close()

# # Visualize predicted storage levels
# # Plot annual series
# plt.figure(figsize=(12, 6))
# plt.plot(scenario_results["hydro_storage"])
# plt.title('Hydro Controlled Storage')
# plt.xlabel('Hour')
# plt.ylabel('kWh')
# plt.show()  
# plt.close()    

# # Visualize predicted hydro spill distribution
# sns.histplot(scenario_results["hydro_spill"], kde=True)
# # Adding labels and title
# plt.xlabel('kWh')
# plt.ylabel('Frequency')
# plt.title('Distribution Spillage Plot')
# # Display the plot
# plt.show()
# plt.close()

# scenario_results['storage_quartile'] = pd.qcut(scenario_results['hydro_storage'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
# # Create boxplot to understand relationship of hydro dispatch versus storage
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='storage_quartile', y='hydro_dispatched', data=scenario_results)
# # Adding labels and title
# plt.xlabel('Hydro Storage Quartile (kWh)')
# plt.ylabel('Hydro Electricity Hourly Dispatched (kWh)')
# plt.title('Distribution of Hydro Dispatch for Different Ranges of Hydro Storage')
# # Display the plot
# plt.show()
# plt.close()

# # Save renewable intermittent to file
# # Note that 'solar_dispatched' was created during execution of load duration curve plot
# ren_int = ['solar_dispatched','wind_dispatched']
# energy_inter = scenario_results[ren_int]
# energy_inter.to_csv('C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//predicted_renewable_intermittent.csv')

# Apply emission factors to temporal dispatch
EF = pd.read_csv("C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//input_files//emission_factors.csv")
EF = EF.loc[:,['Tech_Code', 'Fuel_Code','EF_tCO2_per_MWh']]
# Only select columns that have electricity dispatched
gen_cols = ['geothermal_dispatched','wind_dispatched','solar_dispatched',
              'hydro_dispatched', 'gas_ccgt_dispatched', 'gas_ocgt_dispatched',
              'coal_dispatched', 'diesel_dispatched', 'cogen_gas', 
              'cogen_wood']

# Example mapping DataFrame to map generation columns to Tech_Code and Fuel_Code
mapping_gen = {
    'Generation_Column': ['cogen_gas', 'cogen_wood', 'geothermal_dispatched', 'gas_ccgt_dispatched', 'gas_ocgt_dispatched', 'coal_dispatched', 'diesel_dispatched'],
    'Tech_Code': ['Cogen', 'Cogen', 'Geo', 'Thrml', 'Thrml', 'Thrml', 'Thrml'],
    'Fuel_Code': ['Gas', 'Wood', 'Geo', 'Gas_CCGT', 'Gas_OCGT','Coal', 'Diesel']
}
gen_mapping = pd.DataFrame(mapping_gen)

def process_scenario_results(scenario_results, gen_cols, gen_mapping, EF):
    
    transformed_results = {}

    for key, scenario_results_df in scenario_results.items():
        # Make a copy of the DataFrame
        scenario_results_df = scenario_results_df.copy()

        # Select generation columns and fill NaN values with 0
        generation_results = scenario_results_df.loc[:, gen_cols].fillna(0)
        generation_results['Hour'] = generation_results.index

        # Melt the generation results to long format
        df_generation_melted = generation_results.melt(
            id_vars='Hour', var_name='Generation_Column', value_name='kWh'
        )

        # Merge with the mapping DataFrame
        df_generation_with_mapping = df_generation_melted.merge(
            gen_mapping, on='Generation_Column', how='left'
        )

        # Merge with emission factors DataFrame
        df_generation_with_emissions = df_generation_with_mapping.merge(
            EF, on=['Tech_Code', 'Fuel_Code'], how='left'
        )
        df_generation_with_emissions = df_generation_with_emissions.fillna(0)

        # Calculate emissions
        df_generation_with_emissions['Emissions_tCO2'] = (
            (df_generation_with_emissions['kWh'] / 1000) *
            df_generation_with_emissions['EF_tCO2_per_MWh']
        )

        # Create a pivot table with 'Emissions_tCO2' and 'kWh'
        df_emissions = df_generation_with_emissions.pivot_table(
            index='Hour', columns='Generation_Column', values=['Emissions_tCO2', 'kWh']
        ).sort_index()

        # Drop columns with all zeros
        df_emissions = df_emissions.loc[:, (df_emissions != 0).any(axis=0)]

        # Calculate row totals for both 'Emissions_tCO2' and 'kWh'
        df_emissions['Total_Emissions_tCO2'] = df_emissions['Emissions_tCO2'].sum(axis=1)
        df_emissions['Total_kWh'] = df_emissions['kWh'].sum(axis=1)

        # Calculate emission factor (EF) in tCO2 per MWh
        df_emissions['EF_tCO2_per_MWh'] = (
            df_emissions['Total_Emissions_tCO2'] / (df_emissions['Total_kWh'] / 1000)
        )

        # Add the transformed DataFrame to the results dictionary
        transformed_results[key] = df_emissions

    return transformed_results

emission_factor_dict = process_scenario_results(scenario_results, gen_cols, gen_mapping, EF)

# Sample function to add mean EF_tCO2_per_MWh to scenario_aggregate_results
def add_mean_emissions_to_results(emission_factor_dict, scenario_aggregate_results):
    # Initialize a list to store results
    mean_emissions = []

    # Iterate over the keys and DataFrames in the dictionary
    for key, df in emission_factor_dict.items():
        # Calculate the mean of the EF_tCO2_per_MWh column
        mean_ef = df['EF_tCO2_per_MWh'].mean()

        # Append the result as a dictionary
        mean_emissions.append({
            'Scenario': key[0],
            'Year': key[1],
            'Resource': key[2],
            'Mean_EF_tCO2_per_MWh': mean_ef
        })

    # Convert the list of results to a DataFrame
    mean_emissions_df = pd.DataFrame(mean_emissions)

    # Merge the mean emissions DataFrame with scenario_aggregate_results
    updated_results = scenario_aggregate_results.merge(
        mean_emissions_df,
        on=['Scenario', 'Year', 'Resource'],
        how='left'  # Use 'left' join to preserve rows in scenario_aggregate_results
    )

    return updated_results

scenario_aggregate_results_emissions = add_mean_emissions_to_results(emission_factor_dict, scenario_aggregate_results)

# Specify columns to plot
columns_to_plot = ['%_demand_deficit', '%_excess_interm_utility', '%_excess_rooftop',
                   '%_hydro_spill','Mean_EF_tCO2_per_MWh']

# Specify output directory
output_dir = "C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont_ccgt//Plots"

# Create and save plots
save_aggregate_plots(scenario_aggregate_results_emissions, columns_to_plot, output_dir)

# Print average emission factor
#print('Average electricity emission factor is:', df_emissions['EF_tCO2_per_MWh'].mean())

##### Generate data for temporal emission plots in R

simplified_dict_all = {}

for key, df_emissions in emission_factor_dict.items():
    # Create a copy of the dataframe to avoid modifying the original
    df_emissions_simplified = df_emissions.copy()
    # Simplify the column names by merging multi-level indexes
    df_emissions_simplified.columns = ['_'.join(col).strip() for col in df_emissions_simplified.columns.values]
    simplified_dict_all[key] = df_emissions_simplified
    
emissions_all_directory = "C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont_ccgt//Emissions_Energy"

# Ensure the directory exists; create it if it doesn't
os.makedirs(emissions_all_directory, exist_ok=True)

# Save each element of the simplified_dict_all as a CSV file
for key, df in simplified_dict_all.items():
    # Construct a filename based on the key
    filename = f"All_{key[0]}_{key[1]}_{key[2]}.csv"  # e.g., Base_2024_low.csv
    filepath = os.path.join(emissions_all_directory, filename)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)    


##### Process to generate final results for thermal model

# Initialize an empty dictionary to store results
simplified_dict = {}

# Iterate over each key, value pair in the emission_factor_dict dictionary
for key, df_emissions in emission_factor_dict.items():
    # Create a copy of the dataframe to avoid modifying the original
    df_emissions_simplified = df_emissions.copy()

    # Simplify the column names by merging multi-level indexes
    df_emissions_simplified.columns = ['_'.join(col).strip() for col in df_emissions_simplified.columns.values]

    # Add the 'demand' column from the corresponding DataFrame in scenario_results
    if key in scenario_results:  # Check if the key exists in scenario_results
        df_emissions_simplified['demand'] = scenario_results[key]['demand']
    else:
        raise KeyError(f"No matching key found in scenario_results for {key}")

    # Add a new column 'kWh_interm_dispatched' by summing 'kWh_wind_dispatched' and 'kWh_solar_dispatched'
    df_emissions_simplified['kWh_interm_dispatched'] = (
        df_emissions_simplified['kWh_wind_dispatched'] + df_emissions_simplified['kWh_solar_dispatched']
    )
    
    # Add a new column 'interm_share' by dividing 'kWh_interm_dispatched' and 'Total_kWh_'
    df_emissions_simplified['interm_share'] = (
        df_emissions_simplified['kWh_interm_dispatched'] / df_emissions_simplified['Total_kWh_']
    )
    
    # Keep only the specified columns: 'kWh_wind_dispatched', 'EF_tCO2_per_MWh_', and 'demand'
    df_emissions_simplified = df_emissions_simplified[['kWh_interm_dispatched','interm_share', 'Total_Emissions_tCO2_',
                                                       'EF_tCO2_per_MWh_', 'demand']]
    
    # Add 'demand_rank' column to identify peak hour
    # Sort the DataFrame by the 'demand' column in descending order
    df_emissions_simplified = df_emissions_simplified.sort_values('demand', ascending=True)

    # Assign values from 0 to 8759 based on the sorted order
    df_emissions_simplified['demand_rank'] = range(0, len(df_emissions_simplified))

    # If you want to reverse the ranking (highest demand = 0, lowest demand = 8759)
    df_emissions_simplified['demand_rank'] = df_emissions_simplified['demand_rank'].max() - df_emissions_simplified['demand_rank']

    # Sort the DataFrame back to the original index order
    df_emissions_simplified = df_emissions_simplified.sort_index()
    
    # Create a new column for 24-hour groups (0 for the first 24 rows, 1 for the next 24, etc.)
    df_emissions_simplified['hour_group'] = np.floor(df_emissions_simplified.index / 24)

    # Apply ranking within each 24-hour group
    df_emissions_simplified['interm_share_rank'] = df_emissions_simplified.groupby('hour_group')['interm_share'] \
    .rank(method='dense', ascending=False).astype(int)
    
    # Drop the temporary 'hour_group' column if no longer needed
    df_emissions_simplified.drop(columns=['hour_group'], inplace=True)

    # Replace the original dataframe with the simplified one in the dictionary
    simplified_dict[key] = df_emissions_simplified

emissions_directory = "C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont_ccgt//Emissions"

# Ensure the directory exists; create it if it doesn't
os.makedirs(emissions_directory, exist_ok=True)

# Save each element of the simplified_dict as a CSV file
for key, df in simplified_dict.items():
    # Construct a filename based on the key
    filename = f"{key[0]}_{key[1]}_{key[2]}.csv"  # e.g., Base_2024_low.csv
    filepath = os.path.join(emissions_directory, filename)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)




df_emissions = emission_factor_dict[('Base',2024,'low')]
df_emissions_simplified = df_emissions.copy()
df_emissions_simplified.columns = ['_'.join(col).strip() for col in df_emissions_simplified.columns.values]

# Extract 'Emissions_tCO2' columns and 'Total_kWh'
partial_emissions = df_emissions['Emissions_tCO2']
total_kwh = df_emissions[('Total_kWh', '')]
total_emissions = df_emissions[('Total_Emissions_tCO2', '')]

# Combine the extracted columns into a new DataFrame
model_results = pd.concat([partial_emissions, total_emissions, total_kwh], axis=1)
# Identify the last column name
last_column = model_results.columns[-1]
# Rename the last column
model_results = model_results.rename(columns={last_column: 'Total_kWh'})
# If you also want to rename the second-to-last column
second_last_column = model_results.columns[-2]
model_results = model_results.rename(columns={second_last_column: 'Total_tCO2'})

model_results.to_csv("C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//results_base_low_hydro_update_21-05.csv")


###### Plots to compare intermittent generation
scenarios = ['Reference', 'Environmental']  # Add 'Base'
periods = [2025, 2030, 2035, 2040, 2045]      # Include 2024
hydro_resource = ['low', 'high']
variables = ['demand', 'wind_dispatched', 'solar_dispatched']

def plot_hourly_comparison_multi(df_dict, scenarios, periods, resource, variables, save_dir=None, rolling_hours=12):
    nrows = 1
    ncols = len(scenarios)
    for variable in variables:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4), sharex=True, sharey=True, squeeze=False)
        if variable != 'demand':
            rolling_hours = 24
        for j, scenario in enumerate(scenarios):
            ax = axes[0][j]
            for idx, period in enumerate(periods):
                key = (scenario, period, resource)
                if key in df_dict:
                    df = df_dict[key]
                    if variable in df.columns:
                        # Compute rolling mean for smoothing
                        y = df[variable].rolling(rolling_hours, center=True, min_periods=1).mean()/1000
                        ax.plot(
                            df.index, y,
                            label=f'Period {period}',
                            alpha=0.4,
                            linewidth=1
                        )
            ax.set_title(f'{scenario}')
            ax.set_xlabel('Hour')
            if j == 0:
                ax.set_ylabel('MWh')
            if j == ncols-1:
                ax.legend(loc='upper right')
        #plt.suptitle(f'Hourly {variable.replace("_", " ").capitalize()} (Rolling {rolling_hours}h) by Scenario and Period', y=0.95)
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        if save_dir:
            plt.savefig(f"{save_dir}//{variable}.png")        
        plt.close()


save_dir = "C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//results//midmax70//thermal_update_decom_minhydro_coalcont//Plots"
plot_hourly_comparison_multi(scenario_results, scenarios, periods, 'high', variables, save_dir)
