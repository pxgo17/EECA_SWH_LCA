# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:43:34 2024

@author: pxg11
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv("C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//generation_data//hydro_resource//hydro_model_4y_geo_inc.csv")

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['inflow_MWh'] = 0.5*data['inflow_MW']

data['timestamp'] = pd.to_datetime(data['timestamp'])

# Set the 'timestamp' column as the index
data.set_index('timestamp', inplace=True)

# Resample the data to hourly frequency and apply different aggregation functions
agg_funcs = {
    'Hydro_Hydro': 'sum',
    'Wind_Wind': 'sum',
    'Solar_Solar': 'sum',
    'Geo_Geo': 'sum',
    'kWh': 'sum',
    'diff': 'sum',
    'interm': 'sum',
    'storage_GWh': 'first',  # Keep only the first value for each hour
    'inflow_MW': 'mean',
    'inflow_MWh': 'sum'
}

# Resample the data to hourly frequency and sum the values
data = data.resample('h').agg(agg_funcs)

# Reset the index if you want 'timestamp' to become a column again
data.reset_index(inplace=True)

# This code is for understanding inflow, dispatch and storage logic 
# data_2022 = data[data['timestamp'].dt.year == 2022]

# data_2022['storage_est'] = data_2022['storage_GWh'].shift(1) + ((1000*0.5*data_2022['inflow_MW'] - data_2022['Hydro_Hydro'])/1000000)

# # Drop rows with NA values in 'storage_GWh' or 'estimated_storage_GWh' columns
# data_2022 = data_2022.dropna(subset=['storage_est'])

# # Plotting the actual and estimated storage
# plt.figure(figsize=(12, 6))
# plt.plot(data_2022['timestamp'], data_2022['storage_GWh'], label='Actual Storage (GWh)', color='blue')
# plt.plot(data_2022['timestamp'], data_2022['storage_est'], label='Estimated Storage (GWh)', color='orange', alpha = 0.6)
# # Adding labels, title, and legend
# plt.xlabel('Timestamp')
# plt.ylabel('Storage (GWh)')
# plt.title('Actual vs Estimated Storage Over Time')
# plt.legend()
# # Display the plot
# plt.show()

# Need to aggregate for further model calibration using hourly series


# Select features and target variable
X = data[['diff', 'interm', 'storage_GWh', 'inflow_MWh']]
y = data['Hydro_Hydro']

############################## Rolling window approach #####################

# # Convert timestamp to datetime
# # Set timestamp as index
# data_2022.set_index('timestamp', inplace=True)

# # Initialize the predicted_hydro column
# data_2022['predicted_hydro'] = 0

# # Define the window size (7 days)

# period_days = 7
# window_size = period_days * 24 * 2  # period_days * 24 hours/day * 2 (for half-hourly data)

# # Calculate predicted_hydro iteratively
# for i in range(len(data_2022)):
#     end_idx = i + window_size
#     if end_idx > len(data_2022):
#         end_idx = len(data_2022)
    
#     subset = data_2022.iloc[i:end_idx]
#     total_hydro = subset['Hydro_Hydro'].sum()
#     total_diff = subset['diff'].sum()
    
#     if total_diff != 0:
#         data_2022.loc[subset.index, 'predicted_hydro'] = (subset['diff'] / total_diff) * total_hydro

# # Reset the index
# data_2022.reset_index(inplace=True)

# # Plotting the actual and estimated storage
# plt.figure(figsize=(12, 6))
# plt.plot(data_2022['timestamp'], data_2022['Hydro_Hydro'], label='Actual Dispatch', color='blue')
# plt.plot(data_2022['timestamp'], data_2022['predicted_hydro'], label='Estimated Dispatch', color='orange', alpha = 0.6)
# # Adding labels, title, and legend
# plt.xlabel('Timestamp')
# plt.ylabel('kWh')
# plt.title('Actual vs Estimated Dispatch Over Time')
# plt.legend()
# # Display the plot
# plt.show()

# # Plot the original Hydro_Hydro column against the predicted one
# plt.figure(figsize=(10, 6))
# plt.plot(data_2022['day_of_year'], data_2022['Hydro_Hydro'], label='Original Hydro Power', alpha=0.7)
# plt.plot(data_2022['day_of_year'], data_2022['predicted_hydro'], label='Predicted Hydro Power', alpha=0.5)
# plt.xlabel('day_of_year')
# plt.ylabel('kWh')
# plt.title('Original vs Predicted Hydro Power Generation - rolling window approach')
# plt.legend()
# plt.xticks(fontsize=8)
# plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//hydro_pred_vs_real_series_RW_30.png')


############################################################################

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the RandomForestRegressor model
model_RF = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model_RF.fit(X_train_scaled, y_train)

# Save model and scaler to be reused in another script
joblib.dump(model_RF, 
            'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//hydro_dispatch.pkl')
joblib.dump(scaler, 
            'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//hydro_dispatch_scaler.pkl')

# Predict values
data['Predicted_Hydro_Hydro_RF'] = model_RF.predict(scaler.transform(X))

# Convert timestamp to day of the year
data['day_of_year'] = pd.to_datetime(data['timestamp']).dt.dayofyear
data['year'] = pd.to_datetime(data['timestamp']).dt.year

# Calculate absolute error
data['%_Error'] = (100*(data['Hydro_Hydro'] - data['Predicted_Hydro_Hydro_RF'])/data['Hydro_Hydro'])

# Perform error analysis
mse = mean_squared_error(y_test, model_RF.predict(X_test_scaled))
mae = mean_absolute_error(y_test, model_RF.predict(X_test_scaled))
mape = mean_absolute_percentage_error(y_test, model_RF.predict(X_test_scaled))

print(f'Mean Absolute Percentage Error: {mape}')

# Print mean absolute percent error for a specific year
data.loc[data['year'] == 2024, '%_Error'].abs().mean()

# Plotting function
def save_yearly_plot(year, df):
    yearly_data = df[df['year'] == year]
    yearly_data['timestamp'] = pd.to_datetime(yearly_data['timestamp'])
    yearly_data['hour_of_year'] = yearly_data['timestamp'].dt.dayofyear * 24 + yearly_data['timestamp'].dt.hour
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot original and predicted values on the primary y-axis
    ax1.plot(yearly_data['hour_of_year'], yearly_data['Hydro_Hydro']/1000, label='Original Hydro Power', color='#41b496', alpha=0.7, linewidth=0.5)
    ax1.plot(yearly_data['hour_of_year'], yearly_data['Predicted_Hydro_Hydro_RF']/1000, label='Predicted Hydro Power', color='#ed6d63', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Hour of Year')
    ax1.set_ylabel('MW')
    ax1.set_title('Original vs Predicted Hydro Power Generation')
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='x', labelsize=5)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(50))  # Limit the number of x-axis labels
    ax1.set_xlim(0, 8760)
    
    # Rotate x-axis labels to avoid overlap
    plt.xticks(rotation=45)
    
    # Create a secondary y-axis for the absolute error
    ax2 = ax1.twinx()
    ax2.scatter(yearly_data['hour_of_year'], yearly_data['%_Error'], label='% Error', color='#163f57', alpha=0.4, s=5)
    ax2.set_ylabel('% Error')
    ax2.set_ylim(-100, 100)  # Set y-axis range from -100 to 100
    ax2.legend(loc='upper right')
    
    plt.savefig(f'C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP/hydro_pred_vs_real_series_RF_{year}.png')  # Save the plot as a PNG file
    plt.close()

# Generate plots for each year and save them as PNG files
years = data['year'].unique()
for year in years:
    save_yearly_plot(year, data)

# Visualize real versus predicted values
# Plot the original Hydro_Hydro column against the predicted one
plt.figure(figsize=(10, 6))
plt.plot(data['day_of_year'], data['Hydro_Hydro'], label='Original Hydro Power', alpha=0.7)
plt.plot(data['day_of_year'], data['Predicted_Hydro_Hydro_RF'], label='Predicted Hydro Power', alpha=0.5)
plt.xlabel('Day of the Year')
plt.ylabel('kWh')
plt.title('Original vs Predicted Hydro Power Generation')
plt.legend()
plt.xticks(fontsize=8)
plt.show()
plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//hydro_pred_vs_real_series_RF.png')

# Scatter Plot: Actual vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(data['Hydro_Hydro']/1000, data['Predicted_Hydro_Hydro_RF']/1000, alpha=0.3, color='#447474', s=5)
plt.xlabel("Actual (MW)")
plt.ylabel("Predicted (MW)")
plt.title("Actual vs. Predicted")
plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//hydro_pred_vs_real_scatter_RF.png')


# Filter the dataset to include only the first 100 days
data_f100 = data[data['day_of_year'] <= 100]
# Plot the original Hydro_Hydro column against the predicted one
plt.figure(figsize=(10, 6))
plt.plot(data_f100['day_of_year'], data_f100['Hydro_Hydro'], label='Original Hydro Power', alpha=0.7)
plt.plot(data_f100['day_of_year'], data_f100['Predicted_Hydro_Hydro_RF'], label='Predicted Hydro Power', alpha=0.5)
plt.xlabel('Day of the Year')
plt.ylabel('kWh')
plt.title('Original vs Predicted Hydro Power Generation')
plt.legend()
plt.xticks(fontsize=8)
plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//hydro_pred_vs_real_series_RF_100.png')

