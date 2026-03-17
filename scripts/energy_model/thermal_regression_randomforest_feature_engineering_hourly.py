# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:43:34 2024

@author: pxg11
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
#from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv("C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//generation_data//thermal_fossil//thermal_model_4y_new.csv")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Set the 'timestamp' column as the index
data.set_index('timestamp', inplace=True)

# Analysis will be carried out in hourly series, hence diff and shares need to be recalculated after aggregation
subset = ['timestamp', 'Hydro_Hydro', 'Wind_Wind', 'Solar_Solar', 'Thrml_Coal',
       'Thrml_Gas', 'Geo_Geo', 'kWh']

# Resample the data to hourly frequency and apply different aggregation functions
agg_funcs = {
    'Hydro_Hydro': 'sum',
    'Wind_Wind': 'sum',
    'Solar_Solar': 'sum',
    'Thrml_Coal': 'sum',
    'Thrml_Gas': 'sum',
    'Geo_Geo': 'sum',
    'kWh': 'sum'
}

# Resample the data to hourly frequency and sum the values
data = data.resample('h').agg(agg_funcs)

# Reset the index if you want 'timestamp' to become a column again
data.reset_index(inplace=True)

data['diff'] = data['kWh'] - (data['Hydro_Hydro'] + data['Wind_Wind'] + data['Solar_Solar'] + data['Geo_Geo'])
data['share_gas'] = data['Thrml_Gas'] / (data['Thrml_Gas'] + data['Thrml_Coal'])


# Feature engineering
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month

# Lagged coal generation
data['Thrml_Coal_lag1'] = data['Thrml_Coal'].shift(1)
# data['Thrml_Coal_lag2'] = data['Thrml_Coal'].shift(2)

# Lead time for coal generation
# data['coal_lead_time'] = data['Thrml_Coal'].rolling(window=6, min_periods=1).apply(lambda x: (x > 1000).argmax())

# Residual demand change
# data['diff_change'] = data['diff'].diff()

# Drop rows with NaN values (from lagged features)
data = data.dropna()

# Select features and target variable
X = data[['diff', 'kWh', 'hour', 'day_of_week', 'month', 'Thrml_Coal_lag1']]

X = data[['diff', 'kWh', 'day_of_week', 'month', 'Thrml_Coal_lag1']]
y = data['share_gas']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Hyperparameter tuning
# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# model_RF = RandomForestRegressor(random_state=42)

# # Perform Grid Search
# grid_search = GridSearchCV(estimator=model_RF, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error')
# grid_search.fit(X_train, y_train)

# # Get the best parameters and score
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)

# print("Best Parameters:", best_params)
# print("Mean Absolute Error:", mae)


#################################################################
# Build the RandomForestRegressor model
model_RF = RandomForestRegressor(n_estimators=300, random_state=42)

# Train the model
model_RF.fit(X_train_scaled, y_train)

# Predict values
data['Predicted_Gas_share_RF'] = model_RF.predict(scaler.transform(X))


# Perform error analysis
# In this case the error should be multiplied by 100 for interpretation
# Ex: 0.029*100 = 2.9%. On average, the error between actual share of gas 
# versus predicted is 2.9%

mse = mean_squared_error(y_test, model_RF.predict(X_test_scaled))
mae = mean_absolute_error(y_test, model_RF.predict(X_test_scaled))
mape = mean_absolute_percentage_error(y_test, model_RF.predict(X_test_scaled))


print(f'Mean Absolute Error: {mae}')

# Save model and scaler to be reused in another script
joblib.dump(model_RF, 
            'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//thermal_dispatch.pkl')
joblib.dump(scaler, 
            'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//thermal_dispatch_scaler.pkl')

# Plotting function
def save_yearly_plot(year, df):
    yearly_data = df[df['year'] == year]
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_data['day_of_year'], yearly_data['share_gas'], label='Original Gas Share', alpha=0.7)
    plt.plot(yearly_data['day_of_year'], yearly_data['Predicted_Gas_share_RF'], label='Predicted Gas Share', alpha=0.5)
    plt.title(f'Original vs Predicted Share of Gas Power Generation for {year}')
    plt.xlabel('Day of the Year')
    plt.ylabel('%')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//gas_share_pred_vs_real_series_RF_{year}.png')  # Save the plot as a PNG file
    plt.close()


# Generate plots for each year and save them as PNG files
years = data['year'].unique()
for year in years:
    save_yearly_plot(year, data)
    
    
# Visualize real versus predicted values
# Plot the original column against the predicted one
plt.figure(figsize=(10, 6))
plt.plot(data['day_of_year'], data['share_gas'], label='Original Gas Share', alpha=0.7)
plt.plot(data['day_of_year'], data['Predicted_Gas_share_RF'], label='Predicted Gas Share', alpha=0.5)
plt.xlabel('Day of the Year')
plt.ylabel('%')
plt.title('Original vs Predicted Share of Gas Power Generation')
plt.legend()
plt.xticks(fontsize=8)
plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//gas_share_pred_vs_real_series_RF.png')

# Scatter Plot: Actual vs. Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(data['share_gas'], data['Predicted_Gas_share_RF'], alpha=0.5)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted")
plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//gas_share_pred_vs_real_scatter_RF.png')


# Filter the dataset to include only the first 100 days
data_f100 = data[data['day_of_year'] <= 100]
# Plot the original Hydro_Hydro column against the predicted one
plt.figure(figsize=(10, 6))
plt.plot(data_f100['day_of_year'], data_f100['share_gas'], label='Original Gas Share', alpha=0.7)
plt.plot(data_f100['day_of_year'], data_f100['Predicted_Gas_share_RF'], label='Predicted Gas Share', alpha=0.5)
plt.xlabel('Day of the Year')
plt.ylabel('%')
plt.title('Original vs Predicted Share of Gas Power Generation')
plt.legend()
plt.xticks(fontsize=8)
plt.savefig('C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//gas_share_pred_vs_real_series_RF_100.png')
