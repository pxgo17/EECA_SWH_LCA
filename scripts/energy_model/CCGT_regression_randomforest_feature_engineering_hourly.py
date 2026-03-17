# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:43:34 2024

@author: pxg11
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Load the dataset
data = pd.read_csv("C:/Users/pxg11/OneDrive - University of Canterbury/EECA_LCA_RFP//generation_data//thermal_fossil//CCGT_model_3y.csv")

# Convert timestamp to datetime
data['hourly_timestamp'] = pd.to_datetime(data['hourly_timestamp'])

# Set the 'timestamp' column as the index
data.set_index('hourly_timestamp', inplace=True)

# Analysis will be carried out in hourly series, hence diff and shares need to be recalculated after aggregation
subset = ['hourly_timestamp', 'Geo_Geo', 'interm', 'storage_GWh', 'inflow_MW', 'demand',
          'lag_ccgt','demand_next_8','geo_next_8','inflow_next_8','interm_next_8',
          'dispatch_mode']

# Resample the data to hourly frequency and apply different aggregation functions
# agg_funcs = {
#     'Geo_Geo': 'sum',
#     'interm': 'sum',
#     'storage_GWh': 'mean',
#     'inflow_MW': 'mean',
#     'demand': 'sum',
#     'lag_coal': 'sum',
#     'demand_next_8': 'mean',
#     'geo_next_8': 'mean',
#     'inflow_next_8': 'mean',
#     'interm_next_8': 'mean',
#     'dispatch_mode': 'mean'
#     }

# Resample the data to hourly frequency and sum the values
# data = data.resample('h').agg(agg_funcs)

# Reset the index if you want 'timestamp' to become a column again
data.reset_index(inplace=True)
# Drop rows with NaN values (from lagged features)
data = data.dropna()

#data['dispatch_mode'] = np.floor(data['dispatch_mode']).astype(int)

# Select features and target variable
X = data[['Geo_Geo', 'interm', 'storage_GWh', 'inflow_MW', 'demand', 
          'lag_ccgt', 'demand_next_8','geo_next_8', 'inflow_next_8']]
y = data['dispatch_mode']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#################################################################
# Build the RandomForestClassifier model
model_RF = RandomForestClassifier(n_estimators=300, random_state=42)

# Train the model
model_RF.fit(X_train, y_train)

# Predict on test data
y_pred = model_RF.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

labels = ['No Generation', 'Ramp up', 'Steady']
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_RF.classes_)
ax = disp.plot(cmap='Blues')
plt.xlabel('Predicted regime')
plt.ylabel('True regime')
plt.title('Confusion Matrix')
cbar = ax.figure_.axes[-1]  # The colorbar is the last axes in the figure
cbar.set_ylabel('Hours', rotation=270, labelpad=15)
plt.show()

# Save model and scaler to be reused in another script
joblib.dump(model_RF, 
            'C://Users//pxg11//OneDrive - University of Canterbury//EECA_LCA_RFP//energy_model//ccgt_dispatch.pkl')


# Make a copy of the original DataFrame
data_with_predictions = data.copy()

#Prepare feature columns for prediction
X_all = data_with_predictions[['Geo_Geo', 'interm', 'storage_GWh', 'inflow_MW', 'demand',
                            'lag_coal']]

# Step 3: Predict 'dispatch_mode' using the trained model
# Note: If you scaled the features during training, make sure to scale them here as well.
data_with_predictions['dispatch_mode_predicted'] = model_RF.predict(X_all)

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
