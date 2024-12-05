import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset from CSV file
data = pd.read_csv('CCDC_5mTree_SyntheticBands_Target_Data_2013.csv')

# Drop rows where 'Tree cover' column has NaN values.
data = data.dropna()

# Separate target variable (y) and predictor variables (X)
X = data.drop(columns=['b1','latitude','longitude','.geo','system:index','GEZ_TERM'])

# After a few attempts of data optimization, the month of Feb, May, Aug and Nov were chose to represent the seasonality

# Select columns that contain '_Feb', '_May', '_Aug', or '_Nov'
selected_columns = X.filter(regex='(_Feb|_May|_Aug|_Nov)').columns
X_selected = X[selected_columns]

# Similarly, out of all the bands and indices, the green, blue and swir2 bands along wtih temp and ndvi indices were chosen to be most important for the purpose of this project
selected_columns_2 = X_selected.filter(regex=r'^(GREEN_|BLUE_|SWIR2_|TEMP_|NDVI_)').columns
X_minimal = X_selected[selected_columns_2]

# Add the 'Slope', 'Aspect', and 'DEM' columns
X_minimal = pd.concat([X_minimal, X[['Slope', 'Aspect', 'DEM']]], axis=1)

y = data['b1']
y = y/100

# Split the dataset into training and testing sets
X_test, X_valid, y_test, y_valid = train_test_split(X_tested, y_tested, test_size=0.33, random_state=42)

# Plot histogram to visualize data distribution of training samples
plt.figure(figsize=(12, 6))
plt.hist(y_train, bins=10, alpha=0.7, color='blue', edgecolor='black')
plt.title('Histogram of y_train Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# =================================================================
# AUTOMATIC GRID SEARCH FOR BEST FIT PARAMETERS FOR THE REGRESSOR
# =================================================================

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# ===============
# MODEL TRAINING
# ===============

# Train a Random Forest regressor with the specified best parameters
rf = RandomForestRegressor(
    max_depth=30,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

# Save the model to a file
model_filename = 'Tree5M_Minimal_selected_NonStandardized_X_FractionalStratY_RF_Reg_model_SyntheticBandIndices.joblib'
joblib.dump(rf, model_filename)

# Load the saved model
model_filename = 'Tree5M_Minimal_selected_NonStandardized_X_FractionalStratY_RF_Reg_model_SyntheticBandIndices.joblib'
rf = joblib.load(model_filename)

# Predict on the test set
y_pred_tested = rf.predict(X_tested)

# =================
# MODEL PERFORMANCE
# =================

# Calculate mean_squared_error for Regression
mse = mean_squared_error(y_tested, y_pred_tested)
print("Mean Squared Error:", mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_tested, y_pred_tested)
print("Mean Absolute Error (MAE):", mae)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# R-squared (R²)
r2 = r2_score(y_tested, y_pred_tested)
print("R-squared (R²):", r2)

# =======================================
# VARIABLE IMPORTANCE OF TREE COVER MODEL
# =======================================

# Get the names of the predictor variables
variable_names = X_minimal.columns.tolist()

# Calculate variable importance
variable_importance = rf.feature_importances_

# Create a DataFrame for easier manipulation
importance_df = pd.DataFrame({'Variable': variable_names, 'Importance': variable_importance})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Print the variable importance in decreasing order
print("Variable Importance in Decreasing Order:")
print(importance_df)

# Create a bar plot for variable importance
plt.figure(figsize=(15, 25))
plt.barh(importance_df['Variable'], importance_df['Importance'], color='#0072B2')
plt.xlabel('Variable Importance')
plt.ylabel('Variables')
plt.title('Variable Importance Plot - Tree Cover Model')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important variable at the top
plt.savefig('VIF_RF1.png', dpi=300, bbox_inches='tight')# Show plot

plt.show()

# ==================================
# MODEL UNCERTAINTY VISUALIZATION
# ==================================

# Estimate uncertainty
# Get predictions from each tree
all_tree_predictions = np.array([tree.predict(X_tested.values) for tree in rf.estimators_])

# Calculate the mean and standard deviation of the predictions
mean_predictions = np.mean(all_tree_predictions, axis=0)
std_predictions = np.std(all_tree_predictions, axis=0)

# Plotting mean predictions vs standard deviation
plt.figure(figsize=(10, 6))
plt.scatter(mean_predictions, std_predictions, alpha=0.6)
plt.title('Mean Predictions vs. Standard Deviation - Tree Cover Model')
plt.xlabel('Mean Predictions')
plt.ylabel('Standard Deviation (Uncertainty)')
# plt.grid(True)
plt.savefig('MeanVsStDev_TreeCoverModel.png', dpi=300, bbox_inches='tight')# Show plot

plt.show()

# ===============================================================
# BIAS CORRECTION USING MODEL RESIDUAL IN RANDOM FOREST REGRESSOR
# ===============================================================

residuals = y_test - rf.predict(X_test)

# Train a Random Forest regressor with the specified best parameters
rf_residual_Train = RandomForestRegressor(
    max_depth=30,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=200,
    random_state=42)
rf_residual_Train.fit(X_test, residuals)

# Save the model to a file
model_filename = 'Tree5M_Minimal_selected_BiasCorrectionModel_SyntheticBandIndice_RandomForestRegressor.joblib'
joblib.dump(rf_residual_Train, model_filename)

# Load the saved model for residual prediction
model_filename = 'Tree5M_Minimal_selected_BiasCorrectionModel_SyntheticBandIndice_RandomForestRegressor.joblib'
rf_residual_Train = joblib.load(model_filename)

# Predict on the valid set along with residual correction
y_pred_valid = rf.predict(X_valid)
residuals_pred_valid = rf_residual_Train.predict(X_valid)

y_pred_corrected_valid = y_pred_valid + residuals_pred_valid
y_pred_corrected_valid = np.maximum(y_pred_corrected_valid, 0)
y_pred_corrected_valid = np.minimum(y_pred_corrected_valid, 1)

# ======================================
# VARIABLE IMPORTANCE OF RESIDUAL MODEL
# ======================================
import pandas as pd
import matplotlib.pyplot as plt

# Get the names of the predictor variables
variable_names = X_minimal.columns.tolist()

# Calculate variable importance
variable_importance = rf_residual_Train.feature_importances_

# Create a DataFrame for easier manipulation
importance_df = pd.DataFrame({'Variable': variable_names, 'Importance': variable_importance})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Print the variable importance in decreasing order
print("Variable Importance in Decreasing Order:")
print(importance_df)

# Create a bar plot for variable importance
plt.figure(figsize=(15, 25))
plt.barh(importance_df['Variable'], importance_df['Importance'], color='#0072B2')
plt.xlabel('Variable Importance')
plt.ylabel('Variables')
plt.title('Variable Importance Plot - Residual Model')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important variable at the top
plt.savefig('VIF_ResidualModel.png', dpi=300, bbox_inches='tight')# Show plot

plt.show()


# ==================================================
# MODEL COMPARISON BEFORE AND AFTER BIAS CORRECTION
# ==================================================

# Calculate and compare the Mean Squared Error (MSE)
mse_before = mean_squared_error(y_valid, y_pred_valid)
mse_after = mean_squared_error(y_valid, y_pred_corrected_valid)

print(f"MSE before bias correction: {mse_before}")
print(f"MSE after bias correction: {mse_after}")

# Calculate and print the Mean Absolute Error (MAE)
mae_before = mean_absolute_error(y_valid, y_pred_valid)
mae_after = mean_absolute_error(y_valid, y_pred_corrected_valid)

print(f"MAE before bias correction: {mae_before}")
print(f"MAE after bias correction: {mae_after}")

# Calculate and print the Root Mean Squared Error (RMSE)
rmse_before = np.sqrt(mse_before)
rmse_after = np.sqrt(mse_after)

print(f"RMSE before bias correction: {rmse_before}")
print(f"RMSE after bias correction: {rmse_after}")

# Calculate and print the R-squared (R²)
r2_before = r2_score(y_valid, y_pred_valid)
r2_after = r2_score(y_valid, y_pred_corrected_valid)

print(f"R-squared (R²) before bias correction: {r2_before}")
print(f"R-squared (R²) after bias correction: {r2_after}")

# ==============================================================
# DATA VISUALIZATION USING HEXBIN PLOT AND RESIDUAL DENSITY PLOT
# ==============================================================

## HEXBIN PLOT

# Calculate residuals for both predictions
residuals_before = y_valid - y_pred_valid
residuals_after = y_valid - y_pred_corrected_valid

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Hexbin Plot: Actual vs Predicted Values
hb1 = axs[0].hexbin(y_valid, y_pred_valid, gridsize=25, cmap='Blues', mincnt=1, vmin=0, vmax=30)
axs[0].set_title('Hexbin: Actual vs Predicted Values')
axs[0].set_xlabel('Actual Values (y)')
axs[0].set_ylabel('Predicted Values')
axs[0].plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='black', linestyle='--')
cb1 = plt.colorbar(hb1, ax=axs[0], label='Counts')

# Hexbin Plot: Actual vs Bias Corrected Predicted Values
hb2 = axs[1].hexbin(y_valid, y_pred_corrected_valid, gridsize=25, cmap='Blues', mincnt=1, vmin=0, vmax=30)
axs[1].set_title('Hexbin: Actual vs Bias Corrected Predicted Values')
axs[1].set_xlabel('Actual Values (y)')
axs[1].set_ylabel('Bias Corrected Predicted Values')
axs[1].plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='black', linestyle='--')
cb2 = plt.colorbar(hb2, ax=axs[1], label='Counts')
plt.savefig('Mexico_Actual_Vs_Predicted.png', dpi=300, bbox_inches='tight')# Show plot

plt.tight_layout()
plt.show()

## RESIDUAL DENSITY
from scipy.stats import gaussian_kde

# Calculate residuals for both predictions
residuals_before = y_valid - y_pred_valid
residuals_after = y_valid - y_pred_corrected_valid

# Create a new figure for residual line plots
plt.figure(figsize=(12, 6))

# Calculate density estimates for residuals
density_before = gaussian_kde(residuals_before)
density_after = gaussian_kde(residuals_after)

# Create a range of residual values for the x-axis
x_vals = np.linspace(min(residuals_before.min(), residuals_after.min()), 
                     max(residuals_before.max(), residuals_after.max()), 500)

# Plot the density estimates as line plots
plt.plot(x_vals, density_before(x_vals), color='orange', label='Before Bias Correction', linewidth=2)
plt.plot(x_vals, density_after(x_vals), color='green', label='After Bias Correction', linewidth=2)

# Adding a vertical line at zero
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Title and labels
plt.title('Residuals Distribution - Before and After Bias Correction')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Density')

# Add legend
plt.legend()

# Save the figure
plt.savefig('Residual_Distribution.png', dpi=300, bbox_inches='tight')

# Show plot
plt.tight_layout()
plt.show()

