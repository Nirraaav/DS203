import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
import os

# Create directory for saving images if it doesn't exist
if not os.path.exists('Images'):
    os.makedirs('Images')

# Load data from CSV
data = pd.read_csv("e6-htr-current.csv", parse_dates=['Timestamp'], dayfirst=True)

# Part A: Perform EDA
print(data.describe())

# Plot the current over time using Plotly
fig = px.line(data, x='Timestamp', y='HT R Phase Current', title='HT R Phase Current over Time')
# fig.write_image("Images/ht_r_phase_current.png", scale=1, width=1920, height=1080, format='png')
fig.show()

# Part B: Identify a 2-week unstable period
unstable_period = data[(data['Timestamp'] >= '2019-07-30') & (data['Timestamp'] <= '2019-08-14')].copy()

# Plot the unstable period
plt.figure(figsize=(16, 10))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], color='blue')
plt.title('HT R Phase Current - Unstable Period')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.savefig('Images/unstable_period.png', dpi=300)
plt.show()

# Part D: Remove outliers, smoothen, and impute missing data

# Method 1: Imputation (Replacing outlier values with the mean)
mean_value = unstable_period['HT R Phase Current'].mean()
median_value = unstable_period['HT R Phase Current'].median()

# Impute mean
unstable_period['Current_Imputed_Mean'] = unstable_period['HT R Phase Current'].copy()
unstable_period.loc[(unstable_period['HT R Phase Current'] > unstable_period['HT R Phase Current'].quantile(0.95)) |
                    (unstable_period['HT R Phase Current'] < unstable_period['HT R Phase Current'].quantile(0.05)), 
                    'Current_Imputed_Mean'] = mean_value

# Impute median
unstable_period['Current_Imputed_Median'] = unstable_period['HT R Phase Current'].copy()
unstable_period.loc[(unstable_period['HT R Phase Current'] > unstable_period['HT R Phase Current'].quantile(0.95)) |
                    (unstable_period['HT R Phase Current'] < unstable_period['HT R Phase Current'].quantile(0.05)), 
                    'Current_Imputed_Median'] = median_value

# Method 2: Trimming (Removing outliers)
q_low = unstable_period['HT R Phase Current'].quantile(0.1)
q_high = unstable_period['HT R Phase Current'].quantile(0.9)
unstable_period_trimmed = unstable_period[(unstable_period['HT R Phase Current'] >= q_low) & 
                                          (unstable_period['HT R Phase Current'] <= q_high)]

# Method 3: Capping (Setting a cap on the maximum and minimum values)
max_value = unstable_period['HT R Phase Current'].quantile(0.95)
min_value = unstable_period['HT R Phase Current'].quantile(0.05)
unstable_period['Current_Capped'] = unstable_period['HT R Phase Current'].copy()
unstable_period['Current_Capped'] = np.where(unstable_period['Current_Capped'] > max_value, max_value, unstable_period['Current_Capped'])
unstable_period['Current_Capped'] = np.where(unstable_period['Current_Capped'] < min_value, min_value, unstable_period['Current_Capped'])

# Method 4: Robust Estimation (Using RANSAC regression)
# Prepare the data for RANSAC
X = unstable_period['Timestamp'].astype(np.int64) // 10**9  # Convert datetime to timestamp in seconds
X = X.values.reshape(-1, 1)  # Reshape for RANSAC
y = unstable_period['HT R Phase Current'].values

# Fit RANSAC
model = RANSACRegressor(LinearRegression()).fit(X, y)
unstable_period['Current_Robust'] = model.predict(X)

# Method 5: Local Regression (Loess)
# Apply loess smoothing on the unstable period to get the local trend
loess_smoothed = lowess(unstable_period['HT R Phase Current'], 
                        unstable_period['Timestamp'].astype(np.int64) // 10**9, 
                        frac=0.1)
unstable_period['Current_Loess'] = loess_smoothed[:, 1]

# Plot Imputed data (Mean)
plt.figure(figsize=(10, 6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Imputed_Mean'], label='Imputed Data (Mean)', color='green')
plt.title('HT R Phase Current - Imputation (Mean)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/imputed_mean_data.png', dpi=300)
plt.show()

# Plot Imputed data (Median)
plt.figure(figsize=(10, 6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Imputed_Median'], label='Imputed Data (Median)', color='orange')
plt.title('HT R Phase Current - Imputation (Median)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/imputed_median_data.png', dpi=300)
plt.show()

# Plot Trimmed data
plt.figure(figsize=(10, 6))
plt.plot(unstable_period_trimmed['Timestamp'], unstable_period_trimmed['HT R Phase Current'], label='Trimmed Data', color='purple')
plt.title('HT R Phase Current - Trimming (10th and 90th Percentile)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/trimmed_data.png', dpi=300)
plt.show()

# Plot Capped data
plt.figure(figsize=(10, 6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Capped'], label='Capped Data', color='red')
plt.title('HT R Phase Current - Capping')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/capped_data.png', dpi=300)
plt.show()

# Plot Robust estimation
plt.figure(figsize=(10, 6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Robust'], label='Robust Estimation', color='orange')
plt.title('HT R Phase Current - Robust Estimation (RANSAC)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/robust_estimation_data.png', dpi=300)
plt.show()

# Plot Loess estimation
plt.figure(figsize=(10, 6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Loess'], label='Loess Smoothed Data', color='purple')
plt.title('HT R Phase Current - Local Regression (Loess)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/loess_estimation_data.png', dpi=300)
plt.show()

# Combining the Techniques: Original, Imputed (Mean), Imputed (Median), Trimmed, Capped, Robust, Loess
plt.figure(figsize=(12, 8))

# Original Data
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue', alpha=0.6)

# Imputed Data (Mean)
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Imputed_Mean'], label='Imputed (Mean)', color='green', alpha=0.8)

# Imputed Data (Median)
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Imputed_Median'], label='Imputed (Median)', color='orange', alpha=0.8)

# Trimmed Data
plt.plot(unstable_period_trimmed['Timestamp'], unstable_period_trimmed['HT R Phase Current'], label='Trimmed', color='purple', alpha=0.8)

# Capped Data
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Capped'], label='Capped', color='red', alpha=0.8)

# Robust Estimation
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Robust'], label='Robust Estimation', color='orange', alpha=0.8)

# Loess Smoothed Data
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Loess'], label='Loess Smoothed', color='brown', alpha=0.8)

# Final Plot Details
plt.title('Combining Techniques: Data Cleaning and Smoothing')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/combined_techniques.png', dpi=300)
plt.show()
