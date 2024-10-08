import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import mstats
import os

# Create directory for saving images
if not os.path.exists('Images'):
    os.makedirs('Images')

# Load data from CSV
data = pd.read_csv("e6-htr-current.csv", parse_dates=['Timestamp'], dayfirst=True)

# Part A: Perform EDA
print(data.describe())

# Plot the current over time using Plotly
fig = px.line(data, x='Timestamp', y='HT R Phase Current', title='HT R Phase Current over Time')
fig.show()

# Part B: Identify a 2-week unstable period
unstable_period = data[(data['Timestamp'] >= '2019-07-30') & (data['Timestamp'] <= '2019-08-14')].copy()

# Plot the unstable period
plt.figure(figsize=(16,10))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], color='blue')
plt.title('HT R Phase Current - Unstable Period')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.savefig('Images/unstable_period.png')
plt.show()

# # Part D: Remove outliers, smoothen, and impute missing data

# Method 1: Winsorization (Capping extreme values at the 5th and 95th percentiles)
ht_r_current = unstable_period['HT R Phase Current'].copy()

lower_percentile = 0.35
upper_percentile = 0.65

# Check the 35th and 65th percentiles to understand data range
q_low = ht_r_current.quantile(lower_percentile)
q_high = ht_r_current.quantile(upper_percentile)

print(f"35th Percentile: {q_low}, 65th Percentile: {q_high}")

# Apply Winsorization
unstable_period['Current_Winsorized'] = mstats.winsorize(ht_r_current, limits=[lower_percentile, upper_percentile])

# Verify Winsorized data (it shouldn't be 0 unless there's a reason)
print(unstable_period[['HT R Phase Current', 'Current_Winsorized']].head())

# Method 2: Imputation (using linear interpolation to fill missing values)
unstable_period['Current_Imputed'] = unstable_period['HT R Phase Current'].interpolate()

# Method 3: Trimming (Removing data outside the 10th and 90th percentiles)
q_low = unstable_period['HT R Phase Current'].quantile(0.1)
q_high = unstable_period['HT R Phase Current'].quantile(0.9)
unstable_period_trimmed = unstable_period[(unstable_period['HT R Phase Current'] >= q_low) & (unstable_period['HT R Phase Current'] <= q_high)]

# Plot Winsorized data
plt.figure(figsize=(10,6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Winsorized'], label='Winsorized Data', color='orange')
plt.title('HT R Phase Current - Winsorization')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/winsorized_data.png')
plt.show()

# Plot Imputed data
plt.figure(figsize=(10,6))
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue')
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Imputed'], label='Imputed Data', color='green')
plt.title('HT R Phase Current - Imputation (Linear Interpolation)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/imputed_data.png')
plt.show()

# Plot Trimmed data
plt.figure(figsize=(10,6))
plt.plot(unstable_period_trimmed['Timestamp'], unstable_period_trimmed['HT R Phase Current'], label='Trimmed Data', color='purple')
plt.title('HT R Phase Current - Trimming (5th and 95th Percentile)')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/trimmed_data.png')
plt.show()

# Combining the Techniques: Original, Winsorized, Imputed, Trimmed
plt.figure(figsize=(12,8))

# Original Data
plt.plot(unstable_period['Timestamp'], unstable_period['HT R Phase Current'], label='Original Data', color='blue', alpha=0.6)

# Winsorized Data
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Winsorized'], label='Winsorized', color='orange', alpha=0.8)

# Imputed Data
plt.plot(unstable_period['Timestamp'], unstable_period['Current_Imputed'], label='Imputed', color='green', alpha=0.8)

# Trimmed Data
plt.plot(unstable_period_trimmed['Timestamp'], unstable_period_trimmed['HT R Phase Current'], label='Trimmed', color='purple', alpha=0.8)

plt.title('HT R Phase Current - Comparison of Cleaning Methods')
plt.xlabel('Timestamp')
plt.ylabel('Current')
plt.legend()
plt.savefig('Images/cleaning_comparison.png')
plt.show()

# Statistical Comparison of Each Method
# Original data statistics
original_mean = unstable_period['HT R Phase Current'].mean()
original_std = unstable_period['HT R Phase Current'].std()

# Winsorized data statistics
winsorized_mean = unstable_period['Current_Winsorized'].mean()
winsorized_std = unstable_period['Current_Winsorized'].std()

# Imputed data statistics
imputed_mean = unstable_period['Current_Imputed'].mean()
imputed_std = unstable_period['Current_Imputed'].std()

# Trimmed data statistics
trimmed_mean = unstable_period_trimmed['HT R Phase Current'].mean()
trimmed_std = unstable_period_trimmed['HT R Phase Current'].std()

# Display statistics
print("\nStatistical Comparison:")
print("Original Data:    Mean =", original_mean, ", Std Dev =", original_std)
print("Winsorized Data:  Mean =", winsorized_mean, ", Std Dev =", winsorized_std)
print("Imputed Data:     Mean =", imputed_mean, ", Std Dev =", imputed_std)
print("Trimmed Data:     Mean =", trimmed_mean, ", Std Dev =", trimmed_std)

# # Save cleaned data to CSV
# unstable_period.to_csv("cleaned_e6_htr_current.csv", index=False)
