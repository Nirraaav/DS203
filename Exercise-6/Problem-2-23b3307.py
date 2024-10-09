import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.io as pio
import os

# Create a directory to store images
if not os.path.exists("Images"):
    os.makedirs("Images")

# Create a directory to store csv files
if not os.path.exists("CSV"):  
    os.makedirs("CSV")

# Load Data
df = pd.read_csv('e6-Run2-June22-subset-100-cols.csv')

# Initial Exploration
print(df.head(10))
print(df.describe())
print(df.isnull().sum())

# Clean non-numeric values
df.replace('#REF!', np.nan, inplace=True)

# Step 1: Compute variance of each column (ignoring the timestamp column)
variances = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').var()

# Step 2: Set a threshold for low variance (adjust this threshold as per your data)
low_variance_columns = variances[variances < 0.05].index  # Columns with variance less than 0.05

# Step 3: Drop these low variance columns from the DataFrame
df = df.drop(columns=low_variance_columns)

# Step 4: Optionally, inspect the dropped columns
print(f"Columns dropped due to low variance: {low_variance_columns}")
print(f"Number of columns dropped due to low variance: {len(low_variance_columns)}")

# Data Cleaning
# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Detect and handle outliers
# Using IQR method for simplicity
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_cols] < (Q1 - 2 * IQR)) | (df[numeric_cols] > (Q3 + 2 * IQR))).any(axis=1)]

# Normalization/Standardization
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# Correlation Analysis
corr_matrix = df_scaled.corr()
fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
fig.update_layout(title='Correlation Heatmap', autosize=False, width=1000, height=800)
# Save the figure
# pio.write_image(fig, "Images/correlation_heatmap.png", width=800, height=600, scale=3)  # scale=3 to achieve approx 300 DPI
fig.show()

# Handle correlated columns
# Assuming a threshold of 0.8 for high correlation
high_corr = np.where(np.abs(corr_matrix) > 0.8)
high_corr = [(corr_matrix.columns[x], corr_matrix.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]

# Drop one of each pair of highly correlated columns
columns_to_drop = list(set([y for x, y in high_corr]))
columns_to_drop = [col for col in columns_to_drop if col in df_scaled.columns]  # Ensure columns are in DataFrame
df_scaled.drop(columns=columns_to_drop, axis=1, inplace=True)

# Print the number of columns dropped due to high correlation
print(f"Columns dropped due to high correlation: {columns_to_drop}")
print(f"Number of columns dropped due to high correlation: {len(columns_to_drop)}")

# Multicollinearity Analysis
# Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = df_scaled.columns
vif_data["VIF"] = [variance_inflation_factor(df_scaled.values, i) for i in range(len(df_scaled.columns))]
print("\nVIF analysis:")
print(vif_data)
# Save the VIF data in csv
vif_data.to_csv("CSV/vif_data.csv", index=False)

# Handle multicollinear columns
# Assuming a VIF threshold of 10
columns_to_drop_vif = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
columns_to_drop_vif = [col for col in columns_to_drop_vif if col in df_scaled.columns]  # Ensure columns are in DataFrame
df_scaled.drop(columns=columns_to_drop_vif, axis=1, inplace=True)

# Print the number of columns dropped due to high VIF
print(f"Columns dropped due to high VIF: {columns_to_drop_vif}")
print(f"Number of columns dropped due to high VIF: {len(columns_to_drop_vif)}")

# --- Stage i: PCA after Step 'c' ---
# PCA after normalization/standardization
pca_initial = PCA()
df_pca_initial = pca_initial.fit_transform(df_scaled)
explained_variance_initial = pca_initial.explained_variance_ratio_

# Elbow diagram for PCA after step 'c'
fig_initial = go.Figure()
fig_initial.add_trace(go.Scatter(x=list(range(1, len(explained_variance_initial) + 1)), y=np.cumsum(explained_variance_initial), mode='lines+markers'))
fig_initial.update_layout(title='Elbow Diagram after Step c (Before VIF Handling)', xaxis_title='Number of Components', yaxis_title='Cumulative Explained Variance')
# pio.write_image(fig_initial, "Images/elbow_diagram_after_step_c.png", width=800, height=600, scale=3)  # save image
fig_initial.show()

# --- Stage ii: PCA after Step 'g' ---
# PCA after handling correlated and multicollinear columns
pca_final = PCA()
df_pca_final = pca_final.fit_transform(df_scaled)
explained_variance_final = pca_final.explained_variance_ratio_

# Elbow diagram for PCA after step 'g'
fig_final = go.Figure()
fig_final.add_trace(go.Scatter(x=list(range(1, len(explained_variance_final) + 1)), y=np.cumsum(explained_variance_final), mode='lines+markers'))
fig_final.update_layout(title='Elbow Diagram after Step g (After VIF Handling)', xaxis_title='Number of Components', yaxis_title='Cumulative Explained Variance')
# pio.write_image(fig_final, "Images/elbow_diagram_after_step_g.png", width=800, height=600, scale=3)  # save image
fig_final.show()

# Final data output
print("\nFinal dataset shape after discarding columns:", df_scaled.shape)
print("\nFinal dataset columns:", df_scaled.columns.tolist())

# Analysis and Interpretation
def interpret_elbow_diagram(explained_variance, title):
    cumulative_variance = np.cumsum(explained_variance)
    optimal_components = np.argmax(cumulative_variance >= 0.95) + 1  # Find the number of components explaining 95% variance
    print(f"For '{title}', optimal number of components to explain 95% variance: {optimal_components}")
    print(f"Cumulative Variance Explained: {cumulative_variance[optimal_components - 1]:.2f}")

# Interpret both diagrams
interpret_elbow_diagram(explained_variance_initial, 'Stage i: After Step c')
interpret_elbow_diagram(explained_variance_final, 'Stage ii: After Step g')
