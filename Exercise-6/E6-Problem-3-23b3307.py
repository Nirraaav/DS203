import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST test dataset
mnist_test = pd.read_csv('mnist_test.csv')

# Separate features from labels (assuming 'label' is the column name for labels)
X = mnist_test.drop('label', axis=1)
y = mnist_test['label']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Elbow Diagram
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='--')
plt.title('Elbow Diagram - Cumulative Explained Variance by PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.savefig('Images/P3-elbow_diagram.png', dpi=300)
plt.show()

# Scatter plot of PC2 vs PC1
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
plt.title('PCA Scatter Plot (PC2 vs PC1)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Label')
plt.grid(True)
plt.savefig('Images/P3-pca_scatter_plot.png', dpi=300)
plt.show()

from sklearn.manifold import TSNE

# Apply t-SNE to reduce data to 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# t-SNE Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=10, alpha=0.7)
plt.title('t-SNE Scatter Plot')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Label')
plt.grid(True)
plt.savefig('Images/P3-tsne_scatter_plot.png', dpi=300)
plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: Load your data
e6_data = pd.read_csv('e6-Run2-June22-subset-100-cols.csv')

# Initial Exploration
print(e6_data.head(10))
print(e6_data.describe())
print(e6_data.isnull().sum())

# Replace '#REF!' with NaN
e6_data.replace('#REF!', np.nan, inplace=True)

# Step 2: Convert the first column to datetime
e6_data.iloc[:, 0] = pd.to_datetime(e6_data.iloc[:, 0], format='%d-%m-%Y')

# Step 3: Convert the datetime column to float (timestamp)
e6_data['timestamp'] = e6_data.iloc[:, 0].apply(lambda x: x.timestamp())

# Step 4: Drop the original datetime column if it's no longer needed
e6_data = e6_data.drop(columns=[e6_data.columns[0]])  # Adjust the index if necessary

# Step 5: Apply the same preprocessing steps
X_e6 = e6_data.drop('label', axis=1, errors='ignore')  # Assuming a 'label' column may or may not exist

# Check for NaN values in X_e6
print("NaN values in the dataset before imputation:")
print(X_e6.isnull().sum())

# Option 1: Drop rows with NaN values
X_e6 = X_e6.dropna()

# Option 2: Or fill NaN values with the mean (you can choose median or another method)
# X_e6.fillna(X_e6.mean(), inplace=True)

# Scale the features
scaler = StandardScaler()
X_e6_scaled = scaler.fit_transform(X_e6)

# Step 6: Apply t-SNE on this data
tsne = TSNE(n_components=2, random_state=42)  # You can adjust parameters as needed
X_e6_tsne = tsne.fit_transform(X_e6_scaled)

# Step 7: t-SNE Scatter Plot for e6 dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_e6_tsne[:, 0], X_e6_tsne[:, 1], cmap='viridis', s=10, alpha=0.7)
plt.title('t-SNE Scatter Plot for e6 Dataset')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.savefig('Images/P3-tsne_scatter_plot_e6.png', dpi=300)
plt.show()
