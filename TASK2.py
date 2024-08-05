import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the environment variable to avoid memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Data Collection and Preparation
# Load the dataset from the text file
df = pd.read_csv('customer_purchase_history.txt')

# Step 2: Feature Engineering
# Create aggregated features for each customer
customer_agg = df.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'count'],
    'product_id': 'nunique'
}).reset_index()

# Rename columns for better readability
customer_agg.columns = ['customer_id', 'total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products']

# Standardize the feature values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_agg[['total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products']])

# Step 3: Implementing K-means Clustering
# Determine the optimal number of clusters using the Elbow method
inertia_values = []
cluster_range = range(1, 11)  # Adjusted range for a larger dataset

for k in cluster_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(scaled_data)
    inertia_values.append(kmeans_model.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 5))
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Select the optimal number of clusters, for instance, k=4
optimal_clusters = 4
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_agg['cluster'] = kmeans_model.fit_predict(scaled_data)

# Step 4: Analyzing and Interpreting Clusters
# Visualize the clustering results
sns.pairplot(customer_agg, hue='cluster', vars=['total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products'])
plt.suptitle('Customer Segments', y=1.02)
plt.show()

# Understanding cluster characteristics
cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
cluster_summary = pd.DataFrame(cluster_centers, columns=['total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products'])
print(cluster_summary)

# Optional: Save the clustering results to a CSV file
customer_agg.to_csv('customer_segments.csv', index=False)
