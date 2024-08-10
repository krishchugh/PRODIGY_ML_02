import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ["OMP_NUM_THREADS"] = "1"

df = pd.read_csv('customer_purchase_history.txt')

customer_agg = df.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'count'],
    'product_id': 'nunique'
}).reset_index()

customer_agg.columns = ['customer_id', 'total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products']

scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_agg[['total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products']])

inertia_values = []
cluster_range = range(1, 11)  

for k in cluster_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42)
    kmeans_model.fit(scaled_data)
    inertia_values.append(kmeans_model.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

optimal_clusters = 4
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_agg['cluster'] = kmeans_model.fit_predict(scaled_data)

sns.pairplot(customer_agg, hue='cluster', vars=['total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products'])
plt.suptitle('Customer Segments', y=1.02)
plt.show()

cluster_centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
cluster_summary = pd.DataFrame(cluster_centers, columns=['total_spent', 'avg_spent_per_purchase', 'num_purchases', 'unique_products'])
print(cluster_summary)

customer_agg.to_csv('customer_segments.csv', index=False)