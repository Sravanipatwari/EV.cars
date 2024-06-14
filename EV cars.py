import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load the dataset
file_path = r'C:\Users\hp\Downloads\ev.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
print(df.head())

# Display basic information about the dataframe
print(df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Display summary statistics
print(df.describe(include='all'))

# Handle missing values if any
df = df.dropna()  # Simplest approach: drop rows with any missing values

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Verify the encoding
print(df.head())

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
scaled_df_minmax = minmax_scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
scaled_df_minmax = pd.DataFrame(scaled_df_minmax, columns=df.columns)

# Elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_df_minmax)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans clustering with the optimal number of clusters determined from the elbow method
optimal_clusters = 4  # Assuming 4 from your initial code
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_df_minmax)

# Add cluster labels to the scaled data
scaled_df_minmax['Cluster'] = clusters

# Analyze each cluster
for cluster in range(optimal_clusters):
    print(f"Cluster {cluster} Description")
    print(scaled_df_minmax[scaled_df_minmax['Cluster'] == cluster].describe())
    print("\n")

# Summarize the clusters
cluster_summary = scaled_df_minmax.groupby('Cluster').mean()
print(cluster_summary)

# Summarize the counts of each cluster
cluster_counts = scaled_df_minmax['Cluster'].value_counts()
print(cluster_counts)

# Plot the cluster counts
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Number of Data Points per Cluster')
plt.show()

# Age distribution in each cluster
plt.figure(figsize=(10, 5))
for cluster in range(optimal_clusters):
    subset = scaled_df_minmax[scaled_df_minmax['Cluster'] == cluster]
    subset['Age'].plot(kind='density', label=f'Cluster {cluster}', alpha=0.6)
plt.title('Age Distribution Across Clusters')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# Income distribution in each cluster
plt.figure(figsize=(10, 5))
for cluster in range(optimal_clusters):
    subset = scaled_df_minmax[scaled_df_minmax['Cluster'] == cluster]
    subset['Income'].plot(kind='density', label=f'Cluster {cluster}', alpha=0.6)
plt.title('Income Distribution Across Clusters')
plt.xlabel('Income')
plt.ylabel('Density')
plt.legend()
plt.show()

# Distribution of Age and Income per Cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for cluster in range(optimal_clusters):
    subset = scaled_df_minmax[scaled_df_minmax['Cluster'] == cluster]
    axes[cluster].scatter(subset['Age'], subset['Income'], label=f'Cluster {cluster}', alpha=0.6)
    axes[cluster].set_title(f'Cluster {cluster}')
    axes[cluster].set_xlabel('Age')
    axes[cluster].set_ylabel('Income')

plt.suptitle('Distribution of Age and Income per Cluster')
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Add the original 'Price of Car' to the scaled dataframe for plotting
scaled_df_minmax['Price of Car'] = df['Price of Car']

# Plot the box plot of Price of Car by Cluster
plt.figure(figsize=(12, 8))
scaled_df_minmax.boxplot(column='Price of Car', by='Cluster', grid=False)
plt.xlabel('Cluster')
plt.ylabel('Price of Car')
plt.title('Price of Car by Cluster')
plt.suptitle('')  # Suppress the default title
plt.show()
