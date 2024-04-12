from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, normalized_df):
        self.normalized_df = normalized_df

    def determine_optimal_clusters(self):
        inertia_values = []
        cluster_range = range(1, 15)

        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.normalized_df)
            inertia_values.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, inertia_values, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method to Determine Optimal Number of Clusters')
        plt.grid(True)
        plt.show()

    def apply_kmeans(self, num_clusters=4):
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.normalized_df)
        return clusters
