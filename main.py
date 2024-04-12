import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from preprocessing import Preprocessing
from content_based import ContentBased
from knn import KNN
from kmeans_clustering import KMeansClustering
from plotting import Plotting
from gcn import GCN
from neural_network import NeuralNetworkRecommender 
import torch

# Load data and preprocess
spotify_data_path = '/content/spotify-2023.csv'
preprocessing = Preprocessing(spotify_data_path)
similarity_matrix = cosine_similarity(preprocessing.normalized_df)
similarity_df = pd.DataFrame(similarity_matrix, index=preprocessing.data['track_name'], columns=preprocessing.data['track_name'])
content_based = ContentBased(similarity_df)

# Initialize KNN model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
knn_model.fit(preprocessing.normalized_df)
knn = KNN(knn_model, preprocessing.scaler, preprocessing.normalized_df, preprocessing.data)

# Perform K-means clustering
kmeans = KMeansClustering(preprocessing.normalized_df)
kmeans.determine_optimal_clusters()
clusters = kmeans.apply_kmeans()

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(preprocessing.normalized_df)

# Plot principal components
Plotting.scatter_plot_principal_components(principal_components)

# Plot principal components with clusters
Plotting.scatter_plot_with_clusters(principal_components, clusters)

# Example of using GCN
input_dim = len(preprocessing.recommendation_features)
hidden_dim = 64
output_dim = len(clusters)
adj_matrix = np.random.rand(len(preprocessing.normalized_df), len(preprocessing.normalized_df))  
gcn_model = GCN(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(gcn_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Example of forward pass
x = torch.tensor(preprocessing.normalized_df.values, dtype=torch.float32)
adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
output = gcn_model(x, adj_matrix)
print(output)
Plotting.plot_cluster_graph(principal_components, clusters)

# Initialize the recommender
input_dim_nn = len(preprocessing.recommendation_features)
hidden_dim_nn = 64
output_dim_nn = len(preprocessing.data)  # Number of songs in the dataset
nn_recommender = NeuralNetworkRecommender(input_dim_nn, hidden_dim_nn, output_dim_nn)

# Convert normalized data to tensor
input_features_tensor = torch.tensor(preprocessing.normalized_df.values, dtype=torch.float32)

# Forward pass through the neural network
recommendations = nn_recommender(input_features_tensor)

# Assuming recommendations are scores for each song
num_recommendations_nn = 5
# Get the top recommended songs
top_recommendations_idx = recommendations.argsort(descending=True)[:num_recommendations_nn].tolist()
top_recommendations_idx = torch.tensor(top_recommendations_idx)  
top_recommendations_idx = top_recommendations_idx.squeeze().tolist() 
top_recommendations = [preprocessing.data['track_name'].iloc[idx] for idx in top_recommendations_idx]

print("Top Recommendations from Neural Network Recommender:")
print(top_recommendations)
