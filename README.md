# Music Recommender System

This music recommender system leverages various content-based machine learning algorithms to provide personalized song suggestions.

## Introduction
In this repository, we focus on implementing a music recommender system using the dataset ["Most Streamed Spotify Songs 2023"](https://www.kaggle.com/datasets/rajatsurana979/most-streamed-spotify-songs-2023). The dataset encompasses valuable information such as track name, artist(s) name, release date, Spotify playlists and charts, streaming statistics, Apple Music presence, Deezer presence, Shazam charts, and various audio features. Our content-based recommender system suggests songs similar to the given input by analyzing these features. We preprocess the data by performing feature normalization on selected recommendation features, ensuring they are on a comparable scale for accurate machine learning model training.

## Methods 
### KNN (K-Nearest Neighbors)
This KNN-based music recommendation algorithm utilizes a pre-trained model and feature scaling to provide personalized song suggestions. By measuring the similarity between songs, it efficiently identifies tracks aligned with user preferences. Users can receive recommendations based on either specific song inputs or feature vectors, enhancing their music discovery experience.

![KNN](http://url/to/img.png)

### KMeans
The KMeans algorithm determines the optimal number of clusters using the Elbow Method, ensuring accurate grouping of songs. It identifies the ideal clustering configuration and applies KMeans clustering to categorize songs based on their features, enabling personalized music recommendations aligned with user preferences.

![KMeans](http://url/to/img.png)

### GCN (Graph Convolutional Network)
The GCN model for music recommendation processes graph-structured data to infer relationships between songs. With input, hidden, and output layers, it learns representations of songs based on their features and the connections between them. By leveraging adjacency matrices, it captures intricate relationships within the music dataset, enabling accurate recommendation predictions.

![GCN](http://url/to/img.png)

### NN (Neural Network)
The Neural Network model for music recommendation utilizes a feedforward neural network architecture. With input, hidden, and output layers, it learns complex patterns within music data to make personalized recommendations. By applying nonlinear activation functions, such as ReLU, it captures intricate relationships between input features, enhancing recommendation accuracy.

![NN](http://url/to/img.png)

## Example 
### KNN Recommendations based on sample features:
- BPM: 90
- Danceability: 70%
- Valence: 65%
- Energy: 75%
- Acousticness: 10%
- Instrumentalness: 5%
- Liveness: 15%
- Speechiness: 5%

Recommended Songs:
1. SPIT IN MY FACE!
2. LALA
3. Besos Moja2
4. Marisola - Remix
5. En La De Ella

### Content-Based Recommendations for "En La De Ella":
1. Blank Space
2. Besos Moja2
3. Wait a Minute!
4. LALA
5. Pantysito



