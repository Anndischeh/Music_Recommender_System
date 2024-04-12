from sklearn.neighbors import NearestNeighbors
import pandas as pd

class KNN:
    def __init__(self, knn_model, scaler, normalized_df, spotify_data):
        self.knn_model = knn_model
        self.scaler = scaler
        self.normalized_df = normalized_df
        self.spotify_data = spotify_data

    def get_recommendations_knn_song(self, song_name, num_recommendations=5):
        if song_name not in self.spotify_data['track_name'].values:
            return "Sorry, the song is not found in the dataset."
        song_data = self.normalized_df[self.spotify_data['track_name'] == song_name]
        distances, indices = self.knn_model.kneighbors(song_data, n_neighbors=num_recommendations+1)
        recommended_songs = self.spotify_data['track_name'].iloc[indices[0][1:]].tolist()
        return recommended_songs

    def get_recommendations_knn_features(self, input_features, num_recommendations=5):
        input_df = pd.DataFrame([input_features])
        normalized_input = self.scaler.transform(input_df)
        distances, indices = self.knn_model.kneighbors(normalized_input, n_neighbors=num_recommendations)
        recommended_songs = self.spotify_data['track_name'].iloc[indices[0]].tolist()
        return recommended_songs
