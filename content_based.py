import pandas as pd

class ContentBased:
    def __init__(self, similarity_df):
        self.similarity_df = similarity_df

    def get_song_recommendations(self, song_name, num_recommendations=5):
        if song_name not in self.similarity_df.index:
            return "Sorry, the song is not found in the dataset."
        song_similarities = self.similarity_df[song_name].sort_values(ascending=False)
        recommended_songs = song_similarities.iloc[1:num_recommendations+1].index.tolist()
        return recommended_songs
