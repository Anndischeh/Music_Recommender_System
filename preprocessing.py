import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, encoding='ISO-8859-1')
        self.recommendation_features = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                                        'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
        self.scaler = MinMaxScaler()
        self.normalized_df = self._normalize_data()

    def _normalize_data(self):
        normalized_features = self.scaler.fit_transform(self.data[self.recommendation_features])
        return pd.DataFrame(normalized_features, columns=self.recommendation_features)
