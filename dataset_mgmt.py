import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import display
from functools import reduce

class Dataset_MGMT:
    def __init__(self):
        self.csv_tracks = "./Data Sources/spotify_tracks.csv"
        self.csv_albums = "./Data Sources/spotify_albums.csv"
        self.csv_ll_features = "./Features Extracted/low_level_audio_features.csv"
        self.csv_lyric_features = "./Features Extracted/lyrics_features.csv"
        self.df_tracks = pd.read_csv(self.csv_tracks, index_col=0)
        self.df_album = pd.read_csv(self.csv_albums, index_col=0)
        self.df_ll_features = pd.read_csv(self.csv_ll_features, index_col=0)
        self.df_lyrics_features = pd.read_csv(self.csv_lyric_features, index_col=0)
        self.dataframe = self.create_dataframe()

    def parameter_analysis(self, dataset, parameter):
        dataset[parameter].value_counts().sort_index(ascending=True).plot(kind = 'bar')
        mean_p = np.average(dataset[parameter])
        print("Avg popularity: ",mean_p)
        std_p = np.std(dataset[parameter])
        print("Std popularity: ",std_p)

    def create_dataframe(self):
        
         # Join spotify_tracks e spotify_albums on album_id
        
        album_df = self.df_album[['uri','release_date']]
        album_df['uri'] = album_df['uri'].apply(lambda x: str(x)[str(x).rfind(':')+1:])
        album_df = album_df.rename({'uri': 'album_id'}, axis=1)
        album_df['release_date'] = album_df['release_date'].apply(lambda x: int(str(x)[0:4])) #extract year from release date
        album_df = album_df.rename({'release_date': 'year'}, axis=1)

        df_songs_per_year = self.df_tracks.merge(album_df, on='album_id')

        # Join merge track per year, low level features and lyrics features
        df_songs_per_year['uri'] = df_songs_per_year['uri'].apply(lambda x: str(x)[str(x).rfind(':')+1:]) #add track_id retrieving it from the uri
        df_songs_per_year = df_songs_per_year.rename({'uri': 'track_id'}, axis=1)
        df_join = [df_songs_per_year, self.df_ll_features,
                           self.df_lyrics_features]
        
        df_final = reduce(lambda left, right: pd.merge(left, right, on='track_id'), df_join)
        df_final.to_csv(os.path.join('./input', 'df_final.csv'), index=False)
        print(df_final.columns)
        #return df_final


if __name__ == "__main__":
    m = Dataset_MGMT()
    m.create_dataframe()