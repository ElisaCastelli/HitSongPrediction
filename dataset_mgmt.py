import pandas as pd
import numpy as np
import os
from functools import reduce


class DatasetMGMT:
    def __init__(self):
        self.csv_tracks = "./Data Sources/spotify_tracks.csv"
        self.csv_albums = "./Data Sources/spotify_albums.csv"
        self.csv_artists = "./Data Sources/spotify_artists.csv"
        self.csv_ll_features = "./Features Extracted/low_level_audio_features.csv"
        self.csv_lyric_features = "./Features Extracted/lyrics_features.csv"
        self.df_tracks = pd.read_csv(self.csv_tracks, index_col=0)
        self.df_album = pd.read_csv(self.csv_albums, index_col=0)
        self.df_artists = pd.read_csv(self.csv_artists, index_col=0)
        self.df_ll_features = pd.read_csv(self.csv_ll_features, index_col=0)
        self.df_lyrics_features = pd.read_csv(self.csv_lyric_features, index_col=0)
        # self.dataframe = self.create_dataframe()

    def popularity_analysis(self):
        # pop_series = self.df_tracks['popularity'].value_counts().sort_index(ascending=True)
        # pop_series.to_csv("Data Sources/popularity_stats.csv")
        mean_p = np.average(self.df_tracks['popularity'])
        print("Avg popularity: ", mean_p)
        std_p = np.std(self.df_tracks['popularity'])
        print("Std popularity: ", std_p)

    def save_url(self):
        df_url = self.df_tracks[['preview_url', 'uri']]
        df_url['uri'] = df_url['uri'].apply(lambda x: str(x)[str(x).rfind(':') + 1:])
        df_url = df_url.rename({'uri': 'track_id'}, axis=1)
        df_url.to_csv("./Data Sources/preview_url.csv")

    def create_dataframe_year(self):
        # Join spotify_tracks e spotify_albums on album_id

        album_df = self.df_album[['uri', 'release_date']]
        album_df['uri'] = album_df['uri'].apply(lambda x: str(x)[str(x).rfind(':') + 1:])
        album_df = album_df.rename({'uri': 'album_id'}, axis=1)
        album_df['release_date'] = album_df['release_date'].apply(
            lambda x: int(str(x)[0:4]))  # extract year from release date
        album_df = album_df.rename({'release_date': 'year'}, axis=1)

        df_songs_per_year = self.df_tracks.merge(album_df, on='album_id')

        # Join merge track per year, low level features and lyrics features
        df_songs_per_year['uri'] = df_songs_per_year['uri'].apply(
            lambda x: str(x)[str(x).rfind(':') + 1:])  # add track_id retrieving it from the uri
        df_songs_per_year = df_songs_per_year.rename({'uri': 'track_id'}, axis=1)
        df_join = [df_songs_per_year, self.df_ll_features,
                   self.df_lyrics_features]

        df_final = reduce(lambda left, right: pd.merge(left, right, on='track_id'), df_join)
        df_final.to_csv(os.path.join('./input', 'df_final_year.csv'), index=False)
        print(df_final.columns)

    def create_dataframe_year_noComedy(self):
        # Join spotify_tracks e spotify_albums on album_id

        album_df = self.df_album[['uri', 'release_date', 'artist_id']]
        album_df['uri'] = album_df['uri'].apply(lambda x: str(x)[str(x).rfind(':') + 1:])
        album_df = album_df.rename({'uri': 'album_id'}, axis=1)
        album_df['release_date'] = album_df['release_date'].apply(
            lambda x: int(str(x)[0:4]))  # extract year from release date
        album_df = album_df.rename({'release_date': 'year'}, axis=1)
        df_artist = self.df_artists[['id', 'genres']]
        df_artist = df_artist.rename({'id': 'artist_id'}, axis=1)
        df_album_genre = album_df.merge(df_artist, on='artist_id')
        df_album_genre.drop(columns=['artist_id'], axis=1, inplace=True)
        # df_album_genre = df_album_genre.drop(columns='artist_id',axis=1, inplace=True)
        print(df_album_genre.columns)
        # Drop rows of podcasts = podcasts are the tracks with artists with comedy inside genre column
        # df_album_genre[~df_album_genre.genres.str.contains('comedy')]
        df_album_genre = df_album_genre[df_album_genre['genres'].str.contains('comedy') == False]
        df_songs_per_year = self.df_tracks.merge(df_album_genre, on='album_id')
        print(df_songs_per_year.columns)

        # Join merge track per year, low level features and lyrics features
        df_songs_per_year['uri'] = df_songs_per_year['uri'].apply(
            lambda x: str(x)[str(x).rfind(':') + 1:])  # add track_id retrieving it from the uri
        df_songs_per_year = df_songs_per_year.rename({'uri': 'track_id'}, axis=1)

        df_join = [df_songs_per_year, self.df_ll_features,
                   self.df_lyrics_features]

        df_final = reduce(lambda left, right: pd.merge(left, right, on='track_id'), df_join)
        print(df_final.columns)
        df_final.to_csv(os.path.join('./input', 'df_final_year_noComedy.csv'), index=False)

    def remove_empty_lyrics(self):
        # id_lyrics_songs = set(self.df_tracks['track_id'])
        id_lyrics_lyrics = set(self.df_lyrics_features['track_id'])
        id_lyrics_songs = set(self.df_tracks['uri'].apply(
            lambda x: str(x)[str(x).rfind(':') + 1:]))  # add track_id retrieving it from the uri
        listEmpty = list(id_lyrics_songs - id_lyrics_lyrics)
        d = pd.DataFrame(listEmpty)
        d.to_csv("Data Sources/tracks_without_lyrics.csv")
        print(len(listEmpty))

    def create_dataframe(self):
        # Join merge track per year, low level features and lyrics features
        self.df_tracks['uri'] = self.df_tracks['uri'].apply(
            lambda x: str(x)[str(x).rfind(':') + 1:])  # add track_id retrieving it from the uri
        self.df_tracks = self.df_tracks.rename({'uri': 'track_id'}, axis=1)
        df_join = [self.df_tracks, self.df_ll_features,
                   self.df_lyrics_features]

        df_final = reduce(lambda left, right: pd.merge(left, right, on='track_id'), df_join)
        df_final.to_csv(os.path.join('./input', 'df_final_noyear.csv'), index=False)
        print(df_final.columns)
        # return df_final


if __name__ == "__main__":
    m = DatasetMGMT()
    # df = m.create_dataframe_year_noComedy()
    # m.remove_empty_lyrics()
    # m.save_url()
    print(len(m.df_tracks))
