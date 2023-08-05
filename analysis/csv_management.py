import pandas as pd
import os
from functools import reduce


class CSVManagement:
    def __init__(self):
        self.csv_tracks = "/nas/home/ecastelli/thesis/Data Sources/spotify_tracks.csv"
        self.csv_albums = "/nas/home/ecastelli/thesis/Data Sources/spotify_albums.csv"
        self.csv_artists = "/nas/home/ecastelli/thesis/Data Sources/spotify_artists.csv"
        self.csv_ll_features = "/nas/home/ecastelli/thesis/Features Extracted/low_level_audio_features.csv"
        self.csv_lyric_features = "/nas/home/ecastelli/thesis/Features Extracted/lyrics_features.csv"
        self.finalDF_csv = "/nas/home/ecastelli/thesis/input/df_final_year.csv"
        self.music_speech = pd.read_csv(
            "/nas/home/ecastelli/thesis/Data Sources/music_speech.csv")
        self.music_speech2 = pd.read_csv(
            "/nas/home/ecastelli/thesis/Data Sources/music_speech2.csv")
        self.music_speech3 = pd.read_csv(
            "/nas/home/ecastelli/thesis/Data Sources/music_speech3.csv")
        self.df_tracks = pd.read_csv(self.csv_tracks, index_col=0)
        self.create_track_Id()
        self.df_album = pd.read_csv(self.csv_albums, index_col=0)
        self.create_album_id()
        self.df_artists = pd.read_csv(self.csv_artists, index_col=0)
        self.df_ll_features = pd.read_csv(self.csv_ll_features, index_col=0)
        self.df_lyrics_features = pd.read_csv(self.csv_lyric_features, index_col=0)

    def create_track_Id(self):
        self.df_tracks['uri'] = self.df_tracks['uri'].apply(
            lambda x: str(x)[str(x).rfind(':') + 1:])  # add track_id retrieving it from the uri
        self.df_tracks = self.df_tracks.rename({'uri': 'track_id'}, axis=1)

    def create_album_id(self):
        self.df_album['uri'] = self.df_album['uri'].apply(lambda x: str(x)[str(x).rfind(':') + 1:])
        self.df_album = self.df_album.rename({'uri': 'album_id'}, axis=1)
        self.df_album['release_date'] = self.df_album['release_date'].apply(
            lambda x: int(str(x)[0:4]))  # extract year from release date
        self.df_album = self.df_album.rename({'release_date': 'year'}, axis=1)

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
        album_df = self.df_album[['album_id', 'year', 'artist_id']]
        # Add artist csv only to filter comedy
        df_artist = self.df_artists[['id', 'genres']]
        df_artist = df_artist.rename({'id': 'artist_id'}, axis=1)
        df_album_genre = album_df.merge(df_artist, on='artist_id')
        df_album_genre.drop(columns=['artist_id'], axis=1, inplace=True)
        # Drop rows of podcasts = podcasts are the tracks with artists with comedy inside genre column
        df_album_genre = df_album_genre[df_album_genre['genres'].str.contains('comedy') == False]
        df_album_genre.drop(columns=['genres'], axis=1, inplace=True)
        # Join track to obtain year and url
        df_tracks = self.df_tracks[['track_id', 'lyrics',  'album_id', 'popularity']]
        df_songs_per_year = df_tracks.merge(df_album_genre, on='album_id')
        df_songs_per_year.drop(columns=['album_id'], axis=1, inplace=True)
        print(len(df_songs_per_year))
        podcasts = self.get_list_podcasts_id()
        print(len(podcasts))
        df_songs_per_year = df_songs_per_year[~df_songs_per_year['track_id'].isin(podcasts)]
        print(len(df_songs_per_year))
        empty_lyrics = self.get_empty_lyrics()['track_id']
        empty_lyrics.tolist()
        print(len(empty_lyrics))
        df_songs_per_year = df_songs_per_year[~df_songs_per_year['track_id'].isin(empty_lyrics)]
        print(len(df_songs_per_year))
        df_songs_per_year.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/input',
                         'df_final_year_noPodcasts.csv'), index=False)

    def create_dataframe_no_year(self):
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

    def get_list_podcasts_id(self):
        file_to_remove = pd.concat([self.music_speech[self.music_speech['result'] == 'Speech'],
                                    self.music_speech2[self.music_speech2['result'] == 'Speech'],
                                    self.music_speech3[self.music_speech3['result'] == 'Speech']], ignore_index=False,
                                   sort=False)
        ids = file_to_remove['idSong'].apply(
            lambda x: str(x)[:str(x).rfind('.')])
        ids.tolist()
        return ids

    def get_empty_lyrics(self):
        mask = ((self.df_tracks['lyrics'].str.len() < 5) | (self.df_tracks['lyrics'].str.len() == 96) | (self.df_tracks['lyrics'].str.len() > 19000))
        empty_lyrics = self.df_tracks[mask]
        return empty_lyrics


if __name__ == "__main__":
    c = CSVManagement()
    print(len(c.df_tracks[c.df_tracks['country']=='BE']))
