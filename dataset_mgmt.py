import pandas as pd
import numpy as np
from langdetect import detect
from analysis.csv_management import CSVManagement
import os


# TODO usa tmux per runnare sulle macchine e poi spegnere (dopo aver fatto abbstanza debug)

class DatasetMGMT:
    def __init__(self):
        # self.csv_tracks = "/nas/home/ecastelli/thesis/Data Sources/spotify_tracks.csv"
        # self.csv_albums = "/nas/home/ecastelli/thesis/Data Sources/spotify_albums.csv"
        # self.csv_artists = "/nas/home/ecastelli/thesis/Data Sources/spotify_artists.csv"
        # self.csv_ll_features = "/nas/home/ecastelli/thesis/Features Extracted/low_level_audio_features.csv"
        # self.csv_lyric_features = "/nas/home/ecastelli/thesis/Features Extracted/lyrics_features.csv"
        self.csv_tracks = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/spotify_tracks.csv"
        self.csv_albums = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/spotify_albums.csv"
        self.csv_artists = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/spotify_artists.csv"
        self.csv_ll_features = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Features Extracted/low_level_audio_features.csv"
        self.csv_lyric_features = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Features Extracted/lyrics_features.csv"
        self.finalDF_csv = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/input/df_final_year_noPodcasts.csv"
        self.csv_languages = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/languages_stats.csv"
        self.df_lang = pd.read_csv(self.csv_languages, index_col=0)
        self.df_tracks = pd.read_csv(self.csv_tracks, index_col=0)
        self.create_track_Id()
        self.df_album = pd.read_csv(self.csv_albums, index_col=0)
        self.df_artists = pd.read_csv(self.csv_artists, index_col=0)
        self.df_ll_features = pd.read_csv(self.csv_ll_features, index_col=0)
        self.df_lyrics_features = pd.read_csv(self.csv_lyric_features, index_col=0)
        self.finalDF = pd.read_csv(self.finalDF_csv, index_col=0)
        # self.dataframe = self.create_dataframe()

    def create_track_Id(self):
        self.df_tracks['uri'] = self.df_tracks['uri'].apply(
            lambda x: str(x)[str(x).rfind(':') + 1:])  # add track_id retrieving it from the uri
        self.df_tracks = self.df_tracks.rename({'uri': 'track_id'}, axis=1)

    def get_track_path(self, row):
        song_name = str(self.df_tracks.iloc[row]['track_id']) + ".mp3"
        return "/nas/home/ecastelli/thesis/Audio/" + song_name

    def popularity_analysis(self):
        # pop_series = self.df_tracks['popularity'].value_counts().sort_index(ascending=True)
        # pop_series.to_csv("Data Sources/popularity_stats.csv")
        mean_p = np.average(self.df_tracks['popularity'])
        print("Avg popularity: ", mean_p)
        std_p = np.std(self.df_tracks['popularity'])
        print("Std popularity: ", std_p)

    def extract_languages(self):
        languages = []
        ids_to_remove = []
        for index, row in self.finalDF.iterrows():
            lyrics = row['lyrics']
            try:
                lg = detect(lyrics)
                languages.append(lg)
            except Exception as e:
                ids_to_remove.append(row['track_id'])
                print(row['track_id'])
        # df_languages = pd.DataFrame(languages)
        # df_languages.to_csv("Data Sources/languages_stats.csv")
        self.remove_tracks(ids_to_remove)

    def remove_tracks(self, list_to_remove):
        print(f"{len(self.finalDF)} - {len(list_to_remove)} ")
        self.finalDF = self.finalDF[~self.finalDF['track_id'].isin(list_to_remove)]
        print(len(self.finalDF))
        self.saveDataset()

    def saveDataset(self):
        self.finalDF.to_csv(os.path.join('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/input',
                                         'df_final_year_noPodcasts.csv'))

    # TODO fai un csv anche con solo le canzoni inglesi

    def language_analysis(self):
        lang_series = self.df_lang['0'].value_counts().sort_index(ascending=True)
        lang_series.to_csv("Data Sources/lang_stats.csv")

    def save_url(self):
        df_url = self.df_tracks[['preview_url', 'uri']]
        df_url['uri'] = df_url['uri'].apply(lambda x: str(x)[str(x).rfind(':') + 1:])
        df_url = df_url.rename({'uri': 'track_id'}, axis=1)
        df_url.to_csv("./Data Sources/preview_url.csv")

    def get_lyrics_language(self, track_id):
        lyrics = self.get_info_by_id(track_id)['lyrics'].tolist()
        return detect(lyrics[0])

    def check_if_podcast(self, track_id):
        c = CSVManagement()
        podcasts = c.get_list_podcasts_id()
        if podcasts.tolist().count(track_id) > 0:
            return True
        else:
            return False

    def get_info_by_id(self, track_id):
        return self.df_tracks[self.df_tracks['track_id'] == track_id]


if __name__ == "__main__":
    m = DatasetMGMT()
    # df = m.create_dataframe_year_noComedy()
    # m.remove_empty_lyrics()
    # m.save_url()
    # print(len(m.df_tracks))
    # print(m.get_track_path(1))
    # print(m.get_lyrics_language("5qljLQuKnNJf4F4vfxQB0V"))3VAX2MJdmdqARLSU5hPMpm 1WJzRtI1ABzV3TPIeJZVvi 6aCe9zzoZmCojX7bbgKKtf 6rlEcNrUCujtmQK0EDvcp2
    m.extract_languages()
