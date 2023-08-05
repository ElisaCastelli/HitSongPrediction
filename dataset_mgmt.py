import pandas as pd
import numpy as np
# from langdetect import detect
from analysis.csv_management import CSVManagement
import os
from models.Resnet.ESCAudioPreProcessing import AudioPreProcessing

# TODO usa tmux per runnare sulle macchine e poi spegnere (dopo aver fatto abbstanza debug)


def get_class(popularity):
    if popularity < 25:
        pop_class = 0
    elif 25 <= popularity < 50:
        pop_class = 1
    elif 50 <= popularity < 75:
        pop_class = 2
    else:
        pop_class = 3
    return pop_class


class DatasetMGMT:
    def __init__(self):
        self.csv_tracks = "/nas/home/ecastelli/thesis/Data Sources/spotify_tracks.csv"
        self.csv_albums = "/nas/home/ecastelli/thesis/Data Sources/spotify_albums.csv"
        self.csv_artists = "/nas/home/ecastelli/thesis/Data Sources/spotify_artists.csv"
        self.csv_ll_features = "/nas/home/ecastelli/thesis/Features Extracted/low_level_audio_features.csv"
        self.csv_lyric_features = "/nas/home/ecastelli/thesis/Features Extracted/lyrics_features.csv"
        self.csv_languages = "/nas/home/ecastelli/thesis/Data Sources/languages_stats.csv"
        self.finalDF_csv = "/nas/home/ecastelli/thesis/input/df_final_year_noPodcasts.csv"
        self.finalDFen_csv = "/nas/home/ecastelli/thesis/input/df_final_english.csv"
        self.english_id_csv = "/nas/home/ecastelli/thesis/Data Sources/english_tracks_id.csv"
        # self.csv_tracks = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/spotify_tracks.csv"
        # self.csv_albums = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/spotify_albums.csv"
        # self.csv_artists = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/spotify_artists.csv"
        # self.csv_ll_features = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Features Extracted/low_level_audio_features.csv"
        # self.csv_lyric_features = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Features Extracted/lyrics_features.csv"
        # self.finalDF_csv = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/input/df_final_year_noPodcasts.csv"
        # self.csv_languages = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Data Sources/languages_stats.csv"
        self.df_lang = pd.read_csv(self.csv_languages, index_col=0)
        self.df_tracks = pd.read_csv(self.csv_tracks, index_col=0)
        self.create_track_Id()
        self.df_album = pd.read_csv(self.csv_albums, index_col=0)
        self.df_artists = pd.read_csv(self.csv_artists, index_col=0)
        self.df_ll_features = pd.read_csv(self.csv_ll_features, index_col=0)
        self.df_lyrics_features = pd.read_csv(self.csv_lyric_features, index_col=0)
        self.finalDF = pd.read_csv(self.finalDF_csv, on_bad_lines='skip')
        self.finalEN = pd.read_csv(self.finalDFen_csv, index_col=0)
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

    # def extract_languages(self):
    #     languages = []
    #     ids_to_remove = []
    #     for index, row in self.finalDF.iterrows():
    #         lyrics = row['lyrics']
    #         try:
    #             lg = detect(lyrics)
    #             languages.append(lg)
    #         except Exception as e:
    #             ids_to_remove.append(row['track_id'])
    #             print(row['track_id'])
    #             print(row['lyrics'])
    #     # df_languages = pd.DataFrame(languages)
    #     # df_languages.to_csv("Data Sources/languages_stats.csv")
    #     self.remove_tracks(ids_to_remove)

    def remove_tracks(self, list_to_remove):
        print(f"{len(self.finalDF)} - {len(list_to_remove)} ")
        self.finalDF = self.finalDF[~self.finalDF['track_id'].isin(list_to_remove)]
        print(len(self.finalDF))
        self.saveDataset()

    # def extract_ids_by_lang(self, searched_lang):
    #     tracks_id = []
    #     for index, row in self.finalDF.iterrows():
    #         lyrics = row['lyrics']
    #         try:
    #             lg = detect(lyrics)
    #             if lg == searched_lang:
    #                 tracks_id.append(row['track_id'])
    #         except Exception as e:
    #             print(e)
    #     df_english_track = pd.DataFrame(tracks_id)
    #     df_english_track.to_csv(os.path.join('/nas/home/ecastelli/thesis/Data Sources', 'english_tracks_id.csv'))
    #     self.save_songs_from_id_list(tracks_id)
    #     print("Save songs ...")

    def save_songs_from_id_list(self):
        tracks_id = pd.read_csv(self.english_id_csv)['0'].tolist()
        df_english = self.finalDF[self.finalDF['track_id'].isin(tracks_id)]
        df_english.to_csv(os.path.join('/nas/home/ecastelli/thesis/input',
                                       'df_final_english.csv'))
        print("Done!")

    def saveDataset(self):
        self.finalDF.to_csv(os.path.join('/nas/home/ecastelli/thesis/input',
                                         'df_final_year_noPodcasts.csv'))

    def saveENDataset(self):
        self.finalEN.to_csv(os.path.join('/nas/home/ecastelli/thesis/input',
                                         'df_final_english.csv'))

    def language_analysis(self):
        lang_series = self.df_lang['0'].value_counts().sort_index(ascending=True)
        lang_series.to_csv("Data Sources/lang_stats.csv")

    def save_url(self):
        df_url = self.df_tracks[['preview_url', 'uri']]
        df_url['uri'] = df_url['uri'].apply(lambda x: str(x)[str(x).rfind(':') + 1:])
        df_url = df_url.rename({'uri': 'track_id'}, axis=1)
        df_url.to_csv("./Data Sources/preview_url.csv")

    # def get_lyrics_language(self, track_id):
    #     lyrics = self.get_info_by_id(track_id)['lyrics'].tolist()
    #     return detect(lyrics[0])

    def check_if_podcast(self, track_id):
        c = CSVManagement()
        podcasts = c.get_list_podcasts_id()
        if podcasts.tolist().count(track_id) > 0:
            return True
        else:
            return False

    def get_info_by_id(self, track_id):
        return self.df_tracks[self.df_tracks['track_id'] == track_id]

    def get_info_main_dataset(self):
        print(f"Length dataset: {len(self.finalDF)} \nColumn names: {self.finalDF.columns}")

    def get_info_en_dataset(self):
        print(f"Length dataset: {len(self.finalEN)} \nColumn names: {self.finalEN.columns}")

    def remove_song(self, track_id):
        print(len(self.finalEN))
        self.finalEN = self.finalEN[self.finalEN['track_id']!=track_id]
        print(len(self.finalEN))
        self.saveENDataset()

    def save_all_spectrograms(self):
        for index, row in self.finalDF[24071:].iterrows():
            try:
                track_id = row['track_id']
                p = AudioPreProcessing(track_id)
                p.get_save_spectrogram()
            except Exception as e:
                print(row['track_id'])
        print("Done!")

    def add_popularity_class(self):
        for index, row in enumerate(self.finalEN.itertuples(), 0):
            popularity = int(row[-2])
            pop_class = get_class(popularity)
            self.finalEN['pop_class'] = pop_class
        self.finalEN.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/input/',
                         'FINAL_en.csv'), index=True)


if __name__ == "__main__":
    m = DatasetMGMT()
    m.add_popularity_class()
