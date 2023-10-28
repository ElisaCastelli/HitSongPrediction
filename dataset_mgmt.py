import pandas as pd
import numpy as np
from langdetect import detect
from analysis.csv_management import CSVManagement
import os


# from models.Model.ESCAudioPreProcessing import AudioPreProcessing


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

    def extract_languages(self):
        df_languages = pd.DataFrame(columns=['spotify_id', 'language'])
        for index, row in enumerate(self.finalDF.itertuples(), 0):
            print(index)
            track_id = row[-4]
            lyrics = row[-3]
            try:
                lang = detect(lyrics)
                print(lyrics + " " + lang)
            except Exception as e:
                lang = "NULL"
            df_languages = pd.concat([pd.DataFrame([[track_id, lang]],
                                                   columns=df_languages.columns),
                                      df_languages],
                                     ignore_index=True)
        df_languages.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/input',
                         'list_id_lang.csv'), index=True)

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
        self.finalEN = self.finalEN[self.finalEN['track_id'] != track_id]
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

    def find_info_by_id(self, id):
        try:
            title = self.df_tracks.loc[self.df_tracks['id'] == id, 'name'].iloc[0]
        except Exception as e:
            title = "NULL"
        try:
            uri_artist = self.df_tracks.loc[self.df_tracks['id'] == id, 'artists_id']
            firstP = str(uri_artist).find("'")
            uri_artist = str(uri_artist)[firstP + 1:]
            lastP = str(uri_artist).find("'")
            uri_artist = str(uri_artist)[: lastP]
            artist = self.df_artists.loc[self.df_artists['id'] == uri_artist, 'name'].iloc[0]
        except Exception as e:
            artist = "NULL"
        return title, artist

    def create_df_other_lang(self):
        lang_id = pd.read_csv("/nas/home/ecastelli/thesis/input/"
                              "list_id_lang.csv",
                              encoding='utf8', index_col=0)
        lang_es = lang_id[lang_id['language'] == 'es']
        lang_it = lang_id[lang_id['language'] == 'it']
        lang_fr = lang_id[lang_id['language'] == 'fr']
        lang_de = lang_id[lang_id['language'] == 'de']
        lang_pt = lang_id[lang_id['language'] == 'pt']
        other_lang = pd.concat([lang_es, lang_it, lang_fr, lang_de, lang_pt])
        print(len(other_lang))
        other_lang.drop_duplicates('spotify_id', inplace=True)
        print(len(other_lang))
        # self.df_tracks = self.df_tracks[['']]
        # df_final_BB = pd.merge(self.df_final_BB, self.df_bb, on='spotify_id', how='inner')
        other_lang.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/input/',
                         'SPD_other_lang.csv'), index=True)



if __name__ == "__main__":
    m = DatasetMGMT()
    m.create_df_other_lang()
