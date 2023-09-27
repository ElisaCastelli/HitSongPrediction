import pandas as pd
import os
import numpy as np
import librosa
# import spotipy
# from langdetect import detect
# import re
import time

# from spotipy.oauth2 import SpotifyClientCredentials

# import requests
import musixmatch

CLIENT_ID = "40d965faf5b34d1186992f5e244f521a"
CLIENT_SECRET = "9ed8db794cf3449482f3c3f937e4e5c9"
# MAU
# apikey = 'ad204f5f01118594e11b2b1a31950b58'


# IO
# apikey = '7bd05e631df3a816c47098c9746ce29c'
# ILA
# apikey = '7bb608d74a7dd09548133753de4e633b'
apikey = 'c60c484c515c41690adf73a20f1607b4'


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


AUDIO_DIR_SPD = "/nas/home/ecastelli/thesis/Audio/"


class BBDataFrame():
    def __init__(self):
        pass
        # self.csv_bb_64_15 = "/nas/home/ecastelli/thesis/Billboard/CSV/songs_w_features_lyrics_year.csv"
        # self.csv_bb_00_18 = "/nas/home/ecastelli/thesis/Billboard/CSV/billboard_2000_2018_spotify_lyrics.csv"
        # self.csv_my_bb = "/nas/home/ecastelli/thesis/Billboard/CSV/bb_lyrics_1964_2018.csv"
        # self.csv_popularity_url = "/nas/home/ecastelli/thesis/Billboard/CSV/bb_pop_url_DEF.csv"
        # self.df_bb = pd.read_csv(self.csv_my_bb, encoding='utf8', index_col=0)
        # self.df_pop = pd.read_csv(self.csv_popularity_url, encoding='utf8', index_col=0)
        # self.df_bb_64_15 = pd.read_csv(self.csv_bb_64_15, encoding='utf8', index_col=0, encoding_errors='ignore', on_bad_lines='skip')
        # self.df_bb_00_18 = pd.read_csv(self.csv_bb_00_18, encoding='utf8', index_col=0, encoding_errors='ignore', on_bad_lines='skip')
        # self.df_old = pd.read_csv("/nas/home/ecastelli/thesis/input/FINAL_en.csv", encoding='utf8')
        self.old_songs = pd.read_csv("/nas/home/ecastelli/thesis/input/df_final_year.csv",
                                     encoding='utf8')
        self.old_artist = pd.read_csv("/nas/home/ecastelli/thesis/Data Sources/spotify_artists.csv",
                                      encoding='utf8')
        # self.df_final_BB = pd.read_csv("/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/"
        #                                "Billboard/CSV/bb_FINAL.csv",
        #                                encoding='utf8', index_col=0)
        self.BB_english = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/BB_english.csv",
                                      encoding='utf8', index_col=0)
        self.bb_spd_en = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/BB_SPD_en_updated_no_dup.csv",
                                     encoding='utf8', index_col=0)
        self.spd_en = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/SPD_en_no_dup.csv",
                                  encoding='utf8', index_col=0)
        self.spd_other_lang = pd.read_csv("/nas/home/ecastelli/thesis/input/SPD_other_lang.csv",
                                          encoding='utf8', index_col=0)
        self.spd_other_mxm = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/SPD_other_mxm_nd.csv",
                                         encoding='utf8', index_col=0)

    def concat_csv(self):
        df_bb_64_15 = self.df_bb_64_15[['URI', 'lyrics', 'Release_Year']]
        df_bb_64_15.rename(columns={'URI': 'spotify_id', 'Release_Year': 'release_year'}, inplace=True)
        df_bb_00_18 = self.df_bb_00_18[['spotify_id', 'lyrics', 'date']]
        df_bb_00_18['date'] = df_bb_00_18['date'].apply(
            lambda x: str(x)[str(x).rfind('/') + 1:])
        for index, row in enumerate(df_bb_00_18.itertuples(), 0):
            if int(row[3]) > 24:
                new_date = str(19) + row[3]
            else:
                new_date = str(20) + row[3]
            df_bb_00_18.iloc[index, df_bb_00_18.columns.get_loc('date')] = new_date
        df_bb_00_18.rename({'date': 'release_year'}, axis=1, inplace=True)
        self.df_bb = pd.concat([df_bb_00_18, df_bb_64_15])
        self.save_csv()

    def keep_unique_id(self):
        self.df_bb = self.df_bb.drop_duplicates('spotify_id')

    def remove_empty_col(self):
        self.df_bb.dropna(subset=['lyrics'], inplace=True)
        self.df_bb['spotify_id'].replace('', np.nan, inplace=True)
        self.df_bb.dropna(subset=['spotify_id'], inplace=True)

    def print_state(self):
        print(f"Total number of rows: {len(self.df_bb)} \nColumns: {self.df_bb.columns}")

    def save_csv(self):
        self.df_bb = self.df_bb[['spotify_id', 'lyrics']]  # TODO ADD release_year
        self.df_bb.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'Total_lyrics.csv'), index=True)

    def get_audio_features(self, track_id):
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                                   client_secret=CLIENT_SECRET))
        print(sp.audio_features(track_id))

    def get_track_info(self, track_id):
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                                   client_secret=CLIENT_SECRET))
        track_info = sp.track(track_id)
        print(track_info['popularity'])

    def concat_df(self):
        print(f"Length old dataframe: {len(self.df_old)}")
        print(f"Length new dataframe: {len(self.df_pop)}")
        df_final = pd.concat([self.df_pop, self.df_old])
        df_final.drop_duplicates('spotify_id', inplace=True)
        print(f"Length union dataframe: {len(df_final)}")

    def get_lyrics_by_id(self):
        df_bb_64_15 = self.df_bb_64_15[['URI', 'lyrics']]
        df_bb_64_15.rename(columns={'URI': 'spotify_id'}, inplace=True)
        df_bb_00_18 = self.df_bb_00_18[['spotify_id', 'lyrics']]
        self.df_bb = pd.concat([df_bb_00_18, df_bb_64_15])
        self.save_csv()

    def substitute_lyrics(self):
        self.df_final_BB = self.df_final_BB.drop(columns=['lyrics'])
        df_final_BB = pd.merge(self.df_final_BB, self.df_bb, on='spotify_id', how='inner')
        df_final_BB.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'bb_FINAL.csv'), index=True)

    def create_pop_df(self):
        df_popularity_preview = pd.DataFrame(columns=['spotify_id', 'popularity'])
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
                                                                   client_secret=CLIENT_SECRET))
        for index, row in enumerate(self.spd_en.itertuples(), 0):
            track_id = row[1]
            print(index)
            try:
                track_info = sp.track(track_id)
                df_popularity_preview = pd.concat([pd.DataFrame([[track_id, track_info['popularity'],
                                                                  ]],
                                                                columns=df_popularity_preview.columns),
                                                   df_popularity_preview],
                                                  ignore_index=True)
                time.sleep(10)
            except Exception as e:
                print(e)
        df_popularity_preview.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'spd_pop_UPDATED.csv'), index=True)

    def pop_statistics(self):
        print(f"Tot: {len(self.df_pop)}")
        self.df_pop['popularity'].replace('', np.nan, inplace=True)
        self.df_pop.dropna(subset=['popularity'], inplace=True)
        print(f"Drop empty popularity: {len(self.df_pop)}")
        self.df_pop['preview_url'].replace('', np.nan, inplace=True)
        self.df_pop.dropna(subset=['preview_url'], inplace=True)
        print(f"Drop empty preview_url: {len(self.df_pop)}")
        self.df_pop.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'bb_pop_url_DEF.csv'), index=True)

    def popularity_analysis(self):
        spd_all = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/SPD_all_lang_not_updated.csv",
                              encoding='utf8')
        pop_series = spd_all['popularity'].value_counts().sort_index(ascending=True)
        print(pop_series)
        pop_series.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'SPD_all_lang_popularity_stats.csv'), index=True)
        mean_p = np.average(spd_all['popularity'])
        print("Avg popularity: ", mean_p)
        std_p = np.std(spd_all['popularity'])
        print("Std popularity: ", std_p)

    def popularity_class_analysis(self):
        pop_series = self.df_final_BB['pop_class'].value_counts().sort_index(ascending=True)
        print(pop_series)

    def create_def_BB(self):
        df_final_BB = pd.merge(self.df_pop, self.df_bb, on='spotify_id', how='inner')
        print(len(df_final_BB))
        df_final_BB.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'bb_FINAL.csv'), index=True)

    def add_popularity_class(self):
        for index, row in enumerate(self.df_final_BB.itertuples(), 0):
            print(index)
            popularity = int(row[-5])
            pop_class = get_class(popularity)
            self.df_final_BB.at[index, 'pop_class'] = pop_class
        self.df_final_BB.to_csv(
            os.path.join('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/CSV/',
                         'bb_FINAL.csv'), index=False)

    def find_info_by_id(self, id):
        try:
            title = self.old_songs.loc[self.old_songs['id'] == id, 'name'].iloc[0]
        except Exception as e:
            title = "NULL"
        try:
            uri_artist = self.old_songs.loc[self.old_songs['id'] == id, 'artists_id']
            firstP = str(uri_artist).find("'")
            uri_artist = str(uri_artist)[firstP + 1:]
            lastP = str(uri_artist).find("'")
            uri_artist = str(uri_artist)[: lastP]
            artist = self.old_artist.loc[self.old_artist['id'] == uri_artist, 'name'].iloc[0]
        except Exception as e:
            artist = "NULL"
        return title, artist

    def add_popularity(self):
        self.spd_other_mxm['popularity'] = ""
        for index, row in enumerate(self.spd_other_mxm.itertuples()):
            track_id = row[-7]
            try:
                pop = self.old_songs.loc[self.old_songs['id'] == track_id, 'popularity'].iloc[0]
            except Exception as e:
                pop = "NULL"
            self.spd_other_mxm.at[index, 'popularity'] = pop
        self.spd_other_mxm['popularity'].replace('NULL', np.nan, inplace=True)
        self.spd_other_mxm.dropna(subset=['lyrics'], inplace=True)
        self.spd_other_mxm.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_other_mxm_nd.csv'), index=True)

    def add_year(self):
        for index, row in enumerate(self.spd_other_mxm.itertuples()):
            track_id = row[-8]
            try:
                year = self.old_songs.loc[self.old_songs['id'] == track_id, 'year'].iloc[0]
            except Exception as e:
                year = "NULL"
            self.spd_other_mxm.at[index, 'year'] = year
        self.spd_other_mxm['year'].replace('NULL', np.nan, inplace=True)
        self.spd_other_mxm.dropna(subset=['year'], inplace=True)
        self.spd_other_mxm.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_other_mxm_nd.csv'), index=True)



    def remove_nan_lyrics(self):
        print(len(self.spd_other_mxm))
        self.spd_other_mxm['lyrics'].replace('NULL', np.nan, inplace=True)
        self.spd_other_mxm.dropna(subset=['lyrics'], inplace=True)
        self.spd_other_mxm.drop('spotify_id', inplace=True)
        print(len(self.spd_other_mxm))
        self.spd_other_mxm.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_other_mxm_nd.csv'), index=True)

    def get_lyrics_from_x(self, title, artist):
        MxM = musixmatch.Musixmatch(apikey=apikey)
        try:
            response = MxM.matcher_track_get(q_track=title, q_artist=artist)
            tracXid = response['message']['body']['track']['track_id']
            lyrics = MxM.track_lyrics_get(tracXid)
            lyrics = lyrics['message']['body']['lyrics']['lyrics_body']
        except Exception as e:
            tracXid = 0
            lyrics = "NULL"
            print(e)
        return tracXid, lyrics

    def retrieve_lyrics(self):
        rows = []
        print(len(self.spd_other_mxm))
        for index, row in enumerate(self.spd_other_lang[20000:].iterrows()):
            print(index)
            track_id = row[1]['spotify_id']
            title, artist = self.find_info_by_id(track_id)
            MxM_id, lyrics = self.get_lyrics_from_x(title, artist)
            rows.append({
                'spotify_id': track_id,
                'musiXmatch_id': MxM_id,
                'title': title,
                'artist': artist,
                'lyrics': lyrics,
            })
        df_lyriXmatch = pd.DataFrame(rows)
        df_lyriXmatch = pd.concat([self.spd_other_mxm, df_lyriXmatch])
        print(len(df_lyriXmatch))
        df_lyriXmatch.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_other_mxm.csv'), index=True)
        print("Done!")

    def remove_final_quote(self):
        for index, row in enumerate(self.spd_other_mxm.itertuples()):
            lyrics = str(row[-1])
            end_p = lyrics.find("...\n\n******* This Lyrics is NOT for Commercial use *******")
            lyrics = lyrics[:end_p]
            self.spd_other_mxm.at[index, 'lyrics'] = lyrics
        self.spd_other_mxm.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_other_mxm_nd.csv'), index=True)
        print(len(self.spd_other_mxm))

    def remove_duplicate(self):
        print(len(self.spd_en))
        self.spd_en.drop_duplicates(subset=['musiXmatch_id'], inplace=True)
        print(len(self.spd_en))
        self.spd_en.to_csv(
            os.path.join('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/CSV',
                         'SPD_en_no_dup.csv'), index=True)

    def get_length(self):
        print(len(self.spd_en))

    def filter_no_lyrics(self):
        print(len(self.spd_other_mxm))
        self.spd_other_mxm['lyrics'].replace("", np.nan, inplace=True)
        new_data = self.spd_other_mxm.dropna(subset=['lyrics'])
        new_data.drop_duplicates(subset=['spotify_id'], inplace=True)
        print(len(new_data))
        new_data.drop_duplicates(subset=['musiXmatch_id'], inplace=True)
        print(len(new_data))
        new_data.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_other_mxm.csv'), index=True)

    def merge_old(self):
        final_en = self.df_old[['track_id', 'year', 'popularity']]
        final_en.rename(columns={'track_id': 'spotify_id'}, inplace=True)
        mxm = self.old_MxM[['spotify_id', 'musiXmatch_id', 'title', 'artist', 'lyrics']]
        final_mxm = pd.merge(final_en, mxm, on='spotify_id', how='inner')
        final_mxm.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'final_mxm.csv'), index=True)

    def concat_BB_mxm(self):
        bb = self.BB_english[['spotify_id', 'lyrics', 'release_year_x', 'popularity']]
        print(len(bb))
        bb.rename(columns={'release_year_x': 'year'}, inplace=True)
        mxm = self.spd_en[['spotify_id', 'lyrics', 'year', 'popularity']]
        print(len(mxm))
        final_complete = pd.concat([bb, mxm])
        final_complete.drop_duplicates(subset=['spotify_id'], inplace=True)
        print(len(final_complete))
        final_complete.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'BB_SPD_en_updated_no_dup.csv'), index=True)

    def get_language(self):
        df_SPD_lang = pd.DataFrame(columns=['spotify_id', 'language'])
        for index, row in enumerate(self.df_final_BB.itertuples(), 0):
            print(index)
            track_id = row[-7]
            lyrics = row[-2]
            try:
                lang = detect(lyrics)
            except Exception as e:
                lang = "NULL"
            df_SPD_lang = pd.concat([pd.DataFrame([[track_id, lang]],
                                                  columns=df_SPD_lang.columns),
                                     df_SPD_lang],
                                    ignore_index=True)
        df_SPD_lang.to_csv(
            os.path.join('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/CSV',
                         'BB_LANG.csv'), index=True)

    def get_lang_stats(self):
        spd_lang = pd.read_csv("/nas/home/ecastelli/thesis/input/"
                               "list_id_lang.csv",
                               encoding='utf8', index_col=0)
        lang_series = spd_lang['language'].value_counts().sort_index(ascending=True)
        print(lang_series)
        # lang_series.to_csv(
        #     os.path.join('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/CSV',
        #                  'BB_language_stats.csv'), index=True)

    def keep_by_language(self, language='en'):
        spd_lang = pd.read_csv("/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/"
                               "CSV/BB_LANG.csv",
                               encoding='utf8', index_col=0)
        print(len(spd_lang))
        bb_en = pd.merge(spd_lang, self.df_final_BB, on='spotify_id', how='inner')
        print(len(bb_en))
        bb_en = bb_en[bb_en['language'] == language]
        print(len(bb_en))
        bb_en.to_csv(
            os.path.join('/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/CSV',
                         'BB_english.csv'), index=True)

    def get_audio_length(self):
        df_songs_length = pd.DataFrame(columns=['spotify_id', 'length'])
        for index, row in enumerate(self.spd_en.itertuples()):
            spotify_id = row[1]
            file = str(spotify_id) + ".mp3"
            audio_sample_path = os.path.join(AUDIO_DIR_SPD, file)
            audio, sr = librosa.load(audio_sample_path, sr=22050)
            length = len(audio)
            df_songs_length = pd.concat([pd.DataFrame([[spotify_id, length,
                                                        ]],
                                                      columns=df_songs_length.columns),
                                         df_songs_length],
                                        ignore_index=True)
        df_songs_length.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'SPD_length.csv'), index=True)

    def stats_length(self):
        spd_length = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/SPD_length.csv",
                                 encoding='utf8', index_col=0)
        lang_series = spd_length['length'].value_counts().sort_index(ascending=True)
        print(lang_series)
        lang_series.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_length_stats.csv'), index=True)

    def remove_el(self):
        print(len(self.spd_en))
        self.spd_en = self.spd_en[self.spd_en['spotify_id'] != "6vOnTS8EhcUqvaRwsV2Dfn"]
        print(len(self.spd_en))
        self.spd_en.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'SPD_en_no_dup.csv'), index=True)

    def substitute_popularity(self):
        spd_pop = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/spd_pop_UPDATED.csv",
                              encoding='utf8', index_col=0)
        spd = self.spd_en[['spotify_id', 'lyrics', 'year']]
        spd_en = pd.merge(spd_pop, spd, on='spotify_id', how='inner')
        spd_en.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'SPD_english_pop_updated.csv'), index=True)

    def final_dataset(self):
        spd_pop = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/SPD_english_pop_updated.csv",
                              encoding='utf8', index_col=0)
        bb_en = self.BB_english[['spotify_id', 'lyrics', 'release_year_x', 'popularity']]
        bb_en.rename(columns={'release_year_x': 'year'}, inplace=True)
        final = pd.concat([spd_pop, bb_en])
        final.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'SPD_BB_english.csv'), index=True)

    def drop_duplicates(self):
        spd_pop = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/SPD_english_pop_updated.csv",
                              encoding='utf8', index_col=0)
        print(len(spd_pop))
        spd_pop.drop_duplicates('spotify_id', inplace=True)
        print(len(spd_pop))
        spd_pop.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'SPD_english_pop_updated_no_dup.csv'), index=True)


if __name__ == '__main__':
    dataframe = BBDataFrame()
    dataframe.popularity_analysis()
    # other_lang = dataframe.spd_other_mxm[['spotify_id', 'musiXmatch_id', 'year', 'lyrics', 'popularity' ]]
    # english = dataframe.spd_en[['spotify_id', 'musiXmatch_id', 'year', 'lyrics', 'popularity' ]]
    # total = pd.concat([english, other_lang])
    # total.drop_duplicates('spotify_id', inplace=True)
    # total.to_csv(
    #             os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
    #                          'SPD_all_lang_not_updated.csv'), index=True)



