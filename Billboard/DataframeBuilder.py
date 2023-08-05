import pandas as pd
import os
import numpy as np
import spotipy
import re
import time
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import musixmatch

CLIENT_ID = "40d965faf5b34d1186992f5e244f521a"
CLIENT_SECRET = "9ed8db794cf3449482f3c3f937e4e5c9"
# apikey = 'ad204f5f01118594e11b2b1a31950b58' MAU

# apikey = '7bd05e631df3a816c47098c9746ce29c'IO
apikey = '7bb608d74a7dd09548133753de4e633b'

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


class BBDataFrame():
    def __init__(self):
        self.csv_bb_64_15 = "/nas/home/ecastelli/thesis/Billboard/CSV/songs_w_features_lyrics_year.csv"
        self.csv_bb_00_18 = "/nas/home/ecastelli/thesis/Billboard/CSV/billboard_2000_2018_spotify_lyrics.csv"
        self.csv_my_bb = "/nas/home/ecastelli/thesis/Billboard/CSV/bb_lyrics_1964_2018.csv"
        self.csv_popularity_url = "/nas/home/ecastelli/thesis/Billboard/CSV/bb_pop_url_DEF.csv"
        # self.csv_my_bb = "/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/Billboard/CSV/bb_lyrics_1964_2018.csv"
        self.df_bb = pd.read_csv(self.csv_my_bb, encoding='unicode_escape', index_col=0)
        self.df_pop = pd.read_csv(self.csv_popularity_url, encoding='unicode_escape', index_col=0)
        self.df_bb_64_15 = pd.read_csv(self.csv_bb_64_15, encoding='unicode_escape', index_col=0)
        self.df_bb_00_18 = pd.read_csv(self.csv_bb_00_18, encoding='unicode_escape', index_col=0)
        # self.df_popularity_preview = pd.DataFrame(columns=['spotify_id', 'popularity', 'preview_url'])
        # self.df_bb.reset_index(inplace=True)
        # self.df_bb = self.df_bb.drop(self.df_bb.columns[1], axis=1)
        # self.df_bb = self.df_bb[['spotify_id', 'lyrics', 'release_year']]
        self.df_old = pd.read_csv("/nas/home/ecastelli/thesis/input/FINAL_en.csv", encoding='unicode_escape')
        self.old_songs = pd.read_csv("/nas/home/ecastelli/thesis/Data Sources/spotify_tracks.csv",
                                     encoding='unicode_escape')
        self.old_artist = pd.read_csv("/nas/home/ecastelli/thesis/Data Sources/spotify_artists.csv",
                                      encoding='unicode_escape')
        self.df_final_BB = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/bb_FINAL.csv",
                                       encoding='unicode_escape')
        self.old_MxM = pd.read_csv("/nas/home/ecastelli/thesis/Billboard/CSV/Old_MxM.csv",
                                   encoding='unicode_escape', index_col=0)

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

    def keep_unique_id(self):
        self.df_bb = self.df_bb.drop_duplicates('spotify_id')

    def remove_empty_col(self):
        self.df_bb.dropna(subset=['lyrics'], inplace=True)
        self.df_bb['spotify_id'].replace('', np.nan, inplace=True)
        self.df_bb.dropna(subset=['spotify_id'], inplace=True)

    def print_state(self):
        print(f"Total number of rows: {len(self.df_bb)} \nColumns: {self.df_bb.columns}")

    def save_csv(self):
        self.df_bb = self.df_bb[['spotify_id', 'lyrics', 'release_year']]
        self.df_bb.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV',
                         'bb_lyrics_1964_2018.csv'), index=True)

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
        df_final.drop_duplicates('spotify_id')
        print(f"Length union dataframe: {len(df_final)}")

    # def create_pop_df(self):
    #     sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID,
    #                                                                client_secret=CLIENT_SECRET))
    #     # self.df_popularity_preview.columns = ['spotify_id', 'popularity', 'preview_url']
    #     for index, row in enumerate(self.df_bb.itertuples(), 0):
    #         track_id = row[2]
    #         print(track_id)
    #         try:
    #             track_info = sp.track(track_id)
    #             self.df_popularity_preview = pd.concat([pd.DataFrame([[track_id, track_info['popularity'],
    #                                                                    track_info['preview_url']]],
    #                                                                  columns=self.df_popularity_preview.columns),
    #                                                     self.df_popularity_preview],
    #                                                    ignore_index=True)
    #             time.sleep(10)
    #         except Exception as e:
    #             print(e)
    #     self.df_popularity_preview.to_csv(
    #         os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
    #                      'bb_pop_preview.csv'), index=True)

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
        pop_series = self.df_pop['popularity'].value_counts().sort_index(ascending=True)
        print(pop_series)
        mean_p = np.average(self.df_pop['popularity'])
        print("Avg popularity: ", mean_p)
        std_p = np.std(self.df_pop['popularity'])
        print("Std popularity: ", std_p)

    def create_def_BB(self):
        df_final_BB = pd.merge(self.df_pop, self.df_bb, on='spotify_id', how='inner')
        print(len(df_final_BB))
        df_final_BB.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'bb_FINAL.csv'), index=True)

    def add_popularity_class(self):
        for index, row in enumerate(self.df_final_BB.itertuples(), 0):
            popularity = int(row[4])
            pop_class = get_class(popularity)
            self.df_final_BB['pop_class'] = pop_class
        self.df_final_BB.to_csv(
            os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'bb_FINAL.csv'), index=True)

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
        # df_lyriXmatch = pd.DataFrame()
        rows = []
        for index, row in enumerate(self.df_old[6401:8350].iterrows()):
            print(index)
            track_id = row[1]['track_id']
            title, artist = self.find_info_by_id(track_id)
            # MxM_id, lyrics = "None", "None"
            MxM_id, lyrics = self.get_lyrics_from_x(title, artist)
            rows.append({
                'spotify_id': track_id,
                'musiXmatch_id': MxM_id,
                'title': title,
                'artist': artist,
                'lyrics': lyrics,
            })
        df_lyriXmatch = pd.DataFrame(rows)
        df_lyriXmatch = pd.concat([self.old_MxM, df_lyriXmatch])
        df_lyriXmatch.to_csv(os.path.join('/nas/home/ecastelli/thesis/Billboard/CSV/',
                         'Old_MxM.csv'), index=True)
        print("Done!")

    def remove_final_quote(self):
        for index, row in enumerate(self.old_MxM.iterrows()):
            lyrics = str(row[-3])
            end_p = lyrics.find("******* This Lyrics is NOT for Commercial use *******")
            lyrics = lyrics[:end_p]
            self.old_MxM[index][-3] = lyrics




if __name__ == '__main__':
    dataframe = BBDataFrame()
    # dataframe.concat_csv()
    # dataframe.print_state()
    # dataframe.keep_unique_id()
    # dataframe.print_state()
    # dataframe.remove_empty_col()
    # dataframe.print_state()
    # dataframe.save_csv()
    # print(dataframe.df_bb)
    # dataframe.get_audio_features("2Yl4OmDby9iitgNWZPwxkd")
    # title, artist = dataframe.find_info_by_id("5qljLQuKnNJf4F4vfxQB0V")
    # dataframe.get_lyrics_from_x(title, artist)
    dataframe.retrieve_lyrics()
