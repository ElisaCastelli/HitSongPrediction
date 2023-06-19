import pandas as pd
import os

song_df = pd.read_csv(r"models/music_speech.csv")
song_df2 = pd.read_csv(r"models/music_speech2.csv")
song_df3 = pd.read_csv(r"models/music_speech3.csv")
def create_clean_dataframe():
    speech = len(song_df[song_df['result']=='Speech'])+len(song_df2[song_df2['result']=='Speech'])+len(song_df3[song_df3['result']=='Speech'])
    print(f"{speech} file to remove")
    print(f"{len(song_df)+len(song_df2)+len(song_df3)} total files")
    song_final = pd.concat([song_df[song_df['result']!='Speech'], song_df2[song_df2['result']!='Speech'], song_df3[song_df3['result']!='Speech']],ignore_index=False, sort=False)
    #print(len(song_df[song_df['result']!='Speech'])+len(song_df2[song_df2['result']!='Speech']))
    print(f"{len(song_final)} final file")
    #TO DO: SAVE DATAFRAME ON A NEW CSV
    song_final['idSong'] = song_final['idSong'].apply(lambda x: str(x)[:str(x).rfind('.')])
    song_final=song_final.drop(columns=['idN'])
    song_final.to_csv(r"C:\Users\Utente\Documents\GitHub\HitSongPrediction\Data Sources\id_music_tracks.csv")
    #print(song_final.iloc[1]['idSong'].substr)

def remove_speech_files():
    file_to_remove = pd.concat([song_df[song_df['result']=='Speech'], song_df2[song_df2['result']=='Speech']],ignore_index=False, sort=False)
    for index, file in file_to_remove.iterrows():
        data_dir = "C:\Audio/"
        if os.path.exists(data_dir+file['idSong']):
          os.remove(data_dir+file['idSong'])
        else:
          print("The file does not exist")

def remove_empty_lyrics():
    music_tracks = pd.read_csv(r"Data Sources/id_music_tracks.csv")
    print(len(music_tracks))
    no_lyrics_tracks = pd.read_csv(r"Data Sources/tracks_without_lyrics.csv")
    no_lyrics_list = list(no_lyrics_tracks['id'])
    print(no_lyrics_list)
    music_tracks = music_tracks[~music_tracks['idSong'].isin(no_lyrics_list)]
    print(len(music_tracks))
    music_tracks.to_csv("Data Sources/id_music_tracks_with_lyrics.csv")

#create_clean_dataframe()
#remove_speech_files()
remove_empty_lyrics()