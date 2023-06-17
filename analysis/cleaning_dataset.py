import pandas as pd
import os

song_df = pd.read_csv("/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/models/music_speech.csv")
song_df2 = pd.read_csv("/Users/elisacastelli/Documents/GitHub/HitPrediction/HitSongPrediction/models/music_speech2.csv")
def create_clean_dataframe():
    speech = len(song_df[song_df['result']=='Speech'])+len(song_df2[song_df2['result']=='Speech'])
    print(f"{speech} file to remove")
    print(f"{len(song_df)+len(song_df2)} total files")
    song_final = pd.concat([song_df[song_df['result']!='Speech'], song_df2[song_df2['result']!='Speech']],ignore_index=False, sort=False)
    #print(len(song_df[song_df['result']!='Speech'])+len(song_df2[song_df2['result']!='Speech']))
    print(f"{len(song_final)} final file")
    #TO DO: SAVE DATAFRAME ON A NEW CSV

def remove_speech_files():
    file_to_remove = pd.concat([song_df[song_df['result']=='Speech'], song_df2[song_df2['result']=='Speech']],ignore_index=False, sort=False)
    for index, file in file_to_remove.iterrows():
        
        if os.path.exists(file['idSong']):
          os.remove(file['idSong'])
        else:
          print("The file does not exist")